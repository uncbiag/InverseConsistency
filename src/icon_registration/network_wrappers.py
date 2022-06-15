import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .mermaidlite import compute_warped_image_multiNC, identity_map_multiN


class RegistrationModule(nn.Module):
    r"""Base class for icon modules that perform registration.

    A subclass of RegistrationModule should have a forward method that
    takes as input two images image_A and image_B, and returns a python function
    phi_AB that transforms a tensor of coordinates.

    RegistrationModule provides a method as_function that turns a tensor
    representing an image into a python function mapping a tensor of coordinates
    into a tensor of intensities R^N -> R.
    Mathematically, this is what an image is anyway.

    After this class is constructed, but before it is used, you _must_ call
    assign_identity_map on it or on one of its parents to define the coordinate
    system associated with input images.

    The contract that a successful registration fulfils is:
    for a tensor of coordinates X, self.as_function(image_A)(phi_AB(X)) ~= self.as_function(image_B)(X)

    ie

    $$ I^A \circ \Phi^{AB} \simeq I^B $$

    In particular, self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B
    """

    def __init__(self):
        super().__init__()
        self.downscale_factor = 1

    def as_function(self, image):
        """image is a tensor with shape self.input_shape.
        Returns a python function that maps a tensor of coordinates [batch x N_dimensions x ...]
        into a tensor of intensities.
        """

        return lambda coordinates: compute_warped_image_multiNC(
            image, coordinates, self.spacing, 1
        )

    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = np.array(input_shape)
        self.input_shape[0] = 1
        self.spacing = 1.0 / (self.input_shape[2::] - 1)

        # if parents_identity_map is not None:
        #    self.identity_map = parents_identity_map
        # else:
        _id = identity_map_multiN(self.input_shape, self.spacing)
        self.register_buffer("identity_map", torch.from_numpy(_id))

        if self.downscale_factor != 1:
            child_shape = np.concatenate(
                [
                    self.input_shape[:2],
                    np.ceil(self.input_shape[2:] / self.downscale_factor).astype(int),
                ]
            )
        else:
            child_shape = self.input_shape
        for child in self.children():
            if isinstance(child, RegistrationModule):
                child.assign_identity_map(
                    child_shape,
                    # None if self.downscale_factor != 1 else self.identity_map,
                )

    def adjust_batch_size(self, size):
        shape = self.input_shape
        shape[0] = size
        self.assign_identity_map(shape)


class FunctionFromVectorField(RegistrationModule):
    """
    Wrap an inner neural network 'net' that returns a tensor of displacements
    [B x N x H x W (x D)], into a RegistrationModule that returns a function that
    transforms a tensor of coordinates
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        vectorfield_phi = self.net(image_A, image_B)

        def ret(input_):
            if hasattr(input_, "isIdentity") and vectorfield_phi.shape == input_.shape:
                return input_ + vectorfield_phi
            else:
                return input_ + compute_warped_image_multiNC(
                    vectorfield_phi, input_, self.spacing, 1
                )

        return ret


def multiply_matrix_vectorfield(matrix, vectorfield):
    dimension = len(vectorfield.shape) - 2
    if dimension == 2:
        batch_matrix_multiply = "ijkl,imj->imkl"
    else:
        batch_matrix_multiply = "ijkln,imj->imkln"
    return torch.einsum(batch_matrix_multiply, vectorfield, matrix)


class FunctionFromMatrix(RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B)

        def ret(input_):
            shape = list(input_.shape)
            shape[1] = 1
            input_homogeneous = torch.cat(
                [input_, torch.ones(shape, device=input_.device)], axis=1
            )
            return multiply_matrix_vectorfield(matrix_phi, input_homogeneous)[:, :-1]

        return ret


class RandomShift(RegistrationModule):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, image_A, image_B):
        shift_shape = (
            image_A.shape[0],
            len(image_A.shape) - 2,
            *(1 for _ in image_A.shape[2:]),
        )
        # In a real class, the variable that parametrizes the returned transform,
        # in this case shift, would be calculated from image_A and image_B before being captured
        # in the closure as below.
        shift = self.stddev * torch.randn(shift_shape, device=image_A.device)
        return lambda input_: input_ + shift


class TwoStepRegistration(RegistrationModule):
    """Combine two RegistrationModules.

    First netPhi is called on the input images, then image_A is warped with
    the resulting field, and then netPsi is called on warped A and image_B
    in order to find a residual warping. Finally, the composition of the two
    transforms is returned.
    """

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):
        # Tag for optimization. Must be set at the beginning of forward because it is not preserved by .to(config.device)
        self.identity_map.isIdentity = True
        phi = self.netPhi(image_A, image_B)
        phi_vectorfield = phi(self.identity_map)
        self.image_A_comp_phi = self.as_function(image_A)(phi_vectorfield)
        psi = self.netPsi(self.image_A_comp_phi, image_B)

        ret = lambda input_: phi(psi(input_))
        return ret


class DownsampleRegistration(RegistrationModule):
    """
    Perform registration using the wrapped RegistrationModule `net`
    at half input resolution.
    """

    def __init__(self, net, dimension):
        super().__init__()
        self.net = net
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension
        # This member variable is read by assign_identity_map when
        # walking the network tree and assigning identity_maps
        # to know that all children of this module operate at a lower
        # resolution.
        self.downscale_factor = 2

    def forward(self, image_A, image_B):

        image_A = self.avg_pool(image_A, 2, ceil_mode=True)
        image_B = self.avg_pool(image_B, 2, ceil_mode=True)
        return self.net(image_A, image_B)


### DEPRECATED
def warninfo(message):
    from inspect import getframeinfo, stack
    import warnings

    caller = getframeinfo(stack()[2][0])
    warnings.warn("%s:%d - %s" % (caller.filename, caller.lineno, message))


def assignIdentityMap(net, size):
    warninfo("assignIdentityMap is deprecated. use net.assign_identity_map")
    net.assign_identity_map(size)


def adjust_batch_size(net, N):
    warninfo(
        "adjust_batch_size is deprecated. Batch size is now determined at runtime from input shape"
    )


DoubleNet = TwoStepRegistration
DownsampleNet = DownsampleRegistration
