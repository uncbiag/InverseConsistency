from collections import namedtuple

import matplotlib
import torch
import torchvision.transforms.functional_tensor as F_t

from icon_registration import config, network_wrappers

from .mermaidlite import compute_warped_image_multiNC


def to_floats(stats):
    out = []
    for v in stats:
        if isinstance(v, torch.Tensor):
            v = torch.mean(v).cpu().item()
        out.append(v)
    return ICONLoss(*out)


ICONLoss = namedtuple(
    "ICONLoss",
    "all_loss inverse_consistency_loss similarity_loss transform_magnitude flips",
)


class InverseConsistentNet(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def __call__(self, image_A, image_B) -> ICONLoss:
        return super().__call__(image_A, image_B)

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        inbounds_tag = torch.zeros(tuple(image_A.shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        elif len(self.input_shape) - 2 == 2:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1),
            self.phi_AB_vectorfield,
            self.spacing,
            1,
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1),
            self.phi_BA_vectorfield,
            self.spacing,
            1,
        )

        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)

        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(config.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # inverse consistency one way

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )


class GradientICON(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def compute_gradient_icon_loss(self, phi_AB, phi_BA):
        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(config.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = phi_AB(phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = phi_AB(phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        return inverse_consistency_loss

    def compute_similarity_measure(self, phi_AB, phi_BA, image_A, image_B):
        self.phi_AB_vectorfield = phi_AB(self.identity_map)
        self.phi_BA_vectorfield = phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        inbounds_tag = torch.zeros(tuple(image_A.shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        elif len(self.input_shape) - 2 == 2:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1)
        )(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(
            torch.cat([image_B, inbounds_tag], axis=1)
        )(self.phi_BA_vectorfield)
        similarity_loss = self.similarity(
            self.warped_image_A, image_B
        ) + self.similarity(self.warped_image_B, image_A)
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        similarity_loss = self.compute_similarity_measure(
            self.phi_AB, self.phi_BA, image_A, image_B
        )

        inverse_consistency_loss = self.compute_gradient_icon_loss(
            self.phi_AB, self.phi_BA
        )

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_BA_vectorfield),
        )


def normalize(image):
    dimension = len(image.shape) - 2
    if dimension == 2:
        dim_reduce = [2, 3]
    elif dimension == 3:
        dim_reduce = [2, 3, 4]
    image_centered = image - torch.mean(image, dim_reduce, keepdim=True)
    stddev = torch.sqrt(torch.mean(image_centered**2, dim_reduce, keepdim=True))
    return image_centered / stddev


def ncc(image_A, image_B):
    A = normalize(image_A[:, :1])
    B = normalize(image_B)
    res = torch.mean(A * B)
    return 1 - res


def gaussian_blur(tensor, kernel_size, sigma):
    kernel1d = F_t._get_gaussian_kernel1d(kernel_size=kernel_size, sigma=sigma).to(
        tensor.device, dtype=tensor.dtype
    )
    out = tensor

    if len(tensor.shape) - 2 == 1:
        out = torch.conv1d(out, kernel1d[None, None, :], padding="same")
    elif len(tensor.shape) - 2 == 2:
        out = torch.conv2d(out, kernel1d[None, None, :, None], padding="same")
        out = torch.conv2d(out, kernel1d[None, None, None, :], padding="same")
    elif len(tensor.shape) - 2 == 3:
        out = torch.conv3d(out, kernel1d[None, None, :, None, None], padding="same")
        out = torch.conv3d(out, kernel1d[None, None, None, :, None], padding="same")
        out = torch.conv3d(out, kernel1d[None, None, None, None, :], padding="same")

    return out


class LNCC:
    def __init__(self, sigma):
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        I = image_A[:, :1]
        J = image_B[:, :1]
        return torch.mean(
            1
            - (self.blur(I * J) - (self.blur(I) * self.blur(J)))
            / torch.sqrt(
                (self.blur(I * I) - self.blur(I) ** 2 + 0.00001)
                * (self.blur(J * J) - self.blur(J) ** 2 + 0.00001)
            )
        )


class BlurredSSD:
    def __init__(self, sigma):
        self.sigma = sigma

    def blur(self, tensor):
        return gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)

    def __call__(self, image_A, image_B):
        return torch.mean((self.blur(image_A[:, :1]) - self.blur(image_B[:, :1])) ** 2)


def ssd(image_A, image_B):
    return torch.mean((image_A[:, :1] - image_B[:, :1]) ** 2)


def ssd_only_interpolated(image_A, image_B):
    if len(image_A.shape) - 2 == 3:
        dimensions_to_sum_over = [2, 3, 4]
    elif len(image_A.shape) - 2 == 2:
        dimensions_to_sum_over = [2, 3]
    elif len(image_A.shape) - 2 == 1:
        dimensions_to_sum_over = [2]
    inbounds_mask = image_A[:, 1:]
    image_A = image_A[:, :1]
    inbounds_squared_distance = inbounds_mask * (image_A - image_B) ** 2
    sum_squared_distance = torch.sum(inbounds_squared_distance, dimensions_to_sum_over)
    divisor = torch.sum(inbounds_mask, dimensions_to_sum_over)
    ssds = sum_squared_distance / divisor
    return torch.mean(ssds)


def flips(phi):
    if len(phi.size()) == 5:
        a = phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]
        b = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]
        c = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        return torch.sum(dV < 0) / phi.shape[0]
    elif len(phi.size()) == 4:
        du = (phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]).detach().cpu()
        dv = (phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]).detach().cpu()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        return torch.sum(dA < 0) / phi.shape[0]
    elif len(phi.size()) == 3:
        du = (phi[:, :, 1:] - phi[:, :, :-1]).detach().cpu()
        return torch.sum(du < 0) / phi.shape[0]
    else:
        raise ValueError()
