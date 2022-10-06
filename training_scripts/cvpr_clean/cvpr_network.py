import icon_registration as icon

import icon_registration.networks as networks

from icon_registration import config
from icon_registration.mermaidlite import compute_warped_image_multiNC
import torch
import icon_registration.network_wrappers as network_wrappers

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

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
            + 2 * torch.randn(*self.identity_map.shape).to(config.device)
            / self.identity_map.shape[-1]
        )[:, :, ::2, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

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
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return icon.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon.losses.flips(self.phi_BA_vectorfield),
        )


def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5)):
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=3),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)))
    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net
