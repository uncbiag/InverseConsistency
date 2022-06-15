import torch
import footsteps
import icon_registration
import icon_registration.losses
import icon_registration.train
from icon_registration import data


class DirectVectorNet(icon_registration.RegistrationModule):
    def __init__(self, noise_scale=0.0001):
        super().__init__()
        self.noise_scale = noise_scale

    def assign_identity_map(self, input_shape):
        super().assign_identity_map(input_shape)
        self.register_parameter(
            "displacement_field", torch.nn.Parameter(0 * self.identity_map)
        )

    def forward(self, image_A, image_B):
        return self.displacement_field + self.noise_scale * torch.randn(
            self.displacement_field.shape, device="cuda"
        )


class AssymmetricGradientICON(icon_registration.GradientICON):
    def __init__(self, network1, network2, similarity, lmbda):

        super().__init__(network1, similarity, lmbda)

        self.regis_net2 = network2

    def forward(self, image_A, image_B) -> icon_registration.losses.ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net2(image_B, image_A)

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
        return icon_registration.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon_registration.losses.flips(self.phi_BA_vectorfield),
        )


ds1, _ = data.get_dataset_triangles(data_size=128, samples=2, batch_size=2)

x = next(iter(ds1))
image_A = x[0][:1].cuda()
image_B = x[0][1:].cuda()


def make_batch():
    return (image_A, image_B)


def make_network():
    x = icon_registration.FunctionFromVectorField(DirectVectorNet())
    for _ in range(5):
        x = icon_registration.TwoStepRegistration(
            icon_registration.DownsampleRegistration(x, 2),
            icon_registration.FunctionFromVectorField(DirectVectorNet()),
        )
    return x


net = AssymmetricGradientICON(make_network(), make_network(), icon_registration.ssd, 1)


footsteps.initialize(run_name="direct")

input_shape = image_A.shape
net.assign_identity_map(input_shape)
net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
net.train()

import icon_registration.visualize

for i in range(15):

    def reduce_noise_scale(net: torch.nn.Module):
        if hasattr(net, "noise_scale"):
            net.noise_scale = net.noise_scale * 0.999
        for child in net.children():
            reduce_noise_scale(child)

    icon_registration.train.train_batchfunction(
        net,
        optimizer,
        make_batch,
        steps=100,  # step_callback=reduce_noise_scale
    )
    icon_registration.visualize.visualizeRegistration(
        net, image_A, image_B, 0, footsteps.output_dir + f"{i}.png"
    )
