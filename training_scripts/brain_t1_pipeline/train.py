import random

import footsteps
import icon_registration as icon
import icon_registration.networks as networks
import torch


BATCH_SIZE = 8
input_shape = [BATCH_SIZE, 1, 130, 155, 130]

GPUS = 4


def make_network():
    phi = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    psi = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    hires_net = icon.GradientICON(
        icon.TwoStepRegistration(
            icon.DownsampleRegistration(
                icon.TwoStepRegistration(phi, psi), dimension=3
            ),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
        ),
        icon.LNCC(sigma=5),
        .7,
    )
    hires_net.assign_identity_map(input_shape)
    return hires_net


def make_batch():
    image = torch.cat([random.choice(brains) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image


if __name__ == "__main__":
    footsteps.initialize()
    brains = torch.load(
        "/author-ssd/author/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
    )
    hires_net = make_network()

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, lambda: (make_batch(), make_batch()), unwrapped_net=hires_net)
