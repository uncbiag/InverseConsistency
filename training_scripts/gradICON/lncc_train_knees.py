import torch
import random
import icon_registration
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.data as data
import footsteps


GPUS = 4
BATCH_SIZE = 8
def make_network():

    phi = network_wrappers.FunctionFromVectorField(
        networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
    )
    psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    hires_net = icon_registration.GradientICON(
        network_wrappers.DoubleNet(
            network_wrappers.DownsampleNet(
                network_wrappers.TwoStepRegistration(phi, psi), dimension=3
            ),
            network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
        ),
        icon_registration.LNCCOnlyInterpolated(sigma=5),
        3,
    )
    SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
    input_shape = [1, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]
    hires_net.assign_identity_map(input_shape)
    return hires_net


hires_net = make_network()

knees = torch.load("/author-ssd/author/knees_big_2xdownsample_train_set")

if GPUS == 1:
    net_par = hires_net.cuda()
else:
    net_par = torch.nn.DataParallel(hires_net).cuda()
optimizer = torch.optim.Adam(
    (p for p in net_par.parameters() if p.requires_grad), lr=0.00005
)


net_par.train()


def make_batch():
    image_A = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image_A = image_A.cuda()
    image_B = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image_B = image_B.cuda()

    return image_A, image_B


icon_registration.train_batchfunction(net_par, optimizer, make_batch, unwrapped_net = hires_net)
