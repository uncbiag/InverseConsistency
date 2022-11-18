import random

import footsteps
import torch
import torch.nn.functional as F

import icon_registration.data as data
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration.mermaidlite import (compute_warped_image_multiNC,
                                           identity_map_multiN)

BATCH_SIZE = 32
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 4


phi = network_wrappers.FunctionFromVectorField(
    networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
)
psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

net = inverseConsistentNet.GradientICON(
    network_wrappers.DoubleNet(phi, psi),
    inverseConsistentNet.ssd_only_interpolated,
    0.2,
)

net.assign_identity_map(input_shape)


knees = torch.load("/author-ssd/author/knees_14k_small")

if GPUS == 1:
    net_par = net.cuda()
else:
    net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)


net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    return image


loss_curve = []
for _ in range(0, 100000):
    for subbatch in range(1):
        optimizer.zero_grad()
        moving_image = make_batch()
        fixed_image = make_batch()
        loss, a, b, c, flips = net_par(moving_image, fixed_image)
        loss = torch.mean(loss)
        loss.backward()

    loss_curve.append(
        [torch.mean(l.detach().cpu()).item() for l in (a, b, c)] + [flips, net.lmbda]
    )
    print(loss_curve[-1])
    optimizer.step()

    if _ % 300 == 0:
        try:
            import pickle

            with open(footsteps.output_dir + "loss_curve", "wb") as f:
                pickle.dump(loss_curve, f)
        except:
            pass
        torch.save(
            optimizer.state_dict(),
            footsteps.output_dir + "knee_aligner_resi_opt" + str(_),
        )
        torch.save(
            net.regis_net.state_dict(),
            footsteps.output_dir + "knee_aligner_resi_net" + str(_),
        )
