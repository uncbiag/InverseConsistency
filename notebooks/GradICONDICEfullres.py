#!/usr/bin/env python
# coding: utf-8


import os
import random
from collections import OrderedDict
import icon_registration

import icon_registration.data as data
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
import itk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from icon_registration.mermaidlite import (
    compute_warped_image_multiNC,
    identity_map_multiN,
)

r_ds = torch.load("/playpen/tgreer/knees_test_set_fullres")
# batched_ds = list(zip(*[r_ds[i::4] for i in range(2)]))

batched_ds = list(zip(*[r_ds[i::1] for i in range(1)]))


phi = network_wrappers.FunctionFromVectorField(
    networks.tallUNet(unet=networks.UNet2ChunkyMiddle, dimension=3)
)
psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

pretrained_lowres_net = network_wrappers.DoubleNet(phi, psi)

regis_net = network_wrappers.DoubleNet(
    network_wrappers.DownsampleNet(pretrained_lowres_net, 3),
    network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
)

regis_net = network_wrappers.DoubleNet(
    network_wrappers.DownsampleNet(regis_net, 3),
    network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
)

BATCH_SIZE = 1
SCALE = 4  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

regis_net.assign_identity_map(input_shape)

trained_weights = torch.load("../results/grad_fullres-6/knee_aligner_resi_net15000")
regis_net.load_state_dict(trained_weights, strict=False)


net = inverseConsistentNet.GradientICON(
    regis_net,
    icon_registration.ssd,
    .2,
)


BATCH_SIZE = 1
SCALE = 4  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

net.assign_identity_map(input_shape)


net.cuda()
net.eval()
0


def flips(phi):
    a = phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]
    b = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]
    c = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]

    dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
    return torch.sum(dV < 0)  # / BATCH_SIZE


def flips2(phi):
    a = phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]
    b = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]
    c = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]

    dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True) / np.product(
        net.spacing
    )
    return dV.detach().cpu()


dices = []
folds_list = []

filteredDV_list = []


def dice(x):
    x = list(zip(*x))
    x = [torch.cat(r, 0).cuda().float() for r in x]
    fixed_image, fixed_cartilage = x[0], x[2]
    moving_image, moving_cartilage = x[1], x[3]
    net.regis_net.load_state_dict(trained_weights, strict=False)

    optim = torch.optim.Adam(net.parameters(), lr=.00002)

    for _ in range(50):
        optim.zero_grad()
        loss_obj = net(moving_image, fixed_image)
        loss_obj.all_loss.backward()
        optimizer.step()

    phi_AB_vectorfield = net.phi_AB_vectorfield
    fat_phi = phi_AB_vectorfield[:, :3]
    sz = np.array(fat_phi.size())
    spacing = 1.0 / (sz[2::] - 1)
    warped_moving_cartilage = compute_warped_image_multiNC(
        moving_cartilage.float(), fat_phi, spacing, 1
    )
    wmb = warped_moving_cartilage > 0.5
    fb = fixed_cartilage > 0.5
    intersection = wmb * fb
    d = (
        2
        * torch.sum(intersection, [1, 2, 3, 4]).float()
        / (torch.sum(wmb, [1, 2, 3, 4]) + torch.sum(fb, [1, 2, 3, 4]))
    ).item()
    print(d)
    dices.append(d)
    f = flips(phi_AB_vectorfield[:1])
    print(f)
    folds_list.append(f.item())
    f = flips(phi_AB_vectorfield[1:])
    print(f)
    folds_list.append(f.item())

    dV = flips2(phi_AB_vectorfield)

    dV = np.array(dV).flatten()
    dV = dV[dV < 0]
    filteredDV_list.append(dV)


for x in batched_ds[:]:
    dice(x)

dd = torch.mean(torch.cat(dices).cpu())
print(dd)
ff = np.mean(folds_list)
print(ff)


dicesa = np.array(torch.cat(dices).cpu())
plt.hist(dicesa, 20)
plt.xlabel("DICE")
print(np.mean(dicesa))
ff = np.mean(folds_list)
print(ff)
