import parent
from collections import OrderedDict
import torch.nn.functional as F
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch
import random
import inverseConsistentNet
import networks
import data
import describe

BATCH_SIZE = 24
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
working_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 4

tmp_affine_net = inverseConsistentNet.InverseConsistentAffineNet(
    networks.ConvolutionalMatrixNet(dimension=3),
    lmbda=100,
    input_shape=working_shape,
)

phi = networks.StumpyConvolutionalMatrixNet(dimension=3)
psi = networks.StumpyConvolutionalMatrixNet(dimension=3)
affine_net = networks.DoubleAffineNet(
    phi, psi, tmp_affine_net.identityMap, tmp_affine_net.spacing
)

net = inverseConsistentNet.InverseConsistentAffineDeformableNet(
    affine_net,
    networks.tallUNet2(dimension=3),
    lmbda=100,
    input_shape=working_shape,
)

pretrained_weights = torch.load(
    "results/affine_knee_double_stumpy10/knee_aligner_resi_net35400"
)
pretrained_weights = OrderedDict(
    [
        (a.split("regis_net.")[1], b)
        for a, b in pretrained_weights.items()
        if ("regis_net" in a and not ("identityMap$" in a + "$"))
    ]
)
net.adjust_batch_size(32)
net.affine_regis_net.load_state_dict(pretrained_weights)
net.adjust_batch_size(24)
for p in net.affine_regis_net.parameters():
    p.requires_grad = False

# _, knees = data.get_knees_dataset()
knees = torch.load("/author/author/knees_big_train_set")
if GPUS == 1:
    net_par = net.cuda()
else:
    net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(
    (p for p in net_par.parameters() if p.requires_grad), lr=0.00005
)

net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    return image


loss_curve = []
for _ in range(0, 100000):
    optimizer.zero_grad()
    moving_image = make_batch()
    fixed_image = make_batch()
    loss, a, b, c = net_par(moving_image, fixed_image)
    loss = torch.mean(loss)
    loss.backward()

    loss_curve.append([torch.mean(l.detach().cpu()).item() for l in (a, b, c)])
    print(loss_curve[-1])
    optimizer.step()

    if _ % 300 == 0:
        try:
            import pickle

            with open(describe.run_dir + "loss_curve", "wb") as f:
                pickle.dump(loss_curve, f)
        except:
            pass
        torch.save(
            optimizer.state_dict(), describe.run_dir + "knee_aligner_resi_opt" + str(_)
        )
        torch.save(
            net.state_dict(), describe.run_dir + "knee_aligner_resi_net" + str(_)
        )
