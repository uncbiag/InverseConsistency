import torch.nn.functional as F
from mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import torch
import random
import icon_registration.inverseConsistentNet as inverseConsistentNet
import icon_registration.networks as networks
import icon_registration.network_wrappers as network_wrappers
import icon_registration.data as data
import footsteps

BATCH_SIZE = 32
SCALE = 1  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [BATCH_SIZE, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

GPUS = 1

scale_down = [BATCH_SIZE, 1, 20 * SCALE, 48 * SCALE, 48 * SCALE]
net = inverseConsistentNet.InverseConsistentNet(
    network_wrappers.DownsampleNet(
        network_wrappers.FunctionFromVectorField(
            networks.FCNet3D(scale_down, bottleneck=512)
        ),
        dimension=3,
    ),
    lambda x, y: torch.mean((x - y) ** 2),
    70,
)

net.assign_identity_map(input_shape)


knees = torch.load("/author/author/knees_big_train_set")
trained_weights = torch.load("results/fc_smol_knee/knee_aligner_resi_net30000")
network_wrappers.adjust_batch_size(net, 16)
net.load_state_dict(trained_weights)
network_wrappers.adjust_batch_size(net, BATCH_SIZE)

if GPUS == 1:
    net_par = net.cuda()
else:
    net_par = torch.nn.DataParallel(net).cuda()
optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00001)
optimizer_state = torch.load("results/fc_smol_knee/knee_aligner_resi_opt30000")
optimizer.load_state_dict(optimizer_state)

net_par.train()


def make_batch():
    image = torch.cat([random.choice(knees) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    return image


loss_curve = []
for _ in range(0, 100000):
    for subbatch in range(3):
        optimizer.zero_grad()
        moving_image = make_batch()
        fixed_image = make_batch()
        loss, a, b, c = net_par(moving_image, fixed_image)
        loss = torch.mean(loss) / 3
        loss.backward()

    loss_curve.append([torch.mean(l.detach().cpu()).item() for l in (a, b, c)])
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
            net.state_dict(), footsteps.output_dir + "knee_aligner_resi_net" + str(_)
        )
