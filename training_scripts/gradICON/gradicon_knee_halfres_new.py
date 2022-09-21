import random
import torch

import footsteps

import icon_registration as icon
import icon_registration.networks as networks


SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [1, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

def make_network():
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    for _ in range(2):
         inner_net = icon.TwoStepRegistration(
             icon.DownsampleRegistration(inner_net, dimension=3),
             icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
         )

    net = icon.GradientICON(inner_net, icon.ssd_only_interpolated, lmbda=.2)
    net.assign_identity_map(input_shape)
    return net


BATCH_SIZE=8
GPUS = 4
def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

if __name__ == "__main__":
    footsteps.initialize()


    dataset = torch.load("/playpen-ssd/tgreer/knees_big_2xdownsample_train_set")
    hires_net = make_network()

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, lambda: (make_batch(dataset), make_batch(dataset)), unwrapped_net=hires_net)

