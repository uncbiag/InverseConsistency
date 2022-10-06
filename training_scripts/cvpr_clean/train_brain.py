import random
import os
import torch

import footsteps
import cvpr_network

import icon_registration as icon
import icon_registration.networks as networks

input_shape = [1, 1, 130, 155, 130]

BATCH_SIZE=8
GPUS = 4
#ITERATIONS_PER_STEP = 50000
ITERATIONS_PER_STEP = 601

def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

if __name__ == "__main__":
    footsteps.initialize()

    dataset = torch.load(
        "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
    )
    net = cvpr_network.make_network(input_shape, include_last_step=False)

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, lambda: (make_batch(dataset), make_batch(dataset)), unwrapped_net=net, steps=ITERATIONS_PER_STEP)

    net_2 = cvpr_network.make_network(input_shape, include_last_step=True)

    net_2.regis_net.netPhi.load_state_dict(net.regis_net.state_dict())

    del net
    del net_par
    del optimizer

    if GPUS == 1:
        net_2_par = net_2.cuda()
    else:
        net_2_par = torch.nn.DataParallel(net_2).cuda()
    optimizer = torch.optim.Adam(net_2_par.parameters(), lr=0.00005)

    net_2_par.train()
    
    # We're being weird by training two networks in one script. This hack keeps
    # the second training from overwriting the outputs of the first.
    footsteps.output_dir_impl = footsteps.output_dir + "2nd_step/"
    os.makedirs(footsteps.output_dir)

    icon.train_batchfunction(net_2_par, optimizer, lambda: (make_batch(dataset), make_batch(dataset)), unwrapped_net=net_2, steps=ITERATIONS_PER_STEP)
