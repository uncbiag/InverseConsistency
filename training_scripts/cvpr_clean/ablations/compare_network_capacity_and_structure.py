
import random
import sys
import os
from argparse import ArgumentParser
from datetime import datetime

import footsteps
import torch
from torch.utils.tensorboard import SummaryWriter

from icon_registration import networks

sys.path.append('./training_scripts/cvpr_clean')
import icon_registration as icon
import icon_registration.data as data
from cvpr_network import GradientICONSparse

BATCH_SIZE = 4
GPUS = 4
GPU_IDs = [0,1,2,3]
ITERATIONS_PER_STEP = 50000

parser = ArgumentParser()
parser.add_argument("--exp", type=str, default="debug", help="name of the experiment")
parser.add_argument("--output_folder", type=str, default="./results/", help="the path to the output folder")
parser.add_argument("--unet_type", type=str, default="vm", help="value can be [vm, icon]")
parser.add_argument("--multi_res", type=int, default=0, help="Set to 0 if do not use multi-resolutions structure.")

def set_seed_for_demo():
    """ reproduce the training demo"""
    seed = 2021
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_batch_paired(dataset, batch_size):
    image = torch.stack([random.choice(dataset) for _ in range(batch_size)])
    image = image.cuda()
    image = image / torch.max(image)
    return image[:, 0:1], image[:, 1:2]
    
def make_network(unet_class, input_shape, multi_res, lmbda=1.5, loss_fn=icon.LNCC(sigma=5)):
    dimension = len(input_shape) - 2
    inner_net = icon.FunctionFromVectorField(unet_class(dimension=dimension))
    
    if multi_res:
        for _ in range(2):
            inner_net = icon.TwoStepRegistration(
                icon.DownsampleRegistration(inner_net, dimension=dimension),
                icon.FunctionFromVectorField(unet_class(dimension=dimension))
            )
    
    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net

if __name__ == "__main__":
    set_seed_for_demo()

    args = parser.parse_args()

    footsteps.initialize(output_root=args.output_folder, run_name=args.exp)

    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    exp_folder_path = footsteps.output_dir
    print(f"Saving experiment info at {exp_folder_path}")

    writer = SummaryWriter(exp_folder_path + "/log", flush_secs=30)
    os.makedirs(exp_folder_path + "/checkpoints")

    dataset = data.get_copdgene_dataset(data_folder="", cache_folder="/author-raid2/author/data/icon", downscale=2)
    batch_function = lambda : make_batch_paired(dataset.tensors[0], GPUS*BATCH_SIZE)

    example = batch_function()
    input_shape = [1] + [*example[0].shape[1:]]

    if args.unet_type == "vm":
        from unet_from_vm import Unet as unet_class
    else:
        unet_class = networks.tallUNet2
    net = make_network(unet_class, input_shape, args.multi_res)

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    icon.train_batchfunction(net_par, optimizer, batch_function, unwrapped_net=net, steps=ITERATIONS_PER_STEP)
    
    torch.save(
                net.regis_net.state_dict(),
                footsteps.output_dir + "Step_1_final.trch",
            )

    writer.close()
