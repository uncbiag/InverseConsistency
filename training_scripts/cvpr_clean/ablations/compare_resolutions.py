import random
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser

import sys
sys.path.append('./training_scripts/cvpr_clean')

import footsteps
import cvpr_network
from cvpr_network import make_network

import icon_registration as icon
import icon_registration.networks as networks
from icon_registration import data
from icon_registration.losses import ncc
import copy
from icon_registration.mermaidlite import compute_warped_image_multiNC

parser = ArgumentParser()
parser.add_argument("--exp", type=str, default="debug", help="name of the experiment")
parser.add_argument("--dataset", type=str, default="circle", help="Value can be [circle, retina, lung_dataset_path]")
parser.add_argument("--data_shape", type=int, default="128", help="The shape of the 2D toy data.")
parser.add_argument("--reg", type=str, default="gradICON", help='choose from (icon, gradICON)')

BATCH_SIZE= 4
GPUS = 4
GPU_IDs = [0,1,2,3]

def write_stats(writer, stats_dict, ite, prefix= ""):
    for k,v in stats_dict.items():
        writer.add_scalar(f"{prefix}{k}", v, ite)

def set_seed_for_demo():
    """ reproduce the training demo"""
    seed = 2021
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dataset_triangles(
    split=None, data_size=128, hollow=False, samples=6000, batch_size=128
):
    x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
    x = np.reshape(x, (1, data_size, data_size))
    y = np.reshape(y, (1, data_size, data_size))
    cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
    r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
    theta = np.random.random((samples, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((samples, 1, 1)) > 0.5

    triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(np.pi / 3) / np.cos(
        (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
    )

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
    if hollow:
        triangles = 1 - triangles**2
        circles = 1 - circles**2

    images = isTriangle * triangles + (1 - isTriangle) * circles

    ds = torch.Tensor(np.expand_dims(images, 1)) + 1.
    return ds

def get_dataloaders(dataset_root, scale = "4xdown", batch_size={"train":5, "val":5}):
    if dataset_root == "circle":
        train_set = get_dataset_triangles(data_size=int(scale), samples=100, batch_size=GPUS*BATCH_SIZE, hollow=False)
        return lambda : (make_batch(train_set), make_batch(train_set)), None
    elif dataset_root == "retina":
        d1, d2 = data.get_dataset_retina(extra_deformation=True, downsample_factor=int(scale))
        d1 = d1.dataset
        d2 = d2.dataset
        d = torch.cat([d1.tensors[0], d2.tensors[0]], dim=1)
        return lambda : (make_batch_paired(d)), None
    else:
        img = torch.load(f"{dataset_root}/lungs_train_{scale}_scaled", map_location='cpu')
        mask = torch.load(f"{dataset_root}/lungs_seg_train_{scale}_scaled", map_location='cpu')
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.stack(
                    [(torch.cat(d, 0)+1)*torch.cat(m, 0) for d,m in zip(img, mask)],
                    0
                ),
                torch.stack(
                    [torch.cat(d, 0) for d in mask],
                    0
                )
            ),
            batch_size = batch_size["train"],
            shuffle = True,
            drop_last = True
        )
        img = torch.load(f"{dataset_root}/lungs_test_{scale}_scaled", map_location='cpu')
        mask = torch.load(f"{dataset_root}/lungs_seg_test_{scale}_scaled", map_location='cpu')
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.stack(
                    [(torch.cat(d, 0)+1)*torch.cat(m, 0) for d,m in zip(img, mask)],
                    0
                ),
                torch.stack(
                    [torch.cat(d, 0) for d in mask],
                    0
                )
            ),
            batch_size = batch_size["val"],
            shuffle = False,
            drop_last = True
        )
    return train_loader, test_loader

def make_batch(dataset):
    image = torch.stack([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)], dim=0)
    image = image.cuda()
    image = image / torch.max(image)
    return image

def make_batch_paired(dataset):
    image = torch.stack([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)], dim=0)
    image = image.cuda()
    image = image / torch.max(image)
    return image[:, 0:1], image[:, 1:2]

def train_kernel(net, opt, writer, loader, log_prefix, ite_start, ite_end):
    scores = {"inv_loss": 0., "sim_loss": 0., "phi_mag": 0., "neg_jacob": 0.}
    ite_step = 0
    for ite in range(ite_start, ite_end):
        opt.zero_grad()
        image_A, image_B = loader()
        result_object = net(image_A, image_B)
        loss = torch.mean(result_object.all_loss) 
        loss.backward()
        opt.step()

        scores["inv_loss"] += torch.mean(result_object.inverse_consistency_loss.detach().cpu()).item()
        scores["sim_loss"] += torch.mean(result_object.similarity_loss.detach().cpu()).item()
        scores["phi_mag"] += torch.mean(result_object.transform_magnitude.detach().cpu()).item()
        scores["neg_jacob"] += torch.mean(result_object.flips.cpu()).item()
        
        ite_step += 1

    total_run = ite_end - ite_start
    for k,v in scores.items():
        scores[k] = v/total_run
    write_stats(writer, scores, ite_start, f"{log_prefix}_train/")

def val_kernel(net, writer, loader, log_prefix, ite, plot_path):
    def _dice(seg_A, seg_B, phi, spacing):
        warped_seg_A = compute_warped_image_multiNC(
                seg_A, phi, spacing, spline_order=0
            )
        
        len_intersection = torch.sum(warped_seg_A * seg_B, [1, 2, 3, 4])
        fn = torch.sum(seg_B, [1, 2, 3, 4]) - len_intersection
        fp = torch.sum(warped_seg_A, [1, 2, 3, 4]) - len_intersection

        return 2 * len_intersection / (2 * len_intersection + fn + fp + 1e-10)

    device = next(net.parameters()).device
    scores = {"inv_loss": 0., "sim_loss": 0., "phi_mag": 0., "neg_jacob": 0., "dice": 0.}
    with torch.no_grad():
        for img, seg in loader:
            result_object = net(img[:,0].to(device), img[:,1].to(device))
            scores["inv_loss"] += torch.mean(result_object.inverse_consistency_loss.detach().cpu()).item()
            scores["sim_loss"] += torch.mean(result_object.similarity_loss.detach().cpu()).item()
            scores["phi_mag"] += torch.mean(result_object.transform_magnitude.detach().cpu()).item()
            scores["neg_jacob"] += torch.mean(result_object.flips.cpu()).item()
            scores["dice"] += torch.mean(
                    _dice(
                        seg[:,0].to(device),
                        seg[:,1].to(device),
                        net.phi_AB_vectorfield,
                        net.spacing
                    ).detach().cpu()
                ).item()

        total_run = len(loader)
        for k,v in scores.items():
            scores[k] = v/total_run
        write_stats(writer, scores, ite, f"{log_prefix}_val/")


def train(net, writer, exp_root, iterations, train_loader, test_loader, log_per_ites):
    
    GPU_IDs_validate = 1
   
    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net.cuda(), device_ids=GPU_IDs, output_device=GPU_IDs[0])
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)
    ite = 0
    
    net_par.train()

    for ite in range(0, iterations, log_per_ites):

        # if _ % 10 == 0:
        #     net_val = copy.deepcopy(net_par.module).to(f"cuda:{GPU_IDs_validate}")
        #     val_kernel(net_val, writer, test_loader, "down2x", _, f"{exp_root}/figures")
        #     del net_val
        
        if ite % 1 == 0:
            torch.save(
                net.regis_net.state_dict(), exp_root + f"/checkpoints/net_{ite:04}"
            )
        train_kernel(net_par, optimizer, writer, train_loader, "down2x", ite, ite+log_per_ites)

    
    torch.save(
                net.regis_net.state_dict(), exp_root + f"/checkpoints/net_{ite:04}_final"
            )
    

if __name__ == "__main__":
    set_seed_for_demo()

    args = parser.parse_args()
    data_shape = args.data_shape

    footsteps.initialize(output_root=f"./results/{args.exp}/", run_name=f"scale_{data_shape}")

    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    exp_folder_path = footsteps.output_dir

    print(f"Saving experiment info at {exp_folder_path}")

    writer = SummaryWriter(exp_folder_path + "/log", flush_secs=30)

    os.makedirs(exp_folder_path + "/checkpoints")

    train_loader, test_loader = get_dataloaders("retina", scale=f"{data_shape}", batch_size={"train":GPUS*BATCH_SIZE, "val":2*BATCH_SIZE})
    input_shape = [1] + [*(train_loader()[0].shape[1:])]
    print(input_shape)

    net = make_network(input_shape, include_last_step=False, lmbda=1.0*input_shape[2]**2, framework=args.reg, loss_fn=icon.LNCC(sigma=5))
    train(net, writer, exp_folder_path, 10000, train_loader, None, 10)

    writer.close()

