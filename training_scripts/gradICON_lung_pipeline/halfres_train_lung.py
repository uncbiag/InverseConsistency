
import os
import random
import shutil
from argparse import ArgumentParser
from datetime import datetime
from re import M

import icon_registration
import icon_registration.networks as networks
import icon_registration.losses as icon_losses
import numpy as np
import torch
import torch.nn.functional as F
import copy
from icon_registration.mermaidlite import compute_warped_image_multiNC
from torch.utils.tensorboard import SummaryWriter
import footsteps

parser = ArgumentParser()
parser.add_argument("--exp", type=str, default="debug", help="name of the experiment")
parser.add_argument('-g',"--gpu_id",required=False, type=int, default=0, help='gpu_id to use')
parser.add_argument('-o',"--output_root", type=str, default="./", help='path to the experiment root folder.')
parser.add_argument('-d',"--dataset_root",required=True, type=str, default="./", help='path to the dataset root folder.')
parser.add_argument("--reg", type=str, default="GradientICON", help='choose from (InverseConsistentNet, GradientICON)')
parser.add_argument("--sim", type=str, default="NCC", help='choose from (NCC, LNCC, AdaptiveNCC)')
parser.add_argument("--lamda", type=float, default="0.1", help='lambda applied to the regularizer.')
parser.add_argument("--augmentation", type=int, default="0", help='0 -- w/o augmentation, 1 -- w augmentation.')

EXP_LOSSES = {
   "NCC": icon_losses.ncc,
   "LNCC": icon_losses.LNCC(5),
   "AdaptiveNCC": icon_losses.AdaptiveNCC()
}

def set_seed_for_demo():
    """ reproduce the training demo"""
    seed = 2021
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def build_framework(net, input_shape, sim, regularization, labda=.1):
    if regularization == "icon":
        icon = icon_registration.GradientICON(
            net,
            sim,
            labda
        )
        icon.assign_identity_map(input_shape)
    elif regularization == "gradICON":
        icon = icon_registration.GradientICON(
            net,
            sim,
            labda
        )
        icon.assign_identity_map(input_shape)
    else:
        print("Regulariztion in build_framework funciton is not set correctly.")

    return icon

def make_2x_net():
    inner_net = icon_registration.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    for _ in range(2):
        inner_net = icon_registration.TwoStepRegistration(
                icon_registration.DownsampleRegistration(inner_net, 3), 
                icon_registration.FunctionFromVectorField(
                    networks.tallUNet2(dimension=3))
            )

    return inner_net

def get_dataloaders(dataset_root, scale = "4xdown", batch_size={"train":5, "val":5}):
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

def write_stats(writer, stats_dict, ite, prefix= ""):
    for k,v in stats_dict.items():
        writer.add_scalar(f"{prefix}{k}", v, ite)

def augment(image_A, image_B):
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.Tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1 
        identity = identity * (torch.randint_like(identity, 0, 2) * 2  - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    
    noise = torch.randn((image_A.shape[0], 3, 4))

    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
   
    warped_A = F.grid_sample(image_A, forward_grid, padding_mode='border')

    noise = torch.randn((image_A.shape[0], 3, 4))
    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
    warped_B = F.grid_sample(image_B, forward_grid, padding_mode='border')

    return warped_A, warped_B

def no_augment(image_A, image_B):
    return image_A, image_B

def train_kernel(net, opt, writer, loader, augmenter, log_prefix, ite):
    scores = {"inv_loss": 0., "sim_loss": 0., "phi_mag": 0., "neg_jacob": 0.}
    ite_step = 0
    for d, _ in loader:
        opt.zero_grad()

        image_A, image_B = d[:,0].cuda(), d[:,1].cuda()
        image_A, image_B = augmenter(image_A, image_B)
        result_object = net(image_A, image_B)
        loss = torch.mean(result_object.all_loss) 
        loss.backward()
        opt.step()

        scores["inv_loss"] += torch.mean(result_object.inverse_consistency_loss.detach().cpu()).item()
        scores["sim_loss"] += torch.mean(result_object.similarity_loss.detach().cpu()).item()
        scores["phi_mag"] += torch.mean(result_object.transform_magnitude.detach().cpu()).item()
        scores["neg_jacob"] += torch.mean(result_object.flips.cpu()).item()
        
        ite_step += 1

    total_run = len(loader)
    for k,v in scores.items():
        scores[k] = v/total_run
    write_stats(writer, scores, ite, f"{log_prefix}_train/")

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


def train(net, augmenter, writer, exp_root, dataset_root, load_from="", load_epoch=0):
    BATCH_SIZE = 4
    GPUS = 4
    GPU_IDs = [0,1,2,3]
    GPU_IDs_validate = 1
   
    train_loader, test_loader = get_dataloaders(dataset_root, scale="2xdown", batch_size={"train":GPUS*BATCH_SIZE, "val":2*BATCH_SIZE})

    if GPUS == 1:
        net_par = net.cuda()
    else:
        net_par = torch.nn.DataParallel(net.cuda(), device_ids=GPU_IDs, output_device=GPU_IDs[0])
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (0.8 ** int(epoch/20)))
    start_epoch = 0

    if load_from != "":
        weights = torch.load(f"{load_from}/down2x_net_{load_epoch:04}", map_location='cpu')
        net.regis_net.load_state_dict(weights)
        opt_state_dict = torch.load(f"{load_from}/down2x_opt_{load_epoch:04}", map_location='cpu')
        optimizer.load_state_dict(opt_state_dict["optimizer"])
        # lr_scheduler.load_state_dict(opt_state_dict["lr_scheduler"])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.8, last_epoch=opt_state_dict["epoch"])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (0.8 ** int(epoch/20)), last_epoch=opt_state_dict["epoch"])
        start_epoch = opt_state_dict["epoch"] + 1
    
    net_par.train()

    for _ in range(start_epoch, 101):

        if _ % 10 == 0:
            net_val = copy.deepcopy(net_par.module).to(f"cuda:{GPU_IDs_validate}")
            val_kernel(net_val, writer, test_loader, "down2x", _, f"{exp_root}/figures")
            del net_val
        
        if _ % 10 == 0:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": _
                },
                exp_root + f"/checkpoints/down2x_opt_{_:04}"
            )
            torch.save(
                net.regis_net.state_dict(), exp_root + f"/checkpoints/down2x_net_{_:04}"
            )
        train_kernel(net_par, optimizer, writer, train_loader, augmenter, "down2x", _)
        
        write_stats(writer, {"lr":np.array(lr_scheduler.get_last_lr()).mean()}, _, "lr")
        lr_scheduler.step()
    
    torch.save(
                net.regis_net.state_dict(), exp_root + f"/checkpoints/down2x_net_{start_epoch if '_' not in locals() else _:04}_final"
            )
    


if __name__ == "__main__":
    args = parser.parse_args()
    
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    footsteps.initialize(output_root=args.output_root, run_name=args.exp +'/'+ timestamp)
    exp_folder_path = footsteps.output_dir

    print(f"Saving experiment info at {exp_folder_path}")

    writer = SummaryWriter(exp_folder_path + "/log", flush_secs=30)

    os.makedirs(exp_folder_path + "/checkpoints")

    set_seed_for_demo()

    # prepare the network based on configuration
    inshape = [1,1,175,175,175]
    net = make_2x_net()
    icon = getattr(icon_registration, args.reg)(
        net,
        EXP_LOSSES[args.sim],
        args.lamda
    )
    icon.assign_identity_map(inshape)

    if args.augmentation == 0:
        train(icon, no_augment, writer, exp_folder_path, args.dataset_root)
    elif args.augmentation == 1:
        train(icon, augment, writer, exp_folder_path, args.dataset_root)
    else:
        print("Invalid value for augmentation setting.")
    
    writer.close()
