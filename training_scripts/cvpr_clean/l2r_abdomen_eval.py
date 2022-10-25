import argparse

import footsteps
import itk
import numpy as np
import torch

import cvpr_network

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str, help="the path to the weights of the network")
parser.add_argument("--data_folder", type=str, help="the path to the folder containing learn2reg AbdomenCTCT dataset")
parser.add_argument("--output_folder", type=str, help="the path to the folder the computed displacement will be saved")

input_shape = [1, 1, 256, 160, 192]
clamp = [-1000, 0]
device = torch.device('cuda:0')

args = parser.parse_args()
weights_path = args.weights_path
footsteps.initialize(output_root=args.output_folder)

import utils

net = cvpr_network.make_network(input_shape, include_last_step=False)

utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.to(device)
net.eval()

import json
import os

import itk
import nibabel as nib

with open(f"{args.data_folder}/AbdomenCTCT_dataset.json", 'r') as data_info:
    data_info = json.loads(data_info.read())
test_cases = [[c["fixed"], c["moving"]] for c in data_info["registration_val"]]

for (fixed_path, moving_path) in test_cases:
    fixed = np.asarray(itk.imread(os.path.join(args.data_folder, fixed_path)))
    moving = np.asarray(itk.imread(os.path.join(args.data_folder, moving_path)))

    fixed = torch.Tensor(np.array(fixed)).unsqueeze(0).unsqueeze(0)
    fixed = (torch.clamp(fixed, clamp[0], clamp[1]) + clamp[0])/(clamp[1]-clamp[0])
    
    moving = torch.Tensor(np.array(moving)).unsqueeze(0).unsqueeze(0)
    moving = (torch.clamp(moving, clamp[0], clamp[1]) + clamp[0])/(clamp[1]-clamp[0])

    with torch.no_grad():
        net(fixed.to(device), moving.to(device))

        # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
        # maps computed by the model
        phi_AB = net.phi_AB(net.identity_map)

        # Transform to displacement format that l2r evaluation script accepts
        disp = (phi_AB - net.identity_map)[0].cpu()

        network_shape_list = list(net.identity_map.shape[2:])

        dimension = len(network_shape_list)

        # We convert the displacement field into an itk Vector Image.
        scale = torch.Tensor(network_shape_list)

        for _ in network_shape_list:
            scale = scale[:, None]
        disp *= scale

        disp_itk_format = (
            disp.double()
            .numpy()[list(reversed(range(dimension)))]
            .transpose([3,2,1,0])
        )

    # Save to output folders
    disp_itk_format = nib.Nifti1Image(disp_itk_format, affine=np.eye(4))
    nib.save(disp_itk_format, f"{footsteps.output_dir}/disp_{fixed_path.split('_')[1]}_{moving_path.split('_')[1]}.nii.gz")
