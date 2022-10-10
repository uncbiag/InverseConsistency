import itk
import footsteps
footsteps.initialize(output_root="evaluation_results/")
import cvpr_network
import torch
import utils
import numpy as np
import unittest
import matplotlib.pyplot as plt
import numpy as np

import icon_registration.pretrained_models
import icon_registration.itk_wrapper
import icon_registration.test_utils
import icon_registration.pretrained_models.lung_ct

image_root = "/playpen-raid1/lin.tian/data/lung/dirlab_highres_350"
landmark_root = "/playpen-raid1/lin.tian/data/lung/reg_lung_2d_3d_1000_dataset_4_proj_clean_bg/landmarks/"

cases = [f"copd{i}_highres" for i in range(1,11)]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("weights_path")
args = parser.parse_args()
weights_path = args.weights_path

input_shape = [1, 1, 175, 175, 175]
net = cvpr_network.make_network(input_shape, include_last_step=False)


utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

for case in cases:
    image_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_img.nii.gz")
    image_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_img.nii.gz")
    seg_insp = itk.imread(f"{image_root}/{case}/{case}_INSP_STD_COPD_label.nii.gz")
    seg_exp = itk.imread(f"{image_root}/{case}/{case}_EXP_STD_COPD_label.nii.gz")
    

    landmarks_insp = icon_registration.test_utils.read_copd_pointset(landmark_root + f"/{case.split('_')[0]}_300_iBH_xyz_r1.txt")
    landmarks_exp = icon_registration.test_utils.read_copd_pointset(landmark_root + f"/{case.split('_')[0]}_300_eBH_xyz_r1.txt")


    image_insp_preprocessed = icon_registration.pretrained_models.lung_network_preprocess(image_insp, seg_insp)
    image_exp_preprocessed = icon_registration.pretrained_models.lung_network_preprocess(image_exp, seg_exp)

    phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(net, image_insp_preprocessed, image_exp_preprocessed)

    dists = []
    for i in range(len(landmarks_exp)):
        px, py = (
            landmarks_insp[i],
            np.array(phi_AB.TransformPoint(tuple(landmarks_exp[i]))),
        )
    dists.append(np.sqrt(np.sum((px - py) ** 2)))
    print(f"Mean error on {case}: ", np.mean(dists))
    dists = []
    for i in range(len(landmarks_insp)):
        px, py = (
            landmarks_exp[i],
            np.array(phi_BA.TransformPoint(tuple(landmarks_insp[i]))),
        )
    dists.append(np.sqrt(np.sum((px - py) ** 2)))
    print(f"Mean error on {case}: ", np.mean(dists))

    
