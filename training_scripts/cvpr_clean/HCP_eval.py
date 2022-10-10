import footsteps
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("weights_path")
args = parser.parse_args()

weights_path = args.weights_path
footsteps.initialize(output_root="evaluation_results/")
import icon_registration as icon
import cvpr_network
import torch
import itk
import numpy as np
import icon_registration.itk_wrapper as itk_wrapper
import utils



def preprocess(image):
    #image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    #image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image


input_shape = [1, 1, 130, 155, 130]
net = cvpr_network.make_network(input_shape, include_last_step=True)


utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

dices = []

import glob
paths = glob.glob("/playpen-raid1/tgreer/Subcortical_Atlas_Fusion2/*WarpedLabels*")
atlas_registered = [p.split("/malf3")[-1].split("_")[0] for p in paths]

def get_sub_seg(n):
    path = f"/playpen-raid1/tgreer/Subcortical_Atlas_Fusion2/{n}_label.nii.gz"
    return itk.imread(path)

random.seed(1)
for _ in range(20):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(itk.imread(f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz")) for n in (n_A, n_B))

    #import pdb; pdb.set_trace()
    phi_AB, phi_BA = itk_wrapper.register_pair(net, image_A, image_B, finetune_steps=None)

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
            segmentation_A, 
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=segmentation_B
            )
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
