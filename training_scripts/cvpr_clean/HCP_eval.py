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
net = cvpr_network.make_network(input_shape, include_last_step=True)#, framework="svf")


utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

dices = []
flips = []
shape_prod = np.prod(input_shape)

from synthmorph.HCP_segs import (atlas_registered, get_sub_seg, get_brain_image)

pair_list = []
random.seed(1)
for _ in range(100):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(get_brain_image(n)) for n in (n_A, n_B))

    #import pdb; pdb.set_trace()
    phi_AB, phi_BA, loss= itk_wrapper.register_pair(net, image_A, image_B, finetune_steps=None,
        return_artifacts=True,
    )

    # print(net.warped_image_A.shape)
    
    # import shutil
    # shutil.copyfile(f"/playpen-raid2/Data/HCP/HCP_1200/{n_A}/T1w/T1w_acpc_dc_restore_brain.nii.gz", f"{footsteps.output_dir}/{n_A}_T1w_acpc_dc_restore_brain.nii.gz")
    # shutil.copyfile(f"/playpen-raid2/Data/HCP/HCP_1200/{n_B}/T1w/T1w_acpc_dc_restore_brain.nii.gz", f"{footsteps.output_dir}/{n_B}_T1w_acpc_dc_restore_brain.nii.gz")

    # interpolator = itk.LinearInterpolateImageFunction.New(image_A)
    # warped_image_A = itk.resample_image_filter(
    #         image_A,
    #         transform=phi_AB,
    #         interpolator=interpolator,
    #         size=itk.size(image_B),
    #         output_spacing=itk.spacing(image_B),
    #         output_direction=image_B.GetDirection(),
    #         output_origin=image_B.GetOrigin(),
    #     )
    # np.save(f"{footsteps.output_dir}/{n_A}_and_{n_B}_warped.np", np.array(warped_image_A))

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
    
    flip = loss.flips / shape_prod * 100.
    flips.append(flip)
    dices.append(mean_dice)
    utils.log(f"{_}/100 mean DICE {n_A} to {n_B}: {mean_dice} | running DICE: {np.mean(dices)} | Percentage: {flip} | running Per: {np.mean(flips)} ")
    pair_list.append([n_A, n_B])

utils.log("Mean DICE")
utils.log(f"Final DICE: {np.mean(dices)} | final percentage of negative jacobian: {np.mean(flips)}")

with open(f'{footsteps.output_dir}/pair_list.txt', 'w') as f:
    for p in pair_list:
        f.write(",".join(p)+'\n')
