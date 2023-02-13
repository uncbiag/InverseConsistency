import footsteps
import random
import itk
import ants
footsteps.initialize(output_root="evaluation_results/")
import numpy as np
import utils

def preprocess(image):
    #image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    #image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image

dices = []
flips = []
pair_list = []

def mean_dice_f(sA, sB):
    sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
    return utils.itk_mean_dice(sA, sB)

from synthmorph.HCP_segs import (atlas_registered, get_sub_seg, get_brain_image)

random.seed(1)
for _ in range(100):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(get_brain_image(n)) for n in (n_A, n_B))

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))
    segmentation_A, segmentation_B = [
        itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.UC, 3]].New()(image) for image in (segmentation_A, segmentation_B)]

    test_pair = image_A, image_B, segmentation_A, segmentation_B

    test_pair = [ants.from_numpy(1 * itk.array_from_image(t)) for t in test_pair]

    image_A, image_B, segmentation_A, segmentation_B = test_pair
    #image_A, image_B = [ants.resample_image(t, (60, 70, 60), 1, 0 ) for t in [image_A, image_B]]




    reg_res = ants.registration(image_A, image_B, "SyNOnly", outprefix=footsteps.output_dir)

    warped_segmentation_B = ants.apply_transforms(fixed=segmentation_A, moving=segmentation_B, transformlist=reg_res['fwdtransforms'], interpolator="nearestNeighbor")


    mean_dice = mean_dice_f(segmentation_A, warped_segmentation_B)

    jac_np = np.array(itk.displacement_field_jacobian_determinant_filter(itk.imread(reg_res['fwdtransforms'][0])))
    flip = np.mean(jac_np<0) * 100.0
    
    flips.append(flip)
    dices.append(mean_dice)
    utils.log(f"{_}/100 mean DICE {n_A} to {n_B}: {mean_dice} | running DICE: {np.mean(dices)} | Percentage: {flip} | running Per: {np.mean(flips)} ")
    pair_list.append([n_A, n_B])

utils.log("Mean DICE")
utils.log(f"Final DICE: {np.mean(dices)} | final percentage of negative jacobian: {np.mean(flips)}")

with open(f'{footsteps.output_dir}/pair_list.txt', 'w') as f:
    for p in pair_list:
        f.write(",".join(p)+'\n')
