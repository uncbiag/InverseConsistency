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

import glob
paths = glob.glob("/name-raid1/name/Subcortical_Atlas_Fusion2/*WarpedLabels*")
atlas_registered = [p.split("/malf3")[-1].split("_")[0] for p in paths]

def get_sub_seg(n):
    path = f"/name-raid1/name/Subcortical_Atlas_Fusion2/{n}_label.nii.gz"
    return itk.imread(path)

def mean_dice_f(sA, sB):
    sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
    return utils.itk_mean_dice(sA, sB)

random.seed(1)
for _ in range(20):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(itk.imread(f"/name-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz")) for n in (n_A, n_B))

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))
    segmentation_A, segmentation_B = [
        itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.UC, 3]].New()(image) for image in (segmentation_A, segmentation_B)]

    test_pair = image_A, image_B, segmentation_A, segmentation_B

    test_pair = [ants.from_numpy(1 * itk.array_from_image(t)) for t in test_pair]

    image_A, image_B, segmentation_A, segmentation_B = test_pair
    #image_A, image_B = [ants.resample_image(t, (60, 70, 60), 1, 0 ) for t in [image_A, image_B]]




    reg_res = ants.registration(image_A, image_B, "SyNOnly", outprefix=footsteps.output_dir)

    warped_segmentation_B = ants.apply_transforms(fixed=segmentation_A, moving=segmentation_B, transformlist=reg_res['fwdtransforms'], interpolator="nearestNeighbor")


    mean_dice = mean_dice_f(segmentation_A,warped_segmentation_B)

    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
