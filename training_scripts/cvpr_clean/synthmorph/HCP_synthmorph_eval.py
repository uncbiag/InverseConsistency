import footsteps
import subprocess
import glob
import sys
import random
import itk

# footsteps.initialize(output_root="evaluation_results/", run_name="asdf")
import numpy as np
import utils

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
device, nb_devices = vxm.tf.utils.setup_device("0")
import voxelmorph as vxm

ref = vxm.py.utils.load_volfile("ref.nii.gz", add_batch_axis = True, add_feat_axis = True)

inshape = ref.shape[1:-1]
nb_feats = ref.shape[-1]


with tf.device(vxm.tf.utils.setup_device("0")[0]):
    model_path = "/playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5"

    #model_path = "/playpen-raid1/tgreer/voxelmorph/vxm_dense_brain_T1_3D_mse.h5"
    #model_path = "shapes-dice-vel-3-res-8-16-32-256f.h5"
    regis_net = vxm.networks.VxmDense.load(model_path)
    warper_label = vxm.networks.Transform(
          ref.shape[1:-1], interp_method="nearest"
        )   
    warper = vxm.networks.Transform(inshape, nb_feats=nb_feats)
    def voxelmorph_register(moving_p, fixed_p):

        moving = vxm.py.utils.load_volfile(moving_p, add_batch_axis=True, add_feat_axis=True)
        fixed = vxm.py.utils.load_volfile(
              fixed_p, add_batch_axis=True, add_feat_axis=True)
        moving = moving / np.max(moving)
        fixed = fixed / np.max(fixed)

        print(np.min(moving), np.max(moving))
        warp = regis_net.register(moving, fixed)
        #moved = warper.predict([moving, warp])
        return warp




    dices = []


    from HCP_segs import (atlas_registered, get_sub_seg, get_brain_image)

    def mean_dice_f(sA, sB):
        sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
        return utils.itk_mean_dice(sA, sB)


    random.seed(1)
    for _ in range(100):
        n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
        image_A, image_B = (
                itk.imread(
                    f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
                )
            for n in (n_A, n_B)
        )

        segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

        itk.imwrite(segmentation_A, "segA_orig.nii.gz")
        itk.imwrite(image_A, "imageA_orig.nii.gz")
        itk.imwrite(segmentation_B, "segB_orig.nii.gz")
        itk.imwrite(image_B, "imageB_orig.nii.gz")


        subprocess.run("mri_robust_register --mov imageA_orig.nii.gz --dst ref.nii.gz -lta A_Affine.lta --satit --iscale", shell=True)
        subprocess.run("mri_robust_register --mov imageA_orig.nii.gz --dst ref.nii.gz -lta A_Affine.lta --satit --iscale --ixform A_Affine.nii.gz --affine", shell=True)
        subprocess.run("mri_vol2vol --mov imageA_orig.nii.gz --o A_affine.nii.gz --lta A_Affine.lta --targ ref.nii.gz", shell=True)
        subprocess.run("mri_vol2vol --mov segA_orig.nii.gz --o Aseg_affine.nii.gz --lta A_Affine.lta --targ ref.nii.gz --nearest, --keep-precision", shell=True)

        subprocess.run("mri_robust_register --mov imageB_orig.nii.gz --dst ref.nii.gz -lta B_Affine.lta --satit --iscale", shell=True)
        subprocess.run("mri_robust_register --mov imageB_orig.nii.gz --dst ref.nii.gz -lta B_Affine.lta --satit --iscale --ixform B_affine.nii.gz --affine", shell=True)
        subprocess.run("mri_vol2vol --mov imageB_orig.nii.gz --o B_affine.nii.gz --lta B_Affine.lta --targ ref.nii.gz", shell=True)
        subprocess.run("mri_vol2vol --mov segB_orig.nii.gz --o Bseg_affine.nii.gz --lta B_Affine.lta --targ ref.nii.gz --nearest --keep-precision", shell=True)

        #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model /playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5 --warp warp.nii.gz"""
        #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A_affine.nii.gz --moving B_affine.nii.gz --moved out.nii.gz --model shapes-dice-vel-3-res-8-16-32-256f.h5 --warp warp.nii.gz"""
        #subprocess.run(cmd, shell=True)

        import voxelmorph
        import voxelmorph as vxm


        vsegfix = voxelmorph.py.utils.load_labels("Aseg_affine.nii.gz")[1][0][None, :, :, :, None]
        vsegmov = voxelmorph.py.utils.load_labels("Bseg_affine.nii.gz")[1][0][None, :, :, :, None]
        warp = voxelmorph_register("B_affine.nii.gz", "A_affine.nii.gz") 
        warped_seg = warper_label.predict([vsegmov, warp])
        overlap = vxm.py.utils.dice(vsegfix, warped_seg, labels=list(range(1, 29)))

        mean_dice = np.mean(overlap)
        dices.append(np.mean(overlap))

        utils.log(mean_dice)

        dices.append(mean_dice)

    utils.log("Mean DICE")
    utils.log(np.mean(dices))
