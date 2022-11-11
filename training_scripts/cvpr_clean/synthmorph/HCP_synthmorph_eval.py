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
   # model_path = "/playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5"

    #model_path = "/playpen-raid1/tgreer/voxelmorph/vxm_dense_brain_T1_3D_mse.h5"
    model_path = "shapes-dice-vel-3-res-8-16-32-256f.h5"
    regis_net = vxm.networks.VxmDense.load(model_path)
    warper_label = vxm.networks.Transform(
          ref.shape[1:-1], interp_method="nearest"
        )   
    warper = vxm.networks.Transform(inshape, nb_feats=nb_feats)
    def voxelmorph_register(moving_p, fixed_p):

        moving = vxm.py.utils.load_volfile(moving_p, add_batch_axis=True, add_feat_axis=True)
        fixed = vxm.py.utils.load_volfile(
              fixed_p, add_batch_axis=True, add_feat_axis=True)
        warp = regis_net.register(moving, fixed)
        #moved = warper.predict([moving, warp])
        return warp



    def itk_rotate_scale_image(img, label=True):
        if not label:
            a = np.array(img)

            b = a[a != 0]


            max_ = np.max(np.array(img))
            img = itk.shift_scale_image_filter(img, shift=0.0, scale=1.0 / max_)
        scale = [0.618, 0.618, 0.618]
        input_size = itk.size(img)
        input_spacing = itk.spacing(img)
        input_origin = itk.origin(img)
        dimension = img.GetImageDimension()

        output_size = [int(input_size[d] * scale[d]) for d in range(dimension)]
        output_spacing = [input_spacing[d] / scale[d] for d in range(dimension)]
        output_origin = [
            input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
            for d in range(dimension)
        ]

        if label:
            interpolator = itk.NearestNeighborInterpolateImageFunction.New(img)
        else:
            interpolator = itk.LinearInterpolateImageFunction.New(img)

        transform = itk.CenteredEuler3DTransform[itk.D].New()

        params = transform.GetParameters()

        params[0] = 0.5

        transform.SetParameters(params)

        transform.SetCenter([0.0, 65.0, -10.0])

        resampled = itk.resample_image_filter(
            img,
            transform=transform,
            interpolator=interpolator,
            size=output_size,
            output_spacing=output_spacing,
            output_origin=output_origin,
            output_direction=img.GetDirection(),
        )
        #if not label:

        #    max_ = np.max(np.array(resampled))
        #    resampled = itk.shift_scale_image_filter(resampled, shift=0.0, scale=1.0 / max_)

        resamplednt = np.array(resampled)


        iref = itk.imread("ref.nii.gz")
        iref = itk.CastImageFilter[itk.Image[itk.UC, 3], itk.Image[itk.D, 3]].New()(iref)
        iref = itk.shift_scale_image_filter(iref, shift=0.0, scale = 1.0 / 255)

        resamplednt = np.transpose(resamplednt, (1, 0, 2))
        resamplednt = np.flip(resamplednt, axis=1)


        npresampleditk = itk.image_from_array(resamplednt)#.astype(np.float64)

        #transform2 = itk.MatrixTransform[itk.D, 3]

        

        resampled = npresampleditk
        resampled.SetSpacing(iref.GetSpacing())
        resampled.SetDirection(iref.GetDirection())
        resampled.SetOrigin(iref.GetOrigin())
        #resampled = itk.checker_board_image_filter(
        #    resampled, iref

        #)
        # print(img)
        # print(resampled)
        # exit()

        return resampled


    def preprocess(image):
        # image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
        # image = itk.clamp_image_filter(image, bounds=(0, 1))
        return image


    dices = []


    from HCP_segs import (atlas_registered, get_sub_seg, get_brain_image)

    def mean_dice_f(sA, sB):
        sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
        return utils.itk_mean_dice(sA, sB)


    random.seed(1)
    for _ in range(1):
        n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
        image_A, image_B = (
            preprocess(
                itk.imread(
                    f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
                )
            )
            for n in (n_A, n_B)
        )

        segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

        itk.imwrite(segmentation_A, "segA_orig.nii.gz")
        itk.imwrite(image_A, "imageA_orig.nii.gz")

        segmentation_A, segmentation_B = [
            itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.UC, 3]].New()(image)
            for image in (segmentation_A, segmentation_B)
        ]

        image_A = itk_rotate_scale_image(image_A, label=False)
        image_B = itk_rotate_scale_image(image_B, label=False)
        segmentation_A = itk_rotate_scale_image(segmentation_A, label=True)
        segmentation_B = itk_rotate_scale_image(segmentation_B, label=True)

        itk.imwrite(image_A, "A.nii.gz")
        itk.imwrite(image_B, "B.nii.gz")
        itk.imwrite(segmentation_A, "segA.nii.gz")
        itk.imwrite(segmentation_B, "segB.nii.gz")

        #subprocess.run("vshow B.nii.gz -z", shell=True)
        #sys.exit()

        #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model /playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5 --warp warp.nii.gz"""
        # cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model shapes-dice-vel-3-res-8-16-32-256f.h5 --warp warp.nii.gz"""
        #subprocess.run(cmd, shell=True)

        import voxelmorph
        import voxelmorph as vxm

        warp = voxelmorph_register("B.nii.gz", "A.nii.gz")

        vsegfix = voxelmorph.py.utils.load_labels("segA.nii.gz")[1][0][None, :, :, :, None]
        vsegmov = voxelmorph.py.utils.load_labels("segB.nii.gz")[1][0][None, :, :, :, None]
        warped_seg = warper_label.predict([vsegmov, warp])
        overlap = vxm.py.utils.dice(vsegfix, warped_seg, labels=list(range(1, 29)))

        mean_dice = np.mean(overlap)
        dices.append(np.mean(overlap))

        utils.log(mean_dice)

        dices.append(mean_dice)

    utils.log("Mean DICE")
    utils.log(np.mean(dices))
