import footsteps
import subprocess
import glob
import sys
import random
import itk

# footsteps.initialize(output_root="evaluation_results/", run_name="asdf")
import numpy as np
import utils


def itk_rotate_scale_image(img, label=True):
    if not label:
        a = np.array(img)

        b = a[a != 0]

        print(np.min(a), np.percentile(b, 5) , np.max(a))

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

    print(resampled)
    resamplednt = np.array(resampled)


    iref = itk.imread("ref.nii.gz")
    iref = itk.CastImageFilter[itk.Image[itk.UC, 3], itk.Image[itk.D, 3]].New()(iref)
    iref = itk.shift_scale_image_filter(iref, shift=0.0, scale = 1.0 / 255)

    resamplednt = np.transpose(resamplednt, (1, 0, 2))
    resamplednt = np.flip(resamplednt, axis=1)


    npresampleditk = itk.image_from_array(resampled)#.astype(np.float64)

    transform2 = itk.MatrixTransform[itk.D, 3]

    

    sys.exit()

    #resampled.SetSpacing(iref.GetSpacing())
    #resampled.SetDirection(iref.GetDirection())
    #resampled.SetOrigin(iref.GetOrigin())
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
    image_A, image_B = (
        preprocess(
            itk.imread(
                f"/name-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
            )
        )
        for n in (n_A, n_B)
    )

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

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

    cmd = """python /name-raid1/name/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model /name-raid1/name/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5 --warp warp.nii.gz"""
    # cmd = """python /name-raid1/name/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model shapes-dice-vel-3-res-8-16-32-256f.h5 --warp warp.nii.gz"""
    subprocess.run(cmd, shell=True)

    import voxelmorph
    import voxelmorph as vxm

    vsegfix = voxelmorph.py.utils.load_labels("segA.nii.gz")[1][0][None, :, :, :, None]
    vsegmov = voxelmorph.py.utils.load_labels("segB.nii.gz")[1][0][None, :, :, :, None]
    warp = voxelmorph.py.utils.load_volfile("warp.nii.gz", add_batch_axis=True)
    warped_seg = vxm.networks.Transform(
        vsegfix.shape[1:-1], interp_method="nearest"
    ).predict([vsegmov, warp])
    overlap = vxm.py.utils.dice(vsegfix, warped_seg, labels=list(range(1, 29)))

    mean_dice = np.mean(overlap)
    dices.append(np.mean(overlap))

    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
