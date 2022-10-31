import footsteps

footsteps.initialize(output_root="evaluation_results/")
import icon_registration as icon
import icon_registration.losses as losses
import cvpr_network
import torch
import itk
import numpy as np
import icon_registration.itk_wrapper as itk_wrapper
import utils


def itk_half_scale_image(img):
    scale = 0.5
    input_size = itk.size(img)
    input_spacing = itk.spacing(img)
    input_origin = itk.origin(img)
    dimension = img.GetImageDimension()

    output_size = [int(input_size[d] * scale) for d in range(dimension)]
    output_spacing = [input_spacing[d] / scale for d in range(dimension)]
    output_origin = [
        input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
        for d in range(dimension)
    ]

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(img)

    resampled = itk.resample_image_filter(
        img,
        transform=itk.IdentityTransform[itk.D, 3].New(),
        interpolator=interpolator,
        size=output_size,
        output_spacing=output_spacing,
        output_origin=output_origin,
        output_direction=img.GetDirection(),
    )
    # print(img)
    # print(resampled)
    # exit()

    return resampled


input_shape = [1, 1, 80, 192, 192]
net = cvpr_network.make_network(
    input_shape, include_last_step=True#, lmbda=0.2, loss_fn=icon.ssd_only_interpolated
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("weights_path")
args = parser.parse_args()
weights_path = args.weights_path

utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=False))
net.eval()

with open("../oai_paper_pipeline/splits/test/pair_path_list.txt") as f:
    test_pair_paths = f.readlines()

dices = []
flips = []

for test_pair_path in test_pair_paths:
    test_pair_path = test_pair_path.replace("playpen", "playpen-raid").split()
    test_pair = [itk.imread(path) for path in test_pair_path]
    test_pair = [
        (
            itk.flip_image_filter(t, flip_axes=(False, False, True))
            if "RIGHT" in path
            else t
        )
        for (t, path) in zip(test_pair, test_pair_path)
    ]
    image_A, image_B, segmentation_A, segmentation_B = test_pair

    segmentation_A = itk_half_scale_image(segmentation_A)
    segmentation_B = itk_half_scale_image(segmentation_B)

    phi_AB, phi_BA, loss = itk_wrapper.register_pair(
        net, image_A, image_B, finetune_steps=50, return_artifacts=True
    )

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
        segmentation_A,
        transform=phi_AB,
        interpolator=interpolator,
        use_reference_image=True,
        reference_image=segmentation_B,
    )
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    utils.log(mean_dice)
    utils.log(icon.losses.to_floats(loss))
    flips.append(loss.flips)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
utils.log("Mean Flips")
utils.log(np.mean(flips))
