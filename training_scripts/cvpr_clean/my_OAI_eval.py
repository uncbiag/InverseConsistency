import os
import datetime
# import footsteps
# footsteps.initialize(str(datetime.date()))
import cvpr_network
import torch
import itk
import numpy as np
# import icon_registration.itk_wrapper as itk_wrapper
import utils

from icon_registration.itk_wrapper import create_itk_transform
from icon_registration.losses import to_floats, compute_warped_image_multiNC

from monai.metrics import compute_dice


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

def register_pair(
    model, image_tensor_A, image_tensor_B, image_A, image_B, finetune_steps=None, return_artifacts=False
) -> "(itk.CompositeTransform, itk.CompositeTransform)":

    # assert isinstance(image_A, itk.Image)
    # assert isinstance(image_B, itk.Image)

    model.cuda()

    # A_npy = np.array(image_A)
    # B_npy = np.array(image_B)
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    # A_trch = torch.Tensor(A_npy).cuda()[None, None]
    # B_trch = torch.Tensor(B_npy).cuda()[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    # A_resized = torch.nn.functional.interpolate(
    #     A_trch, size=shape[2:], mode="trilinear", align_corners=False
    # )
    # B_resized = torch.nn.functional.interpolate(
    #     B_trch, size=shape[2:], mode="trilinear", align_corners=False
    # )
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(image_tensor_A, image_tensor_B)
    else:
        raise NotImplementedError()

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    # itk_transforms = (
    #     create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
        
    #     create_itk_transform(phi_BA, model.identity_map, image_B, image_A),
    # )
    # if not return_artifacts:
    #     return itk_transforms
    # else:
    #     return itk_transforms + (to_floats(loss),)
    torch_transforms = (
        phi_AB,
        phi_BA,
    )
    if not return_artifacts:
        return torch_transforms
    else:
        return torch_transforms + (to_floats(loss),)

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
    test_pair_path = test_pair_path.replace("author", "author-raid").split()
    # test_pair_path[0] = test_pair_path[0].replace("/author-raid/author/Data/OAI_segmentation/Nifti_rescaled/", "/author-raid2/raul/DATASETS/OAI/192x192x80/")
    # test_pair_path[1] = test_pair_path[1].replace("/author-raid/author/Data/OAI_segmentation/Nifti_rescaled/", "/author-raid2/raul/DATASETS/OAI/192x192x80/")
    # test_pair_path[2] = test_pair_path[2].replace("/author-raid/author/Data/OAI_segmentation/Nifti_rescaled/", "/author-raid2/raul/DATASETS/OAI/192x192x80/")
    # test_pair_path[3] = test_pair_path[3].replace("/author-raid/author/Data/OAI_segmentation/Nifti_rescaled/", "/author-raid2/raul/DATASETS/OAI/192x192x80/")
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

    # create new data
    if not os.path.exists("/author-raid2/raul/DATASETS/OAI/192x192x80_FLIP_RIGHT"):
        os.makedirs("/author-raid2/raul/DATASETS/OAI/192x192x80_FLIP_RIGHT")

    img_tensor_A = torch.tensor(itk.GetArrayFromImage(image_A))[None, None, ...]
    img_tensor_A = torch.nn.functional.interpolate(img_tensor_A, (80, 192, 192), mode="trilinear", align_corners=False)
    img_tensor_B = torch.tensor(itk.GetArrayFromImage(image_B))[None, None, ...]
    img_tensor_B = torch.nn.functional.interpolate(img_tensor_B, (80, 192, 192), mode="trilinear", align_corners=False)
    # itk.imwrite(itk.GetImageFromArray(img_tensor_A.squeeze()), "/author-raid2/raul/DATASETS/OAI/192x192x80_FLIP_RIGHT/{}".format(os.path.basename(test_pair_path[0])))

    # import pdb; pdb.set_trace()

    phi_AB, phi_BA, loss = register_pair(
        net, img_tensor_A.cuda(), img_tensor_B.cuda(), image_A, image_B, finetune_steps=None, return_artifacts=True
    )

    # interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    # warped_segmentation_A = itk.resample_image_filter(
    #     segmentation_A,
    #     transform=phi_AB,
    #     interpolator=interpolator,
    #     use_reference_image=True,
    #     reference_image=segmentation_B,
    # )
    # mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    seg_tensor_A = torch.tensor(itk.GetArrayFromImage(segmentation_A))[None, None, ...]
    seg_tensor_B = torch.tensor(itk.GetArrayFromImage(segmentation_B))[None, None, ...]

    # phi_AB -> xyz -> x
    #        -> zyx -> 0.715/0.691

    #phi_AB = torch.stack((phi_AB.permute(0, 2, 3, 4, 1)[..., 2], phi_AB.permute(0, 2, 3, 4, 1)[..., 1], phi_AB.permute(0, 2, 3, 4, 1)[..., 0]), dim=-1)
    
    warped_seg_A = compute_warped_image_multiNC(seg_tensor_A.cuda().float(), phi_AB, net.spacing, 0)
    # warped_seg_A = torch.nn.functional.grid_sample(
    #     seg_tensor_A.cuda().float(),
    #     phi_AB,
    #     # phi_AB.permute(0, 2, 3, 4, 1).flip(-1),
    #     mode='nearest',
    #     padding_mode='zeros',
    #     align_corners=True
    # )

    dice = []
    for c in range(1, 3, 1):
        dice.append(
            compute_dice(
                warped_seg_A == c,
                seg_tensor_B.cuda() == c,
                include_background=False
            ).item()
        )
    mean_dice = sum(dice) / len(dice)

    utils.log(mean_dice)
    flips.append(loss.flips)
    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
utils.log("Mean Flips")
utils.log(np.mean(flips))
