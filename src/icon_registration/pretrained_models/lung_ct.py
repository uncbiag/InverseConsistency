import random
import shutil

import itk
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration.config as config

from .. import losses, network_wrappers, networks
from ..mermaidlite import compute_warped_image_multiNC, identity_map_multiN


def make_network():

    phi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
    xi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    net = losses.GradientICON(
        network_wrappers.DoubleNet(
            network_wrappers.DownsampleNet(network_wrappers.DoubleNet(phi, psi), 3),
            xi,
        ),
        losses.LNCC(sigma=5),
        1,
    )

    return net


def lung_network_preprocess(
    image: "itk.Image", segmentation: "itk.Image"
) -> "itk.Image":

    image = itk.clamp_image_filter(image, Bounds=(-1000, 0))
    cast_filter = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()
    cast_filter.SetInput(image)
    cast_filter.Update()
    image = cast_filter.GetOutput()

    segmentation_cast_filter = itk.CastImageFilter[
        itk.Image[itk.SI, 3], itk.Image[itk.F, 3]
    ].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()

    image = itk.shift_scale_image_filter(image, shift=1000, scale=1 / 1000)

    mask_filter = itk.MultiplyImageFilter[
        itk.Image[itk.F, 3], itk.Image[itk.F, 3], itk.Image[itk.F, 3]
    ].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()


def LungCT_registration_model(pretrained=True):


    net = make_network()
    input_shape = [1, 1, 175, 175, 175]

    net.assign_identity_map(input_shape)

    if pretrained:
        from os.path import exists

        if not exists("network_weights/lung_model_wms/"):
            print("Downloading pretrained model (200mb)")
            import urllib.request
            import os

            os.makedirs("network_weights", exist_ok=True)
            urllib.request.urlretrieve(
                "https://github.com/uncbiag/ICON/releases/download/pretrained_lung_model/lung_model_wms.zip",
                "network_weights/lung_model_wms.zip",
            )
            shutil.unpack_archive(
                "network_weights/lung_model_wms.zip", "network_weights/lung_model_wms"
            )

        trained_weights = torch.load(
            "network_weights/lung_model_wms/warped_masked_smuth/net91800",
            map_location=torch.device("cpu"),
        )
        net.regis_net.load_state_dict(trained_weights, strict=False)
    net.assign_identity_map(input_shape)

    net.to(config.device)
    net.eval()
    return net
