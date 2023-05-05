import itk
import monai.transforms
import dataset
import torch
import numpy as np


def itk_crop_foreground(image: itk.Image, monai_crop_pad:monai.transforms.CropForeground):

    torch_image = torch.tensor(itk.GetArrayFromImage(image))

    print(torch_image.shape)

    boundary_coords = monai_crop_pad.compute_bounding_box(torch_image[None])

    print(boundary_coords)

    crop_amounts = list(boundary_coords[0].astype(int)), list(np.array(torch_image.size()) - np.array(boundary_coords[1]))

    crop_amounts = [list(reversed(elem)) for elem in crop_amounts]
    print(crop_amounts)
    crop_amounts = [[int(i) for i in q] for q in crop_amounts] # I will give you five dollars if you can explain why this is needed
    # I know the answer I'm just not telling
    # HInt: itk is picky about ints and ints
    filter = itk.CropImageFilter[type(image), type(image)].New()
    filter.SetInput(image)
    crop_amounts
    filter.SetLowerBoundaryCropSize(crop_amounts[0])
    filter.SetUpperBoundaryCropSize(crop_amounts[1])

    filter.Update()

    image = filter.GetOutput()

#    image = itk.crop_image_filter(image)#, lower_boundary_crop_size=crop_amounts[0])#, upper_boundary_crop_size=crop_amounts[1])

    return image

def itk_resize_with_pad_or_crop(image: itk.image, pad_or_crop:monai.transforms.ResizeWithPadOrCrop):
    torch_image = torch.tensor(itk.GetArrayFromImage(image))
    

def itk_crop(image:itk.image, crop:monai.transforms.SpatialPad):
    pass



def itk_spatial_pad(image:itk.image, spatial_pad:monai.transforms.SpatialPad):
    pass


itk.crop_image_filter
