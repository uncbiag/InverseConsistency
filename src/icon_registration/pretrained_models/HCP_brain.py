import itk
from .lung_ct import init_network

def brain_network_preprocess(image: "itk.Image") -> "itk.Image":
    _, max_ = itk.image_intensity_min_max(image)
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    return image

def brain_registration_model(pretrained=True):
    return init_network("brain", pretrained=pretrained)
