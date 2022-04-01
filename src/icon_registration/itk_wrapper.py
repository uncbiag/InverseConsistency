import itk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration.pretrained_models
import icon_registration.network_wrappers

import icon_registration.visualize

import icon_registration.config as config

def register_pair(model, image_A, image_B):

    assert( isinstance(image_A, itk.Image))
    assert( isinstance(image_B, itk.Image))
    icon_registration.network_wrappers.adjust_batch_size(model, 1)
    model.to(config.device) 

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)
    A_trch = torch.Tensor(A_npy).to(config.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(config.device)[None, None]

    shape = model.identityMap.shape

    A_resized = F.interpolate(A_trch, size=shape[2:], mode="trilinear", align_corners=False)
    B_resized = F.interpolate(B_trch, size=shape[2:], mode="trilinear", align_corners=False)

    with torch.no_grad():
        x = model(A_resized, B_resized)
    
    phi_AB = model.phi_AB(model.identityMap)[0].cpu()
    phi_BA = model.phi_BA(model.identityMap)[0].cpu()

    return (
        create_itk_transform(phi_AB, model.identityMap, image_A, image_B),
        create_itk_transform(phi_BA, model.identityMap, image_B, image_A)
    )

def create_itk_transform(phi, ident, image_A, image_B):
    
    disp = phi - ident[0].cpu()

    network_shape_list = list(ident.shape[2:])

    dimension = len(network_shape_list)

    scale = torch.Tensor(network_shape_list)
    for _ in network_shape_list:
        scale = scale[:, None]
    disp *= scale
    tr = itk.DisplacementFieldTransform[(itk.D, dimension)].New()       

    disp_itk_format  = disp.double().numpy()[list(reversed(range(dimension)))].transpose(list(range(1, dimension + 1)) + [0])

    itk_disp_field = array_to_vector_image(disp_itk_format)

    tr.SetDisplacementField(itk_disp_field)

    to_aligned = resampling_transform(image_A, list(reversed(network_shape_list)))

    from_aligned = resampling_transform(image_B, list(reversed(network_shape_list))).GetInverseTransform()

    phi_AB_itk = itk.CompositeTransform[itk.D, dimension].New()

    phi_AB_itk.PrependTransform(from_aligned)
    phi_AB_itk.PrependTransform(tr)
    phi_AB_itk.PrependTransform(to_aligned)

    return phi_AB_itk


anti_garbage_collection_box = []
def array_to_vector_image(array):
    # array is a numpy array of doubles of shape 
    # 3, H, W, D

    # returns an itk.Image of itk.Vector
    assert isinstance(array, np.ndarray)

    array = np.ascontiguousarray(array)

    # if array is ever garbage collected, we crash.
    anti_garbage_collection_box.append(array)

    PixelType = itk.Vector[itk.D, 3]
    ImageType = itk.Image[PixelType, 3]

    vector_image = itk.PyBuffer[ImageType].GetImageViewFromArray(array, array.shape[:-1])

    return vector_image
    
def resampling_transform(image, shape):
    
    imageType = itk.template(image)[0][itk.template(image)[1]]
    
    dummy_image = itk.image_from_array(np.zeros(tuple(reversed(shape)), dtype=itk.array_from_image(image).dtype))
    if len(shape) == 2:
        transformType = itk.MatrixOffsetTransformBase[itk.D, 2, 2]
    else:
        transformType = itk.VersorRigid3DTransform[itk.D]
    initType = itk.CenteredTransformInitializer[transformType, imageType, imageType]
    initializer = initType.New()
    initializer.SetFixedImage(dummy_image)
    initializer.SetMovingImage(image)
    transform = transformType.New()
    
    initializer.SetTransform(transform)
    initializer.InitializeTransform()
    
    if len(shape) == 3:
        transformType = itk.CenteredAffineTransform[itk.D, 3]
        t2 = transformType.New()
        t2.SetCenter(transform.GetCenter())
        t2.SetOffset(transform.GetOffset())
        transform = t2
    m = transform.GetMatrix()
    m_a = itk.array_from_matrix(m)
    
    input_shape = image.GetLargestPossibleRegion().GetSize()
    
    for i in range(len(shape)):
    
        m_a[i, i] = image.GetSpacing()[i] * (input_shape[i] / shape[i])
    
    m_a = itk.array_from_matrix(image.GetDirection()) @ m_a 
    
    transform.SetMatrix(itk.matrix_from_array(m_a))
    
    return transform
     
