import itk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration.pretrained_models
import icon_registration.network_wrappers

import icon_registration.config as config

def register_pair(model, image_A, image_B)->"(itk.CompositeTransform, itk.CompositeTransform)":

    assert( isinstance(image_A, itk.Image))
    assert( isinstance(image_B, itk.Image))
    
    icon_registration.network_wrappers.adjust_batch_size(model, 1)
    # send model to cpu or gpu depending on config- auto detects capability
    model.to(config.device) 

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(config.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(config.device)[None, None]

    shape = model.identityMap.shape
    
    # Here we resize the input images to the shape expected by the neural network. This affects the 
    # pixel stride as well as the magnitude of the displacement vectors of the resulting displacement field, which
    # create_itk_transform will have to compensate for.
    A_resized = F.interpolate(A_trch, size=shape[2:], mode="trilinear", align_corners=False)
    B_resized = F.interpolate(B_trch, size=shape[2:], mode="trilinear", align_corners=False)

    with torch.no_grad():
        model(A_resized, B_resized)
    
    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward 
    # maps computed by the model
    phi_AB = model.phi_AB(model.identityMap)
    phi_BA = model.phi_BA(model.identityMap)
    
    # the parameters ident, image_A, and image_B are used for their metadata
    return (
        create_itk_transform(phi_AB, model.identityMap, image_A, image_B),
        create_itk_transform(phi_BA, model.identityMap, image_B, image_A)
    )

def create_itk_transform(phi, ident, image_A, image_B)->"itk.CompositeTransform":
    
    # itk.DeformationFieldTransform expects a displacement field, so we subtract off the identity map.
    disp = (phi - ident)[0].cpu()

    network_shape_list = list(ident.shape[2:])

    dimension = len(network_shape_list)
    
    tr = itk.DisplacementFieldTransform[(itk.D, dimension)].New()       

    # We convert the displacement field into an itk Vector Image. 
    scale = torch.Tensor(network_shape_list)
    
    for _ in network_shape_list:
        scale = scale[:, None]
    disp *= scale
    
    # disp is a shape [3, H, W, D] tensor with vector components in the order [vi, vj, vk]
    disp_itk_format  = disp.double().numpy()[list(reversed(range(dimension)))].transpose(list(range(1, dimension + 1)) + [0])
    # disp_itk_format is a shape [H, W, D, 3] array with vector components in the order [vk, vj, vi]
    # as expected by itk.

    itk_disp_field = itk.image_from_array(disp_itk_format, is_vector=True)

    tr.SetDisplacementField(itk_disp_field)

    to_network_space = resampling_transform(image_A, list(reversed(network_shape_list)))

    from_network_space = resampling_transform(image_B, list(reversed(network_shape_list))).GetInverseTransform()

    phi_AB_itk = itk.CompositeTransform[itk.D, dimension].New()

    phi_AB_itk.PrependTransform(from_network_space)
    phi_AB_itk.PrependTransform(tr)
    phi_AB_itk.PrependTransform(to_network_space)
    
    # warp(image_A, phi_AB_itk) is close to image_B

    return phi_AB_itk

    
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
     
