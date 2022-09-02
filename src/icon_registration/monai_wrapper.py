import icon_registration
import torch


class ConcatInputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        return self.net(torch.cat([image_A, image_B], axis=1))

class FirstChannelInputs(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        return self.net(image_A[:, :1], image_B[:, :1])

def make_ddf_from_icon_transform(transform, identity_map):
    """Compute A deformation field compatible with monai's Warp 
    using an ICON transform. The assosciated ICON identity_map is also required
    """
    field_0_1 = transform(identity_map) - identity_map
    network_shape_list = list(identity_map.shape[2:])
    scale = torch.Tensor(network_shape_list).to(identity_map.device)

    for _ in network_shape_list:
        scale = scale[:, None]
    scale = scale[None, :]
    field_spacing_1 = scale * field_0_1
    return field_spacing_1

def make_ddf_using_icon_module(module: icon_registration.RegistrationModule, image_A, image_B):
    """Compute A deformation field compatible with monai's Warp 
    using and ICON RegistrationModule. If the RegistrationModule returns a transform, this function
    returns the monai version of that transform. If the RegistrationModule returns a loss,
    this function returns a monai version of the internal transform as well as the loss.
    """

    res = module(image_A, image_B)
    if type(res) == icon_registration.losses.ICONLoss:
        # Module is a loss function wrapping a registration process

        field = make_ddf_from_icon_transform(module.phi_AB, module.identity_map)

        return field, res

    else:
        # Module is an ordinary RegistrationModule

        field = make_ddf_from_icon_transform(res, module.identity_map)

        return field
        

