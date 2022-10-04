import icon_registration as icon

import icon_registration.networks as networks


def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5)):
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=3),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=3))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=3)))
    net = icon.GradientICON(inner_net, loss_fn, lmbda=lmbda)
    net.assign_identity_map(input_shape)
    return net
