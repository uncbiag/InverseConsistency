import random
from icon_registration.mermaidlite import compute_warped_image_multiNC, identity_map_multiN
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn

import footsteps

import icon_registration as icon
import icon_registration.networks as networks
import icon_registration.losses as losses
from icon_registration.losses import ICONLoss


SCALE = 2  # 1 IS QUARTER RES, 2 IS HALF RES, 4 IS FULL RES
input_shape = [1, 1, 40 * SCALE, 96 * SCALE, 96 * SCALE]

class GradientICON(icon.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

        _g_kern =  10 * torch.randn([12, 3, 2, 2, 2])

        _g_kern -= torch.mean(_g_kern, dim=[2, 3, 4], keepdim=True)

        self.register_buffer("gradient_kernel", _g_kern)


    def forward(self, image_A, image_B) -> ICONLoss:

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc

        #p1 = self.regis_net.netPsi.net(image_A, image_B)
        #p2 = self.regis_net.netPsi.net(image_B, image_A)

        #l = torch.mean(p1 - p2)

        #return ICONLoss(l, l, l, l, l)
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        
        down_map = self.identity_map[:, :, ::2, ::2, ::2]

        Iepsilon = (
                down_map
                + torch.randn(*down_map.shape).to(self.identity_map.device)
            * 1
            / down_map.shape[-1]
        )

        self.phi_AB_vectorfield = self.phi_AB(Iepsilon)

        inbounds_tag = torch.zeros(tuple(image_A.shape), device=image_A.device)
        if len(self.input_shape) - 2 == 3:
            inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
        elif len(self.input_shape) - 2 == 2:
            inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
        else:
            inbounds_tag[:, :, 1:-1] = 1.0

        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1)
        )(self.phi_AB_vectorfield)

        matched_image_B = self.as_function(image_B)(Iepsilon)

        similarity_loss = 2 * self.similarity(self.warped_image_A, matched_image_B)

        BA_comp_AB = self.phi_BA(self.phi_AB_vectorfield) - Iepsilon

        BA_comp_AB = torch.nn.functional.conv3d(BA_comp_AB, self.gradient_kernel)
        
        inverse_consistency_loss = torch.mean(BA_comp_AB **2)
        

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (Iepsilon - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            transform_magnitude,#losses.flips(self.phi_AB_vectorfield),
        )

class TwoStepDownsampleRegistration(icon.RegistrationModule):
    """Combine two RegistrationModules.

    First netPhi is called on the input images at half resolution, then image_A is warped with
    the resulting field with upsampling, and then netPsi is called on warped A and image_B
    in order to find a residual warping. Finally, the composition of the two
    transforms is returned.
    """

    def __init__(self, netPhi, netPsi, dimension):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi
        if dimension == 2:
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.dimension = dimension

    def forward(self, image_A, image_B):
        # Tag for optimization. Must be set at the beginning of forward because it is not preserved by .to(config.device)
        self.identity_map.isIdentity = True
        s_image_A = self.avg_pool(image_A, 2, ceil_mode=True)
        s_image_B = self.avg_pool(image_B, 2, ceil_mode=True)
        phi = self.netPhi(s_image_A, s_image_B)
        phi_vectorfield = phi(self.identity_map[:, :, ::2, ::2])
        phi_vectorfield = torch.nn.Upsample(self.identity_map.size()[2:], mode=self.interpolate_mode)(phi_vectorfield)
        self.image_A_comp_phi = self.as_function(image_A)(phi_vectorfield)
        psi = self.netPsi(self.image_A_comp_phi, image_B)

        ret = lambda input_: phi(psi(input_))
        return ret


    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = np.array(input_shape)
        self.input_shape[0] = 1
        self.spacing = 1.0 / (self.input_shape[2::] - 1)

        _id = identity_map_multiN(self.input_shape, self.spacing)
        self.register_buffer("identity_map", torch.from_numpy(_id))

        first_child_shape = np.concatenate(
            [
                self.input_shape[:2],
                np.ceil(self.input_shape[2:] / 2).astype(int),
            ]
        )
        second_child_shape = self.input_shape
        self.netPhi.assign_identity_map(first_child_shape)
        self.netPsi.assign_identity_map(second_child_shape)

def vm_unet(dimension, input_channels=2):
    return icon.networks.UNet2(
        5,
        [[2, 16, 32, 32, 32, 32], [16, 32, 32, 32, 32]],
        dimension,
        input_channels=input_channels,
    )


#unet = vm_unet
unet = networks.tallUNet2
def make_network():
    inner_net = icon.FunctionFromVectorField(unet(dimension=3))

    for _ in range(2):
         inner_net = TwoStepDownsampleRegistration(
             inner_net,
             icon.FunctionFromVectorField(unet(dimension=3)),
             dimension=3
         )

    inner_net = icon.TwoStepRegistration(
        inner_net,
        icon.FunctionFromVectorField(unet(dimension=3))
    )
    net = GradientICON(inner_net, icon.ssd_only_interpolated, lmbda=1)
    net.assign_identity_map(input_shape)
    return net


BATCH_SIZE=2
GPUS = 4
def make_batch(dataset):

    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    #image = image / torch.max(image)
    return image

from torch.cuda.amp import GradScaler, autocast
def train_batchfunction(
    net,
    optimizer,
    make_batch,
    steps=500000,
    step_callback=(lambda net: None),
    unwrapped_net=None,
):
    """A training function intended for long running experiments, with tensorboard logging
    and model checkpoints. Use for medical registration training
    """
    import footsteps
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    from icon_registration.losses import to_floats
    from icon_registration.train import write_stats

    if unwrapped_net is None:
        unwrapped_net = net

    loss_curve = []
    writer = SummaryWriter(
        footsteps.output_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=30,
    )
    scaler = GradScaler()
    for iteration in range(0, steps):
        optimizer.zero_grad()
        with autocast():
            moving_image, fixed_image = make_batch()
            loss_object = net(moving_image, fixed_image)
            loss = torch.mean(loss_object.all_loss)
        scaler.scale(loss).backward()

        step_callback(unwrapped_net)

        print(to_floats(loss_object))
        write_stats(writer, loss_object, iteration)
        scaler.step(optimizer)
        scaler.update()

        if iteration % 300 == 0:
            torch.save(
                optimizer.state_dict(),
                footsteps.output_dir + "optimizer_weights_" + str(iteration),
            )
            torch.save(
                unwrapped_net.regis_net.state_dict(),
                footsteps.output_dir + "network_weights_" + str(iteration),
            )
if __name__ == "__main__":
    footsteps.initialize()


    dataset = torch.load("/author/author/knees_big_2xdownsample_train_set")
    hires_net = make_network()

    if GPUS == 1:
        net_par = hires_net.cuda()
    else:
        net_par = torch.nn.DataParallel(hires_net).cuda()

    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.00005)

    net_par.train()

    import time
    lasttime = [time.time()]

    def timeit(_):
        print(time.time() - lasttime[0])
        lasttime[0] = time.time()

    train_batchfunction(net_par, optimizer, lambda: (make_batch(dataset), make_batch(dataset)), unwrapped_net=hires_net, step_callback=timeit)

