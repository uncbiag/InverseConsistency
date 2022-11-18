### To run this experiment, go to the root folder and run
### bash training_scripts/cvpr_clean/ablations/compute_model_statistics.sh

import sys
import torch
from icon_registration import networks, network_wrappers
from thop import clever_format, profile
from argparse import ArgumentParser

# import progressive_train
sys.path.append('./training_scripts/cvpr_clean')
from cvpr_network import make_network


device = torch.device('cuda:0')
inshape = (1,1,175,175,175)


def get_net(name, inshape):
    inshape = inshape
    if name == 'vm_unet':
        from unet_from_vm import Unet as vm_unet
        net = vm_unet(3)
    elif name == 'icon_unet':
        net = networks.tallUNet2(3)
    elif name == 'vm':
        import os
        os.environ['VXM_BACKEND'] = 'pytorch'
        import voxelmorph as vxm
        net = vxm.networks.VxmDense.load("/playpen-raid2/lin.tian/projects/icon_lung/ICON_lung_results/icon/comparing/vm_diff/2022_05_15_21_22_37/checkpoints/0100.pt", device=device)
    elif name == 'lapirn_disp':
        sys.path.append('/playpen-raid1/lin.tian/projects/pyReg/sotas/LapIRN/Code')
        from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_disp_add_lvl1, \
            Miccai2020_LDR_laplacian_unit_disp_add_lvl2, Miccai2020_LDR_laplacian_unit_disp_add_lvl3
        # Need to change the input shape during training, because LapIRN does not support odd number shape.
        imgshape = (176, 176, 176)
        imgshape_4 = (44, 44, 44)
        imgshape_2 = (88, 88, 88)
        inshape = (1,1,176,176,176)
        range_flow = 0.4
        model_lvl1 = Miccai2020_LDR_laplacian_unit_disp_add_lvl1(2, 3, 7, is_train=True, imgshape=imgshape_4,
                                                         range_flow=0.4).cuda()
        model_lvl2 = Miccai2020_LDR_laplacian_unit_disp_add_lvl2(2, 3, 7, is_train=True, imgshape=imgshape_2,
                                                                range_flow=0.4, model_lvl1=model_lvl1).cuda()

        net = Miccai2020_LDR_laplacian_unit_disp_add_lvl3(2, 3, 7, is_train=False, imgshape=imgshape,
                                                            range_flow=0.4, model_lvl2=model_lvl2).cuda()
    elif name == 'lapirn_diff':
        sys.path.append('/playpen-raid1/lin.tian/projects/pyReg/sotas/LapIRN/Code')
        from miccai2020_model_stage import Miccai2020_LDR_laplacian_unit_add_lvl1, \
            Miccai2020_LDR_laplacian_unit_add_lvl2, Miccai2020_LDR_laplacian_unit_add_lvl3
        # Need to change the input shape during training, because LapIRN does not support odd number shape.
        imgshape = (176, 176, 176)
        imgshape_4 = (44, 44, 44)
        imgshape_2 = (88, 88, 88)
        inshape = (1,1,176,176,176)
        range_flow = 0.4
        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(2, 3, 7, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow).cuda()
        model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(2, 3, 7, is_train=True, imgshape=imgshape_2,
                                            range_flow=range_flow, model_lvl1=model_lvl1).cuda()

        net = Miccai2020_LDR_laplacian_unit_add_lvl3(2, 3, 7, is_train=False, imgshape=imgshape,
                                            range_flow=range_flow, model_lvl2=model_lvl2).cuda()
    elif name == 'icon':
        import icon_registration
        class iconNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                phi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
                psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
                net = icon_registration.InverseConsistentNet(
                        network_wrappers.DoubleNet(
                            network_wrappers.DownsampleNet(network_wrappers.DoubleNet(phi, psi), dimension=3),
                            network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3)),
                        ),
                        icon_registration.ssd_only_interpolated,
                        1600,
                    )
                net.assign_identity_map(inshape)
                self.net = net.regis_net
                self.register_buffer('identity_map', net.identity_map, persistent=False)
            def forward(self, A, B):
                return self.net(A, B)(self.identity_map)
        net = iconNet()
    elif name == 'gradicon_stage1':
        class gradiconNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                net = make_network(inshape, include_last_step=False)
                self.net = net.regis_net
                self.register_buffer('identity_map', net.identity_map, persistent=False)
            def forward(self, A, B):
                return self.net(A, B)(self.identity_map)
        net = gradiconNet()
    elif name == 'gradicon':
        class gradiconNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                net = make_network(inshape, include_last_step=True)
                self.net = net.regis_net
                self.register_buffer('identity_map', net.identity_map, persistent=False)
            def forward(self, A, B):
                return self.net(A, B)(self.identity_map)
        net = gradiconNet()
    else:
        print(f"Model is not supported.")

    net = net.to(device)
    return net, inshape



def print_statistics(net, inshape, device):
    A = torch.rand(inshape).to(device)
    B = torch.rand(inshape).to(device)
    print(f"GPU memory usage after loading data: {torch.cuda.memory_allocated(0)/1024/1024} MB")

    net.eval()
    net(A,B)
    print(f"Peak memory usage after one forward pass: {(torch.cuda.max_memory_reserved(0))/1024/1024} MB")

    # Compute time
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(10):
        net(A, B)
    end.record()
    torch.cuda.synchronize()
    print(f"Elapsed time: {start.elapsed_time(end)/10.} millisec.")

    # Compute FLOPS
    macs, params = profile(net, inputs=(A, B))
    gflops, params = clever_format([macs, params], '%.3f')
    print(f"FLOPs: {gflops} G, #Params: {params}")

    # Compute Parameter numbers
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter count: {count_parameters(net)}")


parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="gradicon", help="name of the model")


args = parser.parse_args()
torch.cuda.empty_cache()
print(f"##################################### {args.model_name} ######################################")
print(f"GPU memory usage at initial: {torch.cuda.memory_allocated(0)/1024/1024} MB")
net, net_in_shape = get_net(args.model_name, inshape)
print(f"GPU memory usage after loading model: {torch.cuda.memory_allocated(0)/1024/1024} MB")
with torch.no_grad():
    print_statistics(net, net_in_shape, device)