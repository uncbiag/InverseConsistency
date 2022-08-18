from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
import torch

# import progressive_train
import halfres_train_lung as progressive_train
import icon_registration
import icon_registration.pretrained_models as pretrained_models

device = torch.device('cuda:2')
inshape = (1,1,175,175,175)

# import icon_registration.network_wrappers as network_wrappers
# import icon_registration.networks as networks
# import icon_registration.inverseConsistentNet as inverseConsistentNet
# phi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
# psi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))
# xi = network_wrappers.FunctionFromVectorField(networks.tallUNet2(dimension=3))

# net = inverseConsistentNet.InverseConsistentNet(
#     network_wrappers.DoubleNet(
#         network_wrappers.DownsampleNet(network_wrappers.DoubleNet(phi, psi), 3),
#         xi,
#     ),
#     inverseConsistentNet.ncc,
#     1300,
# )
# network_wrappers.assignIdentityMap(net, inshape)

net = progressive_train.build_framework(progressive_train.make_2x_net(), inshape)
icon_registration.network_wrappers.adjust_batch_size(net, 1)
net.to(device)

# Compute time
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

A = torch.rand(inshape).to(device)
B = torch.rand(inshape).to(device)
start.record()
for i in range(10):
    net(A, B)
end.record()
torch.cuda.synchronize()
print(f"Elapsed time: {start.elapsed_time(end)/10.} millisec.")

# Compute FLOPS
flops = FlopCountAnalysis(net, (A, B))
print(f"Flops:{flops.total()/1e9} GFLOPs")

# Compute Parameter numbers
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
print(f"Parameter count: {count_parameters(net.regis_net)}")

# Compute forward memory consumption
summary(net, [inshape, inshape], dtypes=[torch.float, torch.float], device=device, depth=6)