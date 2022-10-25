import random
import torch.nn.functional as F
import os
import torch

import footsteps
import cvpr_network

import icon_registration as icon
import icon_registration.networks as networks
import icon_registration.data as data

BATCH_SIZE=1
GPUS = 4
ITERATIONS_PER_STEP = 10000
#ITERATIONS_PER_STEP = 301

def make_batch_paired(dataset, BATCH_SIZE):
    image = torch.stack([random.choice(dataset) for _ in range(BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image[:, 0:1], image[:, 1:2]

def augment(image_A, image_B):
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.Tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
        identity = identity * (torch.randint_like(identity, 0, 2) * 2  - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    
    noise = torch.randn((image_A.shape[0], 3, 4))

    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
   
    warped_A = F.grid_sample(image_A, forward_grid, padding_mode='border')

    noise = torch.randn((image_A.shape[0], 3, 4))
    forward = identity + .05 * noise  

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward.cuda(), grid_shape)
    warped_B = F.grid_sample(image_B, forward_grid, padding_mode='border')

    return warped_A, warped_B



if __name__ == "__main__":
    footsteps.initialize()

    dataset = data.get_learn2reg_lungCT_dataset("/playpen-raid2/lin.tian/data/learn2reg/NLST", clamp=[-1000,-200], downscale=1)
    def batch_function():
        a, b = make_batch_paired(dataset.tensors[0], GPUS*BATCH_SIZE)
        a, b = augment(a, b)
        return a, b

    example = make_batch_paired(dataset.tensors[0], 1)
    input_shape = example[0].shape
    print(input_shape)

    cvpr_network.train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP, BATCH_SIZE)
