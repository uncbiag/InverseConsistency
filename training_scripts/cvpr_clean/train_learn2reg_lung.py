import random
import os
import torch

import footsteps
import cvpr_network

import icon_registration as icon
import icon_registration.networks as networks
import icon_registration.data as data

BATCH_SIZE=1
GPUS = 4
#ITERATIONS_PER_STEP = 50000
ITERATIONS_PER_STEP = 301

def make_batch_paired(dataset, BATCH_SIZE):
    image = torch.stack([random.choice(dataset) for _ in range(BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image[:, 0:1], image[:, 1:2]

if __name__ == "__main__":
    footsteps.initialize()

    dataset = data.get_learn2reg_lungCT_dataset("/playpen-raid2/lin.tian/data/learn2reg/NLST", clamp=[-1000,-200], downscale=2)

    batch_function = lambda : make_batch_paired(dataset.tensors[0], GPUS*BATCH_SIZE)
    example = make_batch_paired(dataset.tensors[0], 1)
    input_shape = example[0].shape

    cvpr_network.train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP, BATCH_SIZE)