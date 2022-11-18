import random
import os
import torch

import footsteps
import cvpr_network

import icon_registration as icon
import icon_registration.networks as networks

input_shape = [1, 1, 80, 192, 192]

BATCH_SIZE=8
GPUS = 4
#ITERATIONS_PER_STEP = 50000
ITERATIONS_PER_STEP = 301

def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

if __name__ == "__main__":
    footsteps.initialize()

    dataset = torch.load("/name-ssd/name/knees_big_2xdownsample_train_set")

    batch_function = lambda : (make_batch(dataset), make_batch(dataset))

    cvpr_network.train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP, BATCH_SIZE)
