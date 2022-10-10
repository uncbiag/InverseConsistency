import random
import os
import torch
import numpy as np

import sys
sys.path.append('./training_scripts/cvpr_clean')

import footsteps
import cvpr_network

import icon_registration as icon
import icon_registration.networks as networks
from icon_registration import data

# data_shape = 128
# input_shape = [1, 1, data_shape, data_shape]
input_shape = [1, 1, 80, 192, 192]

BATCH_SIZE=2
GPUS = 4
#ITERATIONS_PER_STEP = 50000
ITERATIONS_PER_STEP = 301

class data_ite():
    def __init__(self, dataset):
        self.data_ite = iter(dataset)
    
    def make_batch(self):
        image = next(self.data_ite)[0]
        image = image.cuda() + 1.0 # Set background to 0
        return image

def make_batch_for_knee(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    return image

if __name__ == "__main__":
    footsteps.initialize()

    dataset = torch.load("/playpen-ssd/tgreer/knees_big_2xdownsample_train_set")
    batch_function = lambda : (make_batch_for_knee(dataset), make_batch_for_knee(dataset))

    # Toy dataset
    # train_I1, train_I2 = data.get_dataset_triangles(data_size=data_shape, samples=100, batch_size=GPUS*BATCH_SIZE, hollow=False)
    # batch_function = lambda : (data_ite(train_I1).make_batch(), data_ite(train_I2).make_batch())

    cvpr_network.train_two_stage(input_shape, batch_function, GPUS, ITERATIONS_PER_STEP, BATCH_SIZE, framework='icon')

