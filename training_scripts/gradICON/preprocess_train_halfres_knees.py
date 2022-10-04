import os

with open("../oai_paper_pipeline/splits/train/pair_name_list.txt") as f:
    train_pair_names = f.readlines()
with open("../oai_paper_pipeline/splits/train/pair_path_list.txt") as f:
    train_pair_paths = f.readlines()


train_paths = set()

[
    [train_paths.add(p) for p in [pp.split()[0], pp.split()[1]]]
    for pp in train_pair_paths
]


import torch

import itk
import tqdm
import numpy as np

ds = []
for tt in tqdm.tqdm(list(iter(train_paths))[:]):
    tt = tt.replace("playpen", "playpen-raid")
    image = torch.tensor(np.asarray(itk.imread(tt)))
    if "RIGHT" in tt:
        image = torch.flip(image, [0])
    elif "LEFT" in tt:
        pass
    else:
        raise AssertionError()

    iA = image[None, None, :, :, :]

    iA = torch.nn.functional.avg_pool3d(iA, 2)

    ds.append(iA)

torch.save(ds, "/playpen-ssd/tgreer/knees_big_2xdownsample_train_set")
