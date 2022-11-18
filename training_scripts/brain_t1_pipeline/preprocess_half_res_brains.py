import os
import argparse
import footsteps
footsteps.initialize(output_root="/name-ssd/name/ICON_brain_preprocessed_data/")


def process(iA, isSeg=False):
    iA = iA[None, None, :, :, :]
    iA = torch.nn.functional.avg_pool3d(iA, 4)
    return iA
downsample = 4
for split in ["train"]:
    with open(f"splits/{split}.txt") as f:
        image_paths = f.readlines()

    import torch

    import itk
    import tqdm
    import numpy as np
    import glob

    ds = []

    # /name-raid2/Data/HCP/HCP_1200/100206/T1w/T1w_acpc_dc.nii.gz
    # ==>
    # /name-raid2/Data/HCP/HCP_1200/100206/T1w/T1w_acpc_dc_restore_brain.nii.gz
    for name in tqdm.tqdm(list(iter(image_paths))[:]):
        name = name.split(".nii.gz")[0] + "_restore_brain.nii.gz"

        image = torch.tensor(np.asarray(itk.imread(name)))

        ds.append(process(image))

    torch.save(ds, f"{footsteps.output_dir}/brain_{split}_4xdown_scaled")

