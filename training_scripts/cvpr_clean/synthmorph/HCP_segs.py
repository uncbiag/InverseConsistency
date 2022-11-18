import random
import itk

import glob
paths = glob.glob("/author-raid2/Data/HCP/manual_subcortical_segmentations_BWH/*/*_*_novdc-label.nii.gz")
atlas_registered = [p.split("/")[-1].split("_")[0] for p in paths]

#paths = glob.glob("/author-raid1/author/Subcortical_Atlas_Fusion2/*WarpedLabels*")
#atlas_registered = [p.split("/malf3")[-1].split("_")[0] for p in paths]


def get_sub_seg(n):
    path = f"/author-raid2/Data/HCP/manual_subcortical_segmentations_BWH/{n}/{n}_*_novdc-label.nii.gz"
    path = glob.glob(path)
    print(path)
    if len(path) == 1:
        path = path[0]
    else:
        path = f"/author-raid1/author/Subcortical_Atlas_Fusion2/{n}_label.nii.gz"
    return itk.imread(path)

def get_brain_image(n):
    return itk.imread(f"/author-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz")

