import glob
data_root = "/author-raid2/Data/HCP/"

image_paths = glob.glob(
  "/author-raid2/Data/HCP/HCP_1200/*/T1w/T1w_acpc_dc.nii.gz")
#print(image_paths)
test_paths = glob.glob(
  "/author-raid2/Data/HCP/manual_subcortical_segmentations_BWH/*")
test_set = set([s.split("/")[-1] for s in test_paths])

all_names = [path.split("/")[-3] for path in image_paths]


train_names = set([name for name in all_names if not name in test_set])


train_paths = [path for path in image_paths if (path.split("/")[-3] in train_names)]


with open("train.txt", "w") as f:
    for path in train_paths:
        f.write(path + "\n")
