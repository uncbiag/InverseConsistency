import subprocess
from HCP_segs import atlas_registered, get_brain_image_path, get_sub_seg_path
import os

'''
To run mri_robust_register, you need first source /playpen-raid1/tgreer/freesurfer/activate.sh
'''

def process_image(n, output_folder):
    ref_path = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/cvpr_clean/ref.nii.gz"

    image_path = get_brain_image_path(n)
    label_path = get_sub_seg_path(n)

    subprocess.run(f"mri_robust_register --mov {image_path} --dst {ref_path} -lta {output_folder}/{n}_Affine.lta --satit --iscale --verbose 0", shell=True)
    subprocess.run(f"mri_robust_register --mov {image_path} --dst {ref_path} -lta {output_folder}/{n}_Affine.lta --satit --iscale --ixform {output_folder}/{n}_Affine.lta --affine --verbose 0", shell=True)
    subprocess.run(f"mri_vol2vol --mov {image_path} --o {output_folder}/{n}_affine.nii.gz --lta {output_folder}/{n}_Affine.lta --targ {ref_path}", shell=True)
    subprocess.run(f"mri_vol2vol --mov {label_path} --o {output_folder}/{n}_label_affine.nii.gz --lta {output_folder}/{n}_Affine.lta  --targ {ref_path} --nearest --keep-precision", shell=True)

if __name__ == "__main__":
    prealign_folder = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/cvpr_clean/evaluation_results/synthmorph_preprocessed"
    override = False
    
    if not os.path.exists(prealign_folder):
        os.makedirs(prealign_folder)
    
    for n in atlas_registered:
        if not override and os.path.exists(f"{prealign_folder}/{n}_affine.nii.gz"):
            continue
        process_image(n, prealign_folder)
            

