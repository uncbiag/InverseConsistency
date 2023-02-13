
import random

import nibabel as nib
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

from HCP_synthmorph_helper import save, transform, read_affine

# This script borrows some functions from https://github.com/freesurfer/freesurfer/blob/a810044ae08a24402436c1d43472b3b3df06592a/mri_synthmorph/mri_synthmorph

ref = vxm.py.utils.load_volfile("ref.nii.gz", add_batch_axis = True, add_feat_axis = True)

in_shape = ref.shape[1:-1]
nb_feats = ref.shape[-1]

data_folder = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/results/eval_HCP/vm_shape"
with tf.device(vxm.tf.utils.setup_device("2")[0]):

    from HCP_segs import atlas_registered


    random.seed(1)
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))

    pair_dir = f"{data_folder}/{n_A}_{n_B}"
    ori_fix = nib.load(f"{pair_dir}/imageB_orig.nii.gz")
    ori_mov = nib.load(f"{pair_dir}/imageA_orig.nii.gz")
    fix = nib.load(f"{pair_dir}/B_affine.nii.gz")
    mov = nib.load(f"{pair_dir}/A_affine.nii.gz")


    fix_affine_ras = read_affine(f"{pair_dir}/B_Affine.lta")
    mov_affine_ras = read_affine(f"{pair_dir}/A_Affine.lta")

    ################################# Test pre_fix_to_fix  ###########################################
    x_fix = (tf.range(x, dtype=tf.float32) for x in ori_fix.shape)
    x_fix = tf.meshgrid(*x_fix, indexing='ij')
    x_fix = tf.stack(x_fix)
    x_fix = tf.reshape(x_fix, shape=(3, -1))

    # Transform x_fix from previous fix to after affine fix
    pre_fix_to_fix = np.linalg.inv(fix.affine)@fix_affine_ras@ori_fix.affine
    x_out = pre_fix_to_fix[:-1, -1:] + (pre_fix_to_fix[:-1, :-1] @ x_fix)

    trans_vox = tf.transpose(x_out - x_fix)
    trans_vox = tf.reshape(trans_vox, shape=(*ori_fix.shape, -1))

    warped = transform(fix, trans=trans_vox, shape=ori_fix.shape).numpy()[0,:,:,:,0] 
    save("./warped_back_to_imageB_orig.nii.gz", warped, ori_fix.affine, float)

    ################################# Test mov_to_pre_mov  ###########################################
    x_mov = (tf.range(x, dtype=tf.float32) for x in mov.shape)
    x_mov = tf.meshgrid(*x_mov, indexing='ij')
    x_mov = tf.stack(x_mov)
    x_mov = tf.reshape(x_mov, shape=(3, -1))

    # Transform to before affine move
    mov_to_pre_mov = np.linalg.inv(ori_mov.affine)@np.linalg.inv(mov_affine_ras)@mov.affine
    x_out = mov_to_pre_mov[:-1, -1:] + (mov_to_pre_mov[:-1, :-1] @ x_mov)

    trans_vox = tf.transpose(x_out - x_mov)
    trans_vox = tf.reshape(trans_vox, shape=(*mov.shape, -1))

    warped = transform(ori_mov, trans=trans_vox, shape=mov.shape).numpy()[0,:,:,:,0] 
    save("./warped_back_to_imageA_affine.nii.gz", warped, mov.affine, float)
 