import os
import random

import footsteps
footsteps.initialize(output_root="evaluation_results/")
import nibabel as nib
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

import utils
from HCP_synthmorph_data_prepare import process_image
from HCP_segs import atlas_registered, get_brain_image_path, get_sub_seg_path
from HCP_synthmorph_helper import net_to_vox, transform, read_affine

# This script borrows some functions from https://github.com/freesurfer/freesurfer/blob/a810044ae08a24402436c1d43472b3b3df06592a/mri_synthmorph/mri_synthmorph

if __name__ == "__main__":
    
    prealign_folder = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/cvpr_clean/evaluation_results/synthmorph_preprocessed"
    ref_path = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/cvpr_clean/ref.nii.gz"
    ref = vxm.py.utils.load_volfile(ref_path, add_batch_axis = True, add_feat_axis = True)

    in_shape = ref.shape[1:-1]
    nb_feats = ref.shape[-1]

    dices = []
    flips = []
    pair_list = []

    random.seed(1)
    for _ in range(100):
        n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
        if prealign_folder == "" or not os.path.exists(prealign_folder):
            prealign_folder = f"{footsteps.output_dir}/synthmorph_preprocessed"
            if not os.path.exists(prealign_folder):
                os.mkdir(prealign_folder)
            
            process_image(n_A, prealign_folder)
            process_image(n_B, prealign_folder)
        
        # Input data.
        mov = nib.load(f"{prealign_folder}/{n_A}_affine.nii.gz")
        fix = nib.load(f"{prealign_folder}/{n_B}_affine.nii.gz")

        # ################################# Prepare transformation from image to network ###########################################
        # Coordinate transforms. We will need these to take the images from their
        # native voxel spaces to network space. Voxel and network spaces are different
        # for each image. Network space is an isotropic 1-mm space centered on the
        # original image. Its axes are aligned with the original voxel data but flipped
        # and swapped to gross LIA orientation, which the network will expect.
        net_to_mov = net_to_vox(mov, out_shape=in_shape)
        net_to_fix = net_to_vox(fix, out_shape=in_shape)
        mov_to_net = np.linalg.inv(net_to_mov)
        fix_to_net = np.linalg.inv(net_to_fix)

        # Transforms from and to world space (RAS). There is only one world.
        mov_to_ras = mov.affine
        fix_to_ras = fix.affine
        ras_to_mov = np.linalg.inv(mov_to_ras)
        ras_to_fix = np.linalg.inv(fix_to_ras)

        # Transforms between zero-centered and zero-based voxel spaces.
        ind_to_cen = np.eye(4)
        ind_to_cen[:-1, -1] = -0.5 * (np.asarray(in_shape) - 1)
        cen_to_ind = np.eye(4)
        cen_to_ind[:-1, -1] = +0.5 * (np.asarray(in_shape) - 1)

        # # Incorporate an initial linear transform operating in RAS. It goes from fixed
        # # to moving coordinates, so we start with fixed network space on the right.
        # if arg.init:
        #     aff = np.loadtxt(arg.init)
        #     net_to_mov = ras_to_mov @ aff @ fix_to_ras @ net_to_fix

        ################################# Run networks on the transformed image ###########################################
        fix_affine_ras = read_affine(f"{prealign_folder}/{n_B}_Affine.lta")
        mov_affine_ras = read_affine(f"{prealign_folder}/{n_A}_Affine.lta")
        pre_fix = nib.load(get_brain_image_path(n_B))
        pre_mov = nib.load(get_brain_image_path(n_A))

        ################################# Compose all the transformations  ###########################################
        # Construct grid of zero-based index coordinates and shape (3, N) in native
        # fixed voxel space, where N is the number of voxels.
        x_fix = (tf.range(x, dtype=tf.float32) for x in pre_fix.shape)
        x_fix = tf.meshgrid(*x_fix, indexing='ij')
        x_fix = tf.stack(x_fix)
        x_fix = tf.reshape(x_fix, shape=(3, -1))

        # Transform x_fix from previous fix to after affine fix
        pre_fix_to_fix = np.linalg.inv(fix.affine)@fix_affine_ras@pre_fix.affine
        x_out = pre_fix_to_fix[:-1, -1:] + (pre_fix_to_fix[:-1, :-1] @ x_fix)

        # Transform fixed voxel coordinates to the fixed network space.
        x_out = fix_to_net[:-1, -1:] + (fix_to_net[:-1, :-1] @ x_out)
        # x_out = tf.transpose(x_out)

        # # Add predicted warp to coordinates to go to the moving network space.
        # trans = tf.squeeze(trans)
        # x_out += ne.utils.interpn(trans, x_out, fill_value=0)
        # x_out = tf.transpose(x_out)

        # Transform coordinates to the native moving voxel space. Subtract fixed
        # coordinates to obtain displacement from fixed to moving voxel space.
        x_out = net_to_mov[:-1, -1:] + (net_to_mov[:-1, :-1] @ x_out)

        # Transform to before affine move
        mov_to_pre_mov = np.linalg.inv(pre_mov.affine)@np.linalg.inv(mov_affine_ras)@mov.affine
        x_out = mov_to_pre_mov[:-1, -1:] + (mov_to_pre_mov[:-1, :-1] @ x_out)

        trans_vox = tf.transpose(x_out - x_fix)
        trans_vox = tf.reshape(trans_vox, shape=(*pre_fix.shape, -1))

        # Displacement from fixed to moving RAS coordinates.
        x_ras = fix_to_ras[:-1, -1:] + (fix_to_ras[:-1, :-1] @ x_fix)
        x_out = mov_to_ras[:-1, -1:] + (mov_to_ras[:-1, :-1] @ x_out)
        trans_ras = tf.transpose(x_out - x_ras)
        trans_ras = tf.reshape(trans_ras, shape=(*pre_fix.shape, -1))

        ################################# Evaluate the segmentations  ###########################################
        mov = nib.load(get_sub_seg_path(n_A))
        fixed = nib.load(get_sub_seg_path(n_B))
        fixed_np = fixed.get_fdata(dtype=np.float32)

        warped_seg = transform(mov, trans=trans_vox, shape=fixed_np.shape, interp_method='nearest').numpy()[0,:,:,:,0]
        
        dice = utils.mean_dice(fixed_np, warped_seg)

        jd = vxm.py.utils.jacobian_determinant(trans_vox)
        flip = np.mean(jd<0) * 100.0

        flips.append(flip)
        dices.append(dice)
        utils.log(f"{_}/100 mean DICE {n_A} to {n_B}: {dice} | running DICE: {np.mean(dices)} | Percentage: {flip} | running Per: {np.mean(flips)} ")
        pair_list.append([n_A, n_B])

    utils.log("Mean DICE")
    utils.log(f"Final DICE: {np.mean(dices)} | final percentage of negative jacobian: {np.mean(flips)}")

    with open(f'{footsteps.output_dir}/pair_list.txt', 'w') as f:
        for p in pair_list:
            f.write(",".join(p)+'\n')
