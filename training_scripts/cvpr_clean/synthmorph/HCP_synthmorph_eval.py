import argparse
import glob
import os
import random
import subprocess
import sys
import time

import footsteps
import itk
import neurite as ne
import nibabel as nib
# footsteps.initialize(output_root="evaluation_results/", run_name="asdf")
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

import utils
import voxelmorph as vxm

# This script borrows some functions from https://github.com/freesurfer/freesurfer/blob/a810044ae08a24402436c1d43472b3b3df06592a/mri_synthmorph/mri_synthmorph

ref = vxm.py.utils.load_volfile("ref.nii.gz", add_batch_axis = True, add_feat_axis = True)

in_shape = ref.shape[1:-1]
nb_feats = ref.shape[-1]

def save(path, dat, affine, dtype=None):
    # Usi NiBabel's caching functionality to avoid re-reading from disk.
    if isinstance(dat, nib.filebasedimages.FileBasedImage):
        if dtype is None:
            dtype = dat.dataobj.dtype
        dat = dat.get_fdata(dtype=np.float32)

    dat = np.squeeze(dat)
    dat = np.asarray(dat, dtype)

    # Avoid warning about missing units when reading with FS.
    out = nib.Nifti1Image(dat, affine)
    out.header.set_xyzt_units(xyz='mm', t='sec')
    nib.save(out, filename=path)


def ori_to_ori(old, new='LIA', old_shape=None, zero_center=False):
    '''Construct matrix transforming coordinates from a voxel space with a new
    predominant anatomical axis orientation to an old orientation, by swapping
    and flipping axes. Operates in zero-based index space unless the space is
    to be zero-centered. The old shape must be specified if the old image is
    not a NiBabel object.'''
    def extract_ori(x):
        if isinstance(x, nib.filebasedimages.FileBasedImage):
            x = x.affine
        if isinstance(x, np.ndarray):
            return nib.orientations.io_orientation(x)
        if isinstance(x, str):
            return nib.orientations.axcodes2ornt(x)

    # Old shape.
    if zero_center:
        old_shape = (1, 1, 1)
    if old_shape is None:
        old_shape = old.shape

    # Transform from new to old index coordinates.
    old = extract_ori(old)
    new = extract_ori(new)
    new_to_old = nib.orientations.ornt_transform(old, new)
    return nib.orientations.inv_ornt_aff(new_to_old, old_shape)


def net_to_vox(im, out_shape=in_shape):
    '''Construct coordinate transform from isotropic 1-mm voxel space with
    gross LIA orentiation centered on the FOV - to the original image index
    space. The target space is a scaled and shifted voxel space, not world
    space.'''
    if isinstance(im, str):
        im = nib.load(im)

    # Gross LIA to predominant anatomical orientation of input image.
    assert isinstance(im, nib.filebasedimages.FileBasedImage) 
    lia_to_ori = ori_to_ori(im, new='LIA', old_shape=out_shape)

    # Scaling from millimeter to input voxels.
    vox_size = np.sqrt(np.sum(im.affine[:-1, :-1] ** 2, axis=0))
    scale = np.diag((*1 / vox_size, 1))

    # Shift from cen
    shift = np.eye(4)
    shift[:-1, -1] = 0.5 * (im.shape - out_shape / vox_size)

    # Total transform.
    return shift @ scale @ lia_to_ori


def transform(im, trans, shape=in_shape, normalize=False, interp_method='linear'):
    '''Apply transformation matrix or field operating in zero-based index space
    to an image.'''
    if isinstance(im, nib.filebasedimages.FileBasedImage):
        im = im.get_fdata(dtype=np.float32)

    # Add singleton feature dimension if needed.
    if tf.rank(im) == 3:
        im = im[..., tf.newaxis]

    # Remove last row of matrix transforms.
    if tf.rank(trans) == 2 and trans.shape[0] == trans.shape[1]:
        trans = trans[:-1, :]

    out = vxm.utils.transform(
        im, trans, interp_method=interp_method, fill_value=0, shift_center=False, shape=shape,
    )

    if normalize:
        out -= tf.reduce_min(out)
        out /= tf.reduce_max(out)
    return out[tf.newaxis, ...]



def vm_dense(
    in_shape=None,
    input_model=None,
    enc_nf=[256] * 4,
    dec_nf=[256] * 4,
    add_nf=[256] * 4,
    int_steps=5,
    upsample=True,
    half_res=True,
):
    '''Deformable registration network.'''
    if input_model is None:
        source = tf.keras.Input(shape=(*in_shape, 1))
        target = tf.keras.Input(shape=(*in_shape, 1))
        input_model = tf.keras.Model(*[(source, target)] * 2)
    source, target = input_model.outputs[:2]

    in_shape = np.asarray(source.shape[1:-1])
    num_dim = len(in_shape)
    assert num_dim in (2, 3), 'only 2D and 3D supported'

    down = getattr(tf.keras.layers, f'MaxPool{num_dim}D')()
    up = getattr(tf.keras.layers, f'UpSampling{num_dim}D')()
    act = tf.keras.layers.LeakyReLU(0.2)
    conv = getattr(tf.keras.layers, f'Conv{num_dim}D')
    prop = dict(kernel_size=3, padding='same')

    # Encoder.
    x = tf.keras.layers.concatenate((source, target))
    if half_res:
        x = down(x)
    enc = [x]
    for n in enc_nf:
        x = conv(n, **prop)(x)
        x = act(x)
        enc.append(x)
        x = down(x)

    # Decoder.
    for n in dec_nf:
        x = conv(n, **prop)(x)
        x = act(x)
        x = tf.keras.layers.concatenate([up(x), enc.pop()])

    # Additional convolutions.
    for n in add_nf:
        x = conv(n, **prop)(x)
        x = act(x)

    # Transform.
    x = conv(num_dim, **prop)(x)
    if int_steps > 0:
        x = vxm.layers.VecInt(method='ss', int_steps=int_steps)(x)

    # Rescaling.
    zoom = source.shape[1] // x.shape[1]
    if upsample and zoom > 1:
        x = vxm.layers.RescaleTransform(zoom)(x)

    return tf.keras.Model(input_model.inputs, outputs=x)

def read_affine(lta_path=""):
    with open(lta_path, 'rb') as f:
        lines = f.readlines()
        affine = lines[8:12]
        affine = [str(i).split("'")[1].split(' ')[:-1] for i in affine]
        
    return np.array(affine, np.float)


with tf.device(vxm.tf.utils.setup_device("0")[0]):
    # model_path = "/playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5"
    model_path = "/playpen-raid2/lin.tian/projects/icon_lung/ICON/training_scripts/cvpr_clean/voxelmorph/shapes-dice-vel-3-res-8-16-32-256f.h5"

    #model_path = "/playpen-raid1/tgreer/voxelmorph/vxm_dense_brain_T1_3D_mse.h5"
    #model_path = "shapes-dice-vel-3-res-8-16-32-256f.h5"
    regis_net = vxm.networks.VxmDense.load(model_path)

    dices = []


    from HCP_segs import atlas_registered, get_brain_image, get_sub_seg

    def mean_dice_f(sA, sB):
        sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
        return utils.itk_mean_dice(sA, sB)


    random.seed(1)
    for _ in range(100):
        n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
        image_A, image_B = (
                itk.imread(
                    f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
                )
            for n in (n_A, n_B)
        )

        segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

        pair_dir = f"{footsteps.output_dir}/{n_A}_{n_B}"
        if not os.path.exists(pair_dir):
            os.mkdir(pair_dir)
        itk.imwrite(segmentation_A, f"{pair_dir}/segA_orig.nii.gz")
        itk.imwrite(image_A, f"{pair_dir}/imageA_orig.nii.gz")
        itk.imwrite(segmentation_B, f"{pair_dir}/segB_orig.nii.gz")
        itk.imwrite(image_B, f"{pair_dir}/imageB_orig.nii.gz")

        # Affine pre-aling to reference image
        subprocess.run(f"mri_robust_register --mov {pair_dir}/imageA_orig.nii.gz --dst ref.nii.gz -lta {pair_dir}/A_Affine.lta --satit --iscale --verbose 0", shell=True)
        subprocess.run(f"mri_robust_register --mov {pair_dir}/imageA_orig.nii.gz --dst ref.nii.gz -lta {pair_dir}/A_Affine.lta --satit --iscale --ixform {pair_dir}/A_Affine.lta --affine --verbose 0", shell=True)
        subprocess.run(f"mri_vol2vol --mov {pair_dir}/imageA_orig.nii.gz --o {pair_dir}/A_affine.nii.gz --lta {pair_dir}/A_Affine.lta --targ ref.nii.gz", shell=True)
        subprocess.run(f"mri_vol2vol --mov {pair_dir}/segA_orig.nii.gz --o {pair_dir}/Aseg_affine.nii.gz --lta {pair_dir}/A_Affine.lta --targ ref.nii.gz --nearest --keep-precision", shell=True)

        subprocess.run(f"mri_robust_register --mov {pair_dir}/imageB_orig.nii.gz --dst ref.nii.gz -lta {pair_dir}/B_Affine.lta --satit --iscale --verbose 0", shell=True)
        subprocess.run(f"mri_robust_register --mov {pair_dir}/imageB_orig.nii.gz --dst ref.nii.gz -lta {pair_dir}/B_Affine.lta --satit --iscale --ixform {pair_dir}/B_Affine.lta --affine --verbose 0", shell=True)
        subprocess.run(f"mri_vol2vol --mov {pair_dir}/imageB_orig.nii.gz --o {pair_dir}/B_affine.nii.gz --lta {pair_dir}/B_Affine.lta --targ ref.nii.gz", shell=True)
        subprocess.run(f"mri_vol2vol --mov {pair_dir}/segB_orig.nii.gz --o {pair_dir}/Bseg_affine.nii.gz --lta {pair_dir}/B_Affine.lta --targ ref.nii.gz --nearest --keep-precision", shell=True)

        #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model /playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5 --warp warp.nii.gz"""
        #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A_affine.nii.gz --moving B_affine.nii.gz --moved out.nii.gz --model shapes-dice-vel-3-res-8-16-32-256f.h5 --warp warp.nii.gz"""
        #subprocess.run(cmd, shell=True)

        
        # Input data.
        mov = nib.load(f"{pair_dir}/A_affine.nii.gz")
        fix = nib.load(f"{pair_dir}/B_affine.nii.gz")

        # ################################# Prepare transformation from image to network ###########################################
        # Coordinate transforms. We will need these to take the images from their
        # native voxel spaces to network space. Voxel and network spaces are different
        # for each image. Network space is an isotropic 1-mm space centered on the
        # original image. Its axes are aligned with the original voxel data but flipped
        # and swapped to gross LIA orientation, which the network will expect.
        net_to_mov = net_to_vox(mov)
        net_to_fix = net_to_vox(fix)
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
        inputs = (
            transform(mov, net_to_mov, shape=in_shape, normalize=True),
            transform(fix, net_to_fix, shape=in_shape, normalize=True),
        )

        trans = regis_net.register(*inputs) 

        fix_affine_ras = read_affine(f"{pair_dir}/B_Affine.lta")
        mov_affine_ras = read_affine(f"{pair_dir}/A_Affine.lta")
        pre_fix = nib.load(f"{pair_dir}/imageB_orig.nii.gz")
        pre_mov = nib.load(f"{pair_dir}/imageA_orig.nii.gz")

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
        x_out = tf.transpose(x_out)

        # Add predicted warp to coordinates to go to the moving network space.
        trans = tf.squeeze(trans)
        x_out += ne.utils.interpn(trans, x_out, fill_value=0)
        x_out = tf.transpose(x_out)

        # Transform coordinates to the native moving voxel space. Subtract fixed
        # coordinates to obtain displacement from fixed to moving voxel space.
        x_out = net_to_mov[:-1, -1:] + (net_to_mov[:-1, :-1] @ x_out)
        # x_out = tf.transpose(x_out)

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
        mov = nib.load(f"{pair_dir}/segA_orig.nii.gz")
        fixed = nib.load(f"{pair_dir}/segB_orig.nii.gz")
        fixed_np = fixed.get_data().squeeze()

        warped_seg = transform(mov, trans=trans_vox, shape=fixed_np.shape, interp_method='nearest').numpy()[0,:,:,:,0] 
        save(f"{pair_dir}/warped_segA_orig.nii.gz", warped_seg, fixed.affine, int)
        
        overlap = vxm.py.utils.dice(fixed_np, warped_seg, labels=list(range(1, 29)))

        mean_dice = np.mean(overlap)
    
        dices.append(mean_dice)
        utils.log(f"############## mean DICE {n_A} to {n_B}: {mean_dice} | running DICE: {np.mean(dices)}")

    utils.log("Mean DICE")
    utils.log(f"Final DICE: {np.mean(dices)}")




# import footsteps
# import subprocess
# import glob
# import sys
# import random
# import itk

# # footsteps.initialize(output_root="evaluation_results/", run_name="asdf")
# import numpy as np
# import utils

# import os
# import argparse
# import numpy as np
# import voxelmorph as vxm
# import tensorflow as tf
# device, nb_devices = vxm.tf.utils.setup_device("0")
# import voxelmorph as vxm

# ref = vxm.py.utils.load_volfile("ref.nii.gz", add_batch_axis = True, add_feat_axis = True)

# inshape = ref.shape[1:-1]
# nb_feats = ref.shape[-1]


# with tf.device(vxm.tf.utils.setup_device("0")[0]):
#     model_path = "/playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5"

#     #model_path = "/playpen-raid1/tgreer/voxelmorph/vxm_dense_brain_T1_3D_mse.h5"
#     #model_path = "shapes-dice-vel-3-res-8-16-32-256f.h5"
#     regis_net = vxm.networks.VxmDense.load(model_path)
#     warper_label = vxm.networks.Transform(
#           ref.shape[1:-1], interp_method="nearest"
#         )   
#     warper = vxm.networks.Transform(inshape, nb_feats=nb_feats)
#     def voxelmorph_register(moving_p, fixed_p):

#         moving = vxm.py.utils.load_volfile(moving_p, add_batch_axis=True, add_feat_axis=True)
#         fixed = vxm.py.utils.load_volfile(
#               fixed_p, add_batch_axis=True, add_feat_axis=True)
#         moving = moving / np.max(moving)
#         fixed = fixed / np.max(fixed)

#         print(np.min(moving), np.max(moving))
#         warp = regis_net.register(moving, fixed)
#         #moved = warper.predict([moving, warp])
#         return warp




#     dices = []


#     from HCP_segs import (atlas_registered, get_sub_seg, get_brain_image)

#     def mean_dice_f(sA, sB):
#         sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
#         return utils.itk_mean_dice(sA, sB)


#     random.seed(1)
#     for _ in range(100):
#         n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
#         image_A, image_B = (
#                 itk.imread(
#                     f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore_brain.nii.gz"
#                 )
#             for n in (n_A, n_B)
#         )

#         segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

#         itk.imwrite(segmentation_A, "segA_orig.nii.gz")
#         itk.imwrite(image_A, "imageA_orig.nii.gz")
#         itk.imwrite(segmentation_B, "segB_orig.nii.gz")
#         itk.imwrite(image_B, "imageB_orig.nii.gz")


#         subprocess.run("mri_robust_register --mov imageA_orig.nii.gz --dst ref.nii.gz -lta A_Affine.lta --satit --iscale", shell=True)
#         subprocess.run("mri_robust_register --mov imageA_orig.nii.gz --dst ref.nii.gz -lta A_Affine.lta --satit --iscale --ixform A_Affine.nii.gz --affine", shell=True)
#         subprocess.run("mri_vol2vol --mov imageA_orig.nii.gz --o A_affine.nii.gz --lta A_Affine.lta --targ ref.nii.gz", shell=True)
#         subprocess.run("mri_vol2vol --mov segA_orig.nii.gz --o Aseg_affine.nii.gz --lta A_Affine.lta --targ ref.nii.gz --nearest, --keep-precision", shell=True)

#         subprocess.run("mri_robust_register --mov imageB_orig.nii.gz --dst ref.nii.gz -lta B_Affine.lta --satit --iscale", shell=True)
#         subprocess.run("mri_robust_register --mov imageB_orig.nii.gz --dst ref.nii.gz -lta B_Affine.lta --satit --iscale --ixform B_affine.nii.gz --affine", shell=True)
#         subprocess.run("mri_vol2vol --mov imageB_orig.nii.gz --o B_affine.nii.gz --lta B_Affine.lta --targ ref.nii.gz", shell=True)
#         subprocess.run("mri_vol2vol --mov segB_orig.nii.gz --o Bseg_affine.nii.gz --lta B_Affine.lta --targ ref.nii.gz --nearest --keep-precision", shell=True)

#         #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A.nii.gz --moving B.nii.gz --moved out.nii.gz --model /playpen-raid1/tgreer/voxelmorph/brains-dice-vel-0.5-res-16-256f.h5 --warp warp.nii.gz"""
#         #cmd = """python /playpen-raid1/tgreer/voxelmorph/voxelmorph/scripts/tf/register.py --fixed A_affine.nii.gz --moving B_affine.nii.gz --moved out.nii.gz --model shapes-dice-vel-3-res-8-16-32-256f.h5 --warp warp.nii.gz"""
#         #subprocess.run(cmd, shell=True)

#         import voxelmorph
#         import voxelmorph as vxm


#         vsegfix = voxelmorph.py.utils.load_labels("Aseg_affine.nii.gz")[1][0][None, :, :, :, None]
#         vsegmov = voxelmorph.py.utils.load_labels("Bseg_affine.nii.gz")[1][0][None, :, :, :, None]
#         warp = voxelmorph_register("B_affine.nii.gz", "A_affine.nii.gz") 
#         warped_seg = warper_label.predict([vsegmov, warp])
#         overlap = vxm.py.utils.dice(vsegfix, warped_seg, labels=list(range(1, 29)))

#         mean_dice = np.mean(overlap)
#         dices.append(np.mean(overlap))

#         utils.log(mean_dice)

#         dices.append(mean_dice)

#     utils.log("Mean DICE")
#     utils.log(np.mean(dices))
