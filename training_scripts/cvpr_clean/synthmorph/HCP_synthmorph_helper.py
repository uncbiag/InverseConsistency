import nibabel as nib
import numpy as np
import tensorflow as tf
import voxelmorph as vxm

# This script borrows some functions from https://github.com/freesurfer/freesurfer/blob/a810044ae08a24402436c1d43472b3b3df06592a/mri_synthmorph/mri_synthmorph

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


def net_to_vox(im, out_shape):
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


def transform(im, trans, shape, normalize=False, interp_method='linear'):
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