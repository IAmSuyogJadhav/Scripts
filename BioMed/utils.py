import os
import glob
import cv2
import re
import tables
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import zoom


def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def resize(img, out_shape):
    d, h, w = img.shape
    factors = (out_shape[0] / d, out_shape[1] / h, out_shape[2] / w)
    img_ = zoom(img, factors)
    return img_

def normalize(img):
    return (img - img.mean()) / img.std()


def read_labels(labels_path):
    """
    For reading in labels of patches from ProstateX challenge.
    """
    labels = pd.read_csv(labels_path)
    labels = labels[['ProxID', 'Name', 'coord']]
    labels.dropna(inplace=True)

    labels['ID'] = labels.ProxID.map(lambda x: int(x[-4:]))
    labels['modality'] = labels.Name.map(lambda x: 'sag' if 'sag' in x else None)
    labels['coords'] = labels['coord'].map(lambda x: x.split()).map(lambda x: list(map(int, x)))
    labels = labels[['ID', 'modality', 'coords']]
    labels.dropna(inplace=True)
    return labels


def get_coords(id, labels=None):
    """
    Get ground truth coordinates for given training example ID.
    For use with ProstateX Challenge Dataset.
    """
    if labels is None:
        labels = read_labels(
            '/home/user/prostatechallenge/ProstateX2-DataInfo-Train/ProstateX-2-Images-Train.csv'
            )
    labels.set_index('ID')
    groups = labels.groupby('ID').groups
    return list(labels.coords[groups[2]])


def enlarge_label(img, coords, r=1, mode='area'):
    """
    Expects shape to be (modalities, depth, width, height)
    outputs segmentation of shape (1, depth, width, height)
    If mode is "volume", r should be a list of radii: [x_radius, y_radius z_radius]
    """
    seg = np.zeros_like(img[0]).astype(np.uint8)

    if mode == 'area':
        for coord in coords:
            frame = seg[coord[2] - 1, ...]
            frame = cv2.rectangle(
                frame, (coord[0]-r, coord[1]+r), (coord[0]+r, coord[1]-r), 1, -1)
            seg[coord[2] - 1, ...] = frame

    elif mode == 'volume':
        try:
            _ = len(r)
            assert _ == 3, "In volume mode, r should be a list of radii in each"\
                " dimension: [x_radius, y_radius z_radius]"
        except:
            assert 0, "In volume mode, r should be a list of radii in each"\
                " dimension: [x_radius, y_radius z_radius]"
        n_slices = seg.shape[0]
        for coord in coords:
            center = coord[2] - 1
            for i in range(max(center - r[2], 0), min(center + r[2] + 1, n_slices)):
                frame = seg[i, ...]
                frame = cv2.rectangle(
                    frame,
                    (coord[0]-r[0], coord[1]+r[0]), (coord[0]+r[1], coord[1]-r[1]), 1, -1
                    )
                seg[i, ...] = frame

    return seg


def save_nii(files, names, dir="./saved_nii"):
    """
    Saves input list of numpy arrays in nii.gz files.
    """
    writer = sitk.ImageFileWriter()
    os.makedirs(dir, exist_ok=True)
    for file, name in zip(files, names):
        path = os.path.join(dir, f'{name}.nii.gz' if not name.endswith('nii.gz') else name)
        writer.SetFileName(path)
        writer.Execute(sitk.GetImageFromArray(file))
        print(f"Succesfully saved {path}")


def get_last_state(models_dir, model_prefix='Model'):
    r"""
    Gives out the path to the last model and its epoch number.
    Expect models to be named as (regex): {model}-.*Epoch-\d*\.h5
    where {model} is the prefix of your model checkpoints, defaults to "Model".
    """
    path = os.path.join(models_dir, f"{model_prefix}-*.h5")
    models = glob.glob(path)

    re_escape = ['(', ')', '.']
    for ch in re_escape:
        model_prefix = model_prefix.replace(ch, '\\' + ch)
    
    if models:
        pat = re.compile(f'.*/{model_prefix}-.*Epoch-(\\d*)\\.h5')
        last_model = sorted(models, reverse=True, key=lambda m: int(pat.findall(m)[0]))[0]
        epoch = int(pat.findall(last_model)[0])
        print(f"Found last model at:{last_model}\nEpoch no.: {epoch}")
        return last_model, epoch
    else:
        print(f"No model found matching {path}")
        return os.path.join(models_dir, f"{model_prefix}-train_dice={{loss:.3f}}-val_dice={{val_loss:.3f}}-Epoch-{{epoch}}.h5"), 0


def save_data_file(data, labels, affine=None, h5_file='data_file.h5'):
    """
    For generating the data_file.h5 to be used with ellisdg/3DUnetCNN.

    Expected shapes
    ---------------
    data: (n_samples, modalities, channels, width, height)
    data: (n_samples, 1, channels, width, height)
    """
    f = tables.open_file(
        h5_file if h5_file.endswith('.h5') else f'{h5_file}.h5', 'w'
        )

    f.create_carray('/', 'data', obj=data)
    f.create_carray('/', 'truth', obj=labels)
    if affine is not None:
        f.create_carray('/', 'affine', obj=affine)
    f.close()


def flip_lr(img):
    """
    Expects shape to be (num_examples, modalities, depth, width, height)
    """
    return np.flip(img.copy(), 4)


def flip_ud(img):
    """
    Expects shape to be (num_examples, modalities, depth, width, height)
    """
    return np.flip(img.copy(), 3)


# The following function needs to be worked on before applying to MRI images
def elastic_transform(images, alpha, sigma, random_state=None):
    """
    ***Might not work corectly for lower resolution images***
    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    # assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    out = []
    for i in range(len(images)):
        out_ = []
        for j in [0, 1]:
            image = images[i, j, ...]  # Shape = [16, 32, 32]
            image = image.reshape((image.shape[1], image.shape[2], image.shape[0]))
            shape = image.shape  # = [32, 32, 16]

            dx = gaussian_filter((random_state.rand(*shape[:1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape[:1]) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

            out_.append(map_coordinates(image, indices, order=1).reshape(shape))
        out.append(np.array(out_))

    return np.array(out)


def get_affine(nii_path):
    """
    Get the affine matrix from a nii image.
    """
    return nib.load(nii_path).affine


def apply_affine(coords, affine):
    """
    Apply affine transform to the given `coords`(x, y, z coordinates). The
    affine matrix needs to be provided.
    """
    assert coords is not None and len(coords) == 3, 'Must provide a list of'\
        'coordinates: [x, y, z]'
    M = affine[:3, :3]
    abc = affine[:3, 3]

    return M.dot(coords) + abc
    # return affine.dot(img)  # This is correct too.
