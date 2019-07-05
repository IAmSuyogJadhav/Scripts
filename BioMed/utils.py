import os
import glob
import cv2
import re
import tables
import pandas as pd
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def read_labels(labels_path):
    """
    For reading in labels of patches from ProstateX challenge.
    """
    labels = pd.read_excel(labels_path)
    labels.dropna(inplace=True)

    labels.ProxID = labels.ProxID.map(lambda x: x[-4:])
    labels.Name = labels.Name.map(lambda x: 'ADC' if 'ADC' in x else ('BVAL' if 'BVAL' in x else None))
    labels['coord'] = labels['Patch-Locations'].map(lambda x: x.split()).map(lambda x: list(map(int, x)))
    labels['title'] = labels.ProxID + '-' + labels.Name

    labels = labels[['title', 'coord']]
    labels.dropna(inplace=True)
    labels.set_index('title', inplace=True)
    return labels


def enlarge_label(img, coord, r=1):
    """
    Expects shape to be (2, depth, width, height)
    outputs segmentation of shape (1, depth, width, height)
    """
    seg = np.zeros_like(img[0]).astype(np.uint8)
    frame = seg[coord[2] - 1, ...]
    frame = cv2.rectangle(
        frame, (coord[0]-1, coord[1]+1), (coord[0]+1, coord[1]-1), 255, -1)
    seg[coord[2] - 1, ...] = frame
    return seg


def save_nii(files, names, dir="./saved_nii"):
    """
    Saves input list of numpy arrays in nii.gz files.
    """
    writer = sitk.ImageFileWriter()
    os.makedirs(dir, exist_ok=True)
    for file, name in zip(files, names):
        writer.SetFileName(f"{dir}/{name}.nii.gz" if not name.endswith('nii.gz') else f"{dir}/{name}")
        writer.Execute(sitk.GetImageFromArray(file))


def get_last_state(models_dir):
    r"""
    Gives out the path to the last model and its epoch number.
    Expect models to be named as (regex): Model-.*Epoch-\d*\.h5
    """
    models = glob.glob(f'{models_dir}/Model-.*.h5')

    if models:
        pat = re.compile(r'.*/Model-.*Epoch-(\d*)\.h5')
        last_model = sorted(models, reverse=True, key=lambda model: int(pat.findall(model)[0]))[0]
        return last_model, int(pat.findall(last_model)[0])
    else:
        return "/gdrive/My Drive/brainy/unet/Model-train_dice={loss:.3f}-val_dice={val_loss:.3f}-Epoch-{epoch}.h5", 1


def save_data_file(data, labels, h5_file):
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
    """Elastic deformation of images as described in [Simard2003]_.
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
