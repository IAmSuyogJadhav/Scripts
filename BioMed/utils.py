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


def enlarge_label(img, coords, r=1):
    """
    Expects shape to be (modalities, depth, width, height)
    outputs segmentation of shape (1, depth, width, height)
    """
    seg = np.zeros_like(img[0]).astype(np.uint8)

    for coord in coords:
        frame = seg[coord[2] - 1, ...]
        frame = cv2.rectangle(
            frame, (coord[0]-1, coord[1]+1), (coord[0]+1, coord[1]-1), 1, -1)
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


def get_last_state(models_dir, model_prefix='Model'):
    r"""
    Gives out the path to the last model and its epoch number.
    Expect models to be named as (regex): {model}-.*Epoch-\d*\.h5
    where {model} is the prefix of your model checkpoints, defaults to "Model".
    """
    models = glob.glob(os.path.join(models_dir, f"{model_prefix}-.*.h5"))

    if models:
        pat = re.compile(f'.*/{model_prefix}-.*Epoch-(\\d*)\\.h5')
        last_model = sorted(models, reverse=True, key=lambda m: int(pat.findall(m)[0]))[0]
        epoch = int(pat.findall(last_model)[0])
        print(f"Found last model at:{last_model}\nEpoch no.: {epoch}")
        return last_model, epoch
    else:
        return os.path.join(models_dir, f"{model_prefix}-train_dice={{loss:.3f}}-val_dice={{val_loss:.3f}}-Epoch-{{epoch}}.h5"), 1

                            
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
