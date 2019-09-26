import numpy as np 
import pandas as pd 
from PIL import Image 
from matplotlib import pyplot as plt 


# https://www.kaggle.com/titericz/building-and-visualizing-masks
def rle2maskResize(rle, d_height, d_width):
    # CONVERT RLE TO MASK 
    if (pd.isnull(rle))|(rle==''): 
        return np.zeros((d_height, d_width), dtype=np.uint8)
    height= 256
    width = 1600
    mask= np.zeros(width*height, dtype=np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
    mask = mask.reshape(width, height).T
    img = Image.fromarray(mask, mode='L').resize((d_width, d_height), resample=0)
    return np.array(img)

def run_length_encode(mask):
    return mask2rle(mask)

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)

def dice_coef_score(true_masks, pred_masks):
    eps = 1e-7
    inter = (true_masks * pred_masks).sum(1).sum(1)
    union = true_masks.sum(1).sum(1) + pred_masks.sum(1).sum(1)
    return ((2*inter + eps) / (union + eps)).mean()

def flip(imgs, flip_type):
    if flip_type == 'ud':
        return imgs[:, ::-1, ...]
    elif flip_type == 'lr':
        return imgs[:, :, ::-1, ...]
    elif flip_type == 'udlr':
        return imgs[:, ::-1, ::-1, ...]
    else:
        raise ValueError('flip type {} not support.'.format(flip_type))
