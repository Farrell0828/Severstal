import cv2 
import numpy as np 
from keras.utils.np_utils import to_categorical

def postprocess(y_pred, config, return_full_size=False):
    n_class = y_pred.shape[-1]
    if n_class == 4:
        if return_full_size: 
            y_pred = cv2.resize(y_pred, (1600, 256), interpolation=cv2.INTER_LINEAR)
        if config['triplet']:
            y_pred_bi = (y_pred >= config['threshold_high']).astype(int)
        else:
            y_pred_bi = (y_pred >= config['threshold']).astype(int)
    elif n_class == 5:
        y_pred_bi = to_categorical(y_pred.argmax(axis=-1), num_classes=5)[:, :, :, :4]
    else:
        raise ValueError('The number of channels: {} not valid, '\
            'only support 4 or 5.'.format(n_class))
    if return_full_size:
        processed_pred = np.zeros((y_pred.shape[0], 256, 1600, 4), dtype=np.uint8)
    else:
        processed_pred = np.zeros(y_pred_bi.shape, dtype=np.uint8)
    for i in range(len(y_pred_bi)):
        for j in range(4):
            filted = postprocess_sigle_channel(y_pred_bi[i, :, :, j], 
                                               return_full_size, 
                                               config['filter_small_region'], 
                                               config['min_size'][j])
            if config['triplet'] and filted.sum() != 0:
                processed_pred[i, :, :, j] = (y_pred >= config['threshold_low']).astype(int)
            else:
                processed_pred[i, :, :, j] = filted
    return processed_pred

def postprocess_sigle_channel(mask, return_full_size, filter_small_region, min_size):
    if return_full_size and mask.shape != (256, 1600):
        mask = cv2.resize(mask, (1600, 256), interpolation=cv2.INTER_NEAREST)
    min_size = min_size * (mask.shape[0] / 256)**2
    if filter_small_region:
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        mask = np.zeros(mask.shape, np.uint8)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                mask[p] = 1
    return mask