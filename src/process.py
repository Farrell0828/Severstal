import cv2 
import numpy as np 
from keras.utils.np_utils import to_categorical

def postprocess(y_pred, threshold=0.5, height_scale=1.0, 
                width_scale=1.0, min_counts=[0, 0, 0, 0]):
    n_class = y_pred.shape[-1]
    if n_class == 4:
        y_pred_bi = (y_pred >= threshold).astype(int)
    elif n_class == 5:
        y_pred_bi = to_categorical(y_pred.argmax(axis=-1), num_classes=5)[:, :, :, :4]
    else:
        raise ValueError('The number of channels: {} not valid, '\
            'only support 4 or 5.'.format(n_class))
    y_pred_counts = y_pred_bi.sum(1, keepdims=True).sum(2, keepdims=True)
    mask = (y_pred_counts >= (np.array(min_counts) / (height_scale * width_scale)))
    return y_pred_bi * mask

def post_process(mask, min_size):
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predict = np.zeros((256, 1600), np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = 1
    return predict