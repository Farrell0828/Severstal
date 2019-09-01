import keras.backend as K 
import tensorflow as tf 

def dice_coef(y_true, y_pred, threshold=0.5):
    y_true = K.greater_equal(y_true, threshold)
    y_pred = K.greater_equal(y_pred, threshold)
    inter = K.sum(y_true*y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    dice_coef = (2.0*inter + K.epsilon()) / (union + K.epsilon())
    return K.mean(dice_coef, axis=-1)
