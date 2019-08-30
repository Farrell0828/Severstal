import keras.backend as K 
import tensorflow as tf 

def dice_coef(y_true, y_pred, 
              threshold=0.5, 
              smooth=1.0):
    y_true = K.greater_equal(y_true, threshold)[:, :, :, :4]
    y_pred = K.greater_equal(y_pred, threshold)[:, :, :, :4]
    inter = K.sum(y_true*y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    dice_coef = (2.0*inter + smooth) / (union + smooth)
    y_true_is_empty = K.equal(K.sum(y_true, axis=[1, 2]), 0)
    y_pred_is_not_empty = K.greater(K.sum(y_pred, axis=[1, 2]), 0)
    mask = tf.logical_not(tf.logical_and(y_true_is_empty, y_pred_is_not_empty))
    mask = K.cast(mask, 'float32')
    return K.mean(dice_coef * mask, axis=-1) 
