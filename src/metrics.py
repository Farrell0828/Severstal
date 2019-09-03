import keras.backend as K 
import tensorflow as tf 

def dice_coef_for_sigmoid(y_true, y_pred, threshold=0.5):
    y_true_bi = K.cast(K.greater_equal(y_true, threshold), 'float32')
    y_pred_bi = K.cast(K.greater_equal(y_pred, threshold), 'float32')
    inter = K.sum(y_true_bi*y_pred_bi, axis=[1, 2])
    union = K.sum(y_true_bi, axis=[1, 2]) + K.sum(y_pred_bi, axis=[1, 2])
    dice_coef = (2.0*inter + K.epsilon()) / (union + K.epsilon())
    return K.mean(dice_coef, axis=-1)

def dice_coef_for_softmax(y_true, y_pred, threshold=0.5):
    y_true_bi = K.cast(K.greater_equal(y_true, threshold), 'float32')[:, :, :, :4]
    y_pred_bi = K.cast(tf.one_hot(K.argmax(y_pred), tf.shape(y_pred)[-1]), 'float32')[:, :, :, :4]
    inter = K.sum(y_true_bi*y_pred_bi, axis=[1, 2])
    union = K.sum(y_true_bi, axis=[1, 2]) + K.sum(y_pred_bi, axis=[1, 2])
    dice_coef = (2.0*inter + K.epsilon()) / (union + K.epsilon())
    return K.mean(dice_coef, axis=-1)
