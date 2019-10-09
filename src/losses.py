from keras import backend as K 
import tensorflow as tf 
from segmentation_models.losses import bce_dice_loss, bce_jaccard_loss  

def categorical_focal_loss(gamma=2.0, alpha=0.25):

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)

    return categorical_focal_loss_fixed

def sgm_cls_loss(gamma=2.0, alpha=0.25):

    def sgm_cls_focal_loss(y_true, y_pred):

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        sgm_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        sgm_loss = K.sum(sgm_loss, axis=-1)
        sgm_loss = K.mean(sgm_loss, axis=[1, 2])

        y_true_cls = K.cast(K.sum(1.0 - y_true[:, :, :, -1], axis=[1, 2]) > 0, 'float')
        y_pred_cls = K.max(1.0 - y_pred[:, :, :, -1], axis=[1, 2])
        
        pt_1 = tf.where(tf.equal(y_true_cls, 1), y_pred_cls, tf.ones_like(y_pred_cls))
        pt_0 = tf.where(tf.equal(y_true_cls, 0), y_pred_cls, tf.zeros_like(y_pred_cls))
        pt_1 = K.clip(pt_1, K.epsilon(), 1. - K.epsilon())
        pt_0 = K.clip(pt_0, K.epsilon(), 1. - K.epsilon())
        cls_loss = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) \
                   -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)

        return sgm_loss + cls_loss

    return sgm_cls_focal_loss
    
def sgm_multi_cls_loss(gamma=2.0, alpha=0.25):

    def cls_loss_sigle_channel(y_true, y_pred, c):
        y_true_cls = K.cast(K.sum(y_true[:, :, :, c], axis=[1, 2]) > 0, 'float')
        y_pred_cls = K.max(y_pred[:, :, :, c], axis=[1, 2])
        pt_1 = tf.where(tf.equal(y_true_cls, 1), y_pred_cls, tf.ones_like(y_pred_cls))
        pt_0 = tf.where(tf.equal(y_true_cls, 0), y_pred_cls, tf.zeros_like(y_pred_cls))
        pt_1 = K.clip(pt_1, K.epsilon(), 1. - K.epsilon())
        pt_0 = K.clip(pt_0, K.epsilon(), 1. - K.epsilon())
        return -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) \
               -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)

    def sgm_multi_cls_focal_loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        sgm_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        sgm_loss = K.sum(sgm_loss, axis=-1)
        sgm_loss = K.mean(sgm_loss, axis=[1, 2])

        cls0_loss = cls_loss_sigle_channel(y_true, y_pred, 0)
        cls1_loss = cls_loss_sigle_channel(y_true, y_pred, 1)
        cls2_loss = cls_loss_sigle_channel(y_true, y_pred, 2)
        cls3_loss = cls_loss_sigle_channel(y_true, y_pred, 3)

        return 4*sgm_loss + cls0_loss + cls1_loss + cls2_loss + cls3_loss

    return sgm_multi_cls_focal_loss
