from keras import backend as K 
import tensorflow as tf 

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

        y_true_cls = K.cast(K.not_equal(K.argmax(y_true, axis=-1), 4), 'float')
        y_true_cls = K.cast(K.sum(y_true_cls, axis=[1, 2]) > 0, 'float')
        y_pred_cls = K.cast(K.not_equal(K.argmax(y_pred, axis=-1), 4), 'float')
        y_pred_cls = K.cast(K.sum(y_pred_cls, axis=[1, 2]) > 0, 'float')
        pt_1 = tf.where(tf.equal(y_true_cls, 1), y_pred_cls, tf.ones_like(y_pred_cls))
        pt_0 = tf.where(tf.equal(y_true_cls, 0), y_pred_cls, tf.zeros_like(y_pred_cls))
        pt_1 = K.clip(pt_1, K.epsilon(), 1. - K.epsilon())
        pt_0 = K.clip(pt_0, K.epsilon(), 1. - K.epsilon())
        cls_loss = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1) \
                   -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0)

        return sgm_loss + cls_loss

    return sgm_cls_focal_loss
    
def sgm_multi_cls_loss(gamma=2.0, alpha=0.25):

    def sgm_cls_focal_loss(y_true, y_pred):

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        sgm_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        sgm_loss = K.sum(sgm_loss, axis=-1)
        sgm_loss = K.mean(sgm_loss, axis=[1, 2])

        y_true_cls0 = K.cast(K.equal(K.argmax(y_true, axis=-1), 0), 'float')
        y_true_cls0 = K.cast(K.sum(y_true_cls0, axis=[1, 2]) > 0, 'float')
        y_pred_cls0 = K.cast(K.equal(K.argmax(y_pred, axis=-1), 0), 'float')
        y_pred_cls0 = K.cast(K.sum(y_pred_cls0, axis=[1, 2]) > 0, 'float')
        pt0_1 = tf.where(tf.equal(y_true_cls0, 1), y_pred_cls0, tf.ones_like(y_pred_cls0))
        pt0_0 = tf.where(tf.equal(y_true_cls0, 0), y_pred_cls0, tf.zeros_like(y_pred_cls0))
        pt0_1 = K.clip(pt0_1, K.epsilon(), 1. - K.epsilon())
        pt0_0 = K.clip(pt0_0, K.epsilon(), 1. - K.epsilon())
        cls0_loss = -alpha * K.pow(1. - pt0_1, gamma) * K.log(pt0_1) \
                    -(1 - alpha) * K.pow(pt0_0, gamma) * K.log(1. - pt0_0)

        y_true_cls1 = K.cast(K.equal(K.argmax(y_true, axis=-1), 1), 'float')
        y_true_cls1 = K.cast(K.sum(y_true_cls1, axis=[1, 2]) > 0, 'float')
        y_pred_cls1 = K.cast(K.equal(K.argmax(y_pred, axis=-1), 1), 'float')
        y_pred_cls1 = K.cast(K.sum(y_pred_cls1, axis=[1, 2]) > 0, 'float')
        pt1_1 = tf.where(tf.equal(y_true_cls1, 1), y_pred_cls1, tf.ones_like(y_pred_cls1))
        pt1_0 = tf.where(tf.equal(y_true_cls1, 0), y_pred_cls1, tf.zeros_like(y_pred_cls1))
        pt1_1 = K.clip(pt1_1, K.epsilon(), 1. - K.epsilon())
        pt1_0 = K.clip(pt1_0, K.epsilon(), 1. - K.epsilon())
        cls1_loss = -alpha * K.pow(1. - pt1_1, gamma) * K.log(pt1_1) \
                    -(1 - alpha) * K.pow(pt1_0, gamma) * K.log(1. - pt1_0)

        y_true_cls2 = K.cast(K.equal(K.argmax(y_true, axis=-1), 2), 'float')
        y_true_cls2 = K.cast(K.sum(y_true_cls2, axis=[1, 2]) > 0, 'float')
        y_pred_cls2 = K.cast(K.equal(K.argmax(y_pred, axis=-1), 2), 'float')
        y_pred_cls2 = K.cast(K.sum(y_pred_cls2, axis=[1, 2]) > 0, 'float')
        pt2_1 = tf.where(tf.equal(y_true_cls2, 1), y_pred_cls2, tf.ones_like(y_pred_cls2))
        pt2_0 = tf.where(tf.equal(y_true_cls2, 0), y_pred_cls2, tf.zeros_like(y_pred_cls2))
        pt2_1 = K.clip(pt2_1, K.epsilon(), 1. - K.epsilon())
        pt2_0 = K.clip(pt2_0, K.epsilon(), 1. - K.epsilon())
        cls2_loss = -alpha * K.pow(1. - pt2_1, gamma) * K.log(pt2_1) \
                    -(1 - alpha) * K.pow(pt2_0, gamma) * K.log(1. - pt2_0)

        y_true_cls3 = K.cast(K.equal(K.argmax(y_true, axis=-1), 3), 'float')
        y_true_cls3 = K.cast(K.sum(y_true_cls3, axis=[1, 2]) > 0, 'float')
        y_pred_cls3 = K.cast(K.equal(K.argmax(y_pred, axis=-1), 3), 'float')
        y_pred_cls3 = K.cast(K.sum(y_pred_cls3, axis=[1, 2]) > 0, 'float')
        pt3_1 = tf.where(tf.equal(y_true_cls3, 1), y_pred_cls3, tf.ones_like(y_pred_cls3))
        pt3_0 = tf.where(tf.equal(y_true_cls3, 0), y_pred_cls3, tf.zeros_like(y_pred_cls3))
        pt3_1 = K.clip(pt3_1, K.epsilon(), 1. - K.epsilon())
        pt3_0 = K.clip(pt3_0, K.epsilon(), 1. - K.epsilon())
        cls3_loss = -alpha * K.pow(1. - pt3_1, gamma) * K.log(pt3_1) \
                    -(1 - alpha) * K.pow(pt3_0, gamma) * K.log(1. - pt2_0)

        return 4*sgm_loss + cls0_loss + cls1_loss + cls2_loss + cls3_loss

    return sgm_cls_focal_loss