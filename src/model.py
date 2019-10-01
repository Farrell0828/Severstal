
import os 
import json 
import keras 
import segmentation_models as sm 
import tensorflow as tf 
import numpy as np 
from keras import backend as K 
from keras.models import Model 
from keras.losses import binary_crossentropy, categorical_crossentropy 
from keras.optimizers import Adam, SGD 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 
from tensorflow.python.client import device_lib 
from dataset import DataGenerator 
from metrics import dice_coef_for_sigmoid, dice_coef_for_softmax, acc_for_cls 
from metrics import acc_for_cls0, acc_for_cls1, acc_for_cls2, acc_for_cls3 
from callbacks import DiceCoefCallback 
from losses import categorical_focal_loss, sgm_cls_loss, sgm_multi_cls_loss 


class SMModel(object):

    def __init__(self, config):
        self.type = config['type']
        self.backbone = config['backbone']
        self.n_class = config['n_class']
        self.activate = config['activation']
        self.encoder_weights = config.get('encoder_weights', None)
        self.preprocessing = sm.backbones.get_preprocessing(self.backbone)

        local_device_protos = device_lib.list_local_devices()
        n_available_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        if n_available_gpus < 2:
            self.model = self._build()
        else:
            with tf.device('/cpu:0'):
                model = self._build()
            self.model = keras.utils.multi_gpu_model(model, gpus=n_available_gpus)

    def _build(self):
        if self.type == 'Unet':
            model = sm.Unet(backbone_name=self.backbone, 
                            classes=self.n_class,
                            activation=self.activate,
                            encoder_weights=self.encoder_weights)

        elif self.type == 'Linknet':
            model = sm.Linknet(backbone_name=self.backbone, 
                               classes=self.n_class, 
                               activation=self.activate, 
                               encoder_weights=self.encoder_weights)

        elif self.type == 'FPN':
            model = sm.FPN(backbone_name=self.backbone, 
                           classes=self.n_class, 
                           activation=self.activate, 
                           encoder_weights=self.encoder_weights)
        
        else:
            raise ValueError('Model type {} not support now.'.format(self.type))

        return model

    def train(self, config):
        train_generator = DataGenerator(config=config,
                                        preprocessing=self.preprocessing,
                                        n_class=self.n_class, 
                                        split='train',
                                        shuffle=True)
        val_generator = DataGenerator(config=config, 
                                      preprocessing=self.preprocessing,
                                      n_class=self.n_class,
                                      split='val')
        if self.activate == 'sigmoid':
            loss = sm.losses.jaccard_loss
            dice_coef = dice_coef_for_sigmoid
            monitor = 'val_dice_coef_for_sigmoid'
        elif self.activate == 'softmax':
            bias = self.model.layers[-2].weights[1]
            K.set_value(bias, np.full(bias.shape, -1.99))
            loss = sgm_multi_cls_loss(alpha=config['alpha'], gamma=config['gamma'])
            dice_coef = dice_coef_for_softmax
            monitor = 'val_dice_coef_for_softmax'

        self.model.compile(optimizer=Adam(lr=config['init_lr']),
                           loss=loss,
                           metrics=[dice_coef, acc_for_cls0, acc_for_cls1, acc_for_cls2, acc_for_cls3])

        log_folder_path = './logs/fold{}'.format(config['fold'])
        if not os.path.exists(log_folder_path): os.mkdir(log_folder_path)
        tensorboard = TensorBoard(log_dir=log_folder_path, update_freq='batch')
        reduce_lr = ReduceLROnPlateau(patience=5, 
                                      monitor=monitor,
                                      mode='max',
                                      verbose=1, 
                                      min_delta=1e-5)
        early_stopping = EarlyStopping(patience=8, 
                                       monitor=monitor,
                                       mode='max',
                                       verbose=1, 
                                       min_delta=1e-5)
        if not os.path.exists(config['save_model_folder']): os.mkdir(config['save_model_folder'])
        save_weights_path = os.path.join(config['save_model_folder'], 
                                         'val_best_fold_{}_weights.h5'.format(config['fold']))
        checkpoint = ModelCheckpoint(save_weights_path, 
                                     monitor=monitor,
                                     mode='max',
                                     verbose=1,
                                     save_weights_only=True, 
                                     save_best_only=True)

        callbacks = [tensorboard, early_stopping, reduce_lr, checkpoint]
        
        self.model.fit_generator(generator=train_generator, 
                                 epochs=config['epochs'],
                                 validation_data=val_generator, 
                                 callbacks=callbacks)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    config_path = './configs/config.json'
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    sm = SMModel(config['model'])
    sm.model.summary()