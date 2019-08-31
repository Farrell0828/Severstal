
import os 
import segmentation_models as sm 
import tensorflow as tf 
from keras.models import Model 
from keras.losses import binary_crossentropy, categorical_crossentropy 
from keras.optimizers import Adam, SGD 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
from tensorflow.python.client import device_lib
from dataset import DataGenerator 
from metrics import dice_coef 
from callbacks import DiceCoefCallback 

class SMModel(object):

    def __init__(self, config):
        self.type = config['type']
        self.backbone = config['backbone']
        self.n_class = config['n_class']
        self.activate = config['activate']
        self.encoder_weights = config.get('encoder_weights', None)

        local_device_protos = device_lib.list_local_devices()
        n_available_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        if n_available_gpus < 2:
            self.model = self._build()
        else:
            with tf.device('/cpu:0'):
                model = self._build()
            self.model = tf.keras.utils.multi_gpu_model(model, gpus=n_available_gpus)

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
        preprocessing = sm.backbones.get_preprocessing(self.backbone)
        train_generator = DataGenerator(config=config,
                                        preprocessing=preprocessing,
                                        n_class=self.n_class, 
                                        split='train')
        val_generator = DataGenerator(config=config, 
                                      preprocessing=preprocessing,
                                      n_class=self.n_class,
                                      split='val')

        self.model.compile(optimizer=Adam(lr=config['init_lr']),
                           loss=sm.losses.dice_loss,
                           metrics=[sm.metrics.iou_score, dice_coef])

        tensorboard = TensorBoard(log_dir='./logs', update_freq='batch')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15)
        save_weights_path = os.path.join(config['save_weights_folder'], 
                                         'val_best_fold_{}.h5'.format(config['fold']))
        checkpoint = ModelCheckpoint(save_weights_path, 
                                     monitor='val_loss',
                                     save_weights_only=True, 
                                     save_best_only=True)
        dice_coef_callback = DiceCoefCallback(val_generator)

        self.model.fit_generator(generator=train_generator, 
                                 epochs=config['epochs'],
                                 validation_data=val_generator, 
                                 callbacks=[tensorboard, early_stopping, 
                                            reduce_lr, checkpoint, dice_coef_callback])

        self.model.load_weights(save_weights_path)
        self.model.evaluate_generator(val_generator)
