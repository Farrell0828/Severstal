!pip install segmentation_models

my_seed = 606
from numpy.random import seed
seed(my_seed)
import tensorflow as tf 
tf.set_random_seed(my_seed)
import os 
os.environ['PYTHONHASHSEED'] = str(my_seed)
import torch
torch.backends.cudnn.deterministic = True
torch.manual_seed(my_seed)
torch.cuda.manual_seed_all(my_seed)

config = {
    "model" : {
        "type"                     : "Unet",
        "backbone"                 : "resnet34",
        "encoder_weights"          : "imagenet",
        "n_class"                  : 5,
        "activation"               : "softmax"
    },

    "train": {
        "images_folder"            : "../input/severstal-steel-defect-detection/train_images", 
        "csv_file_path"            : "./train_new.csv", 
        "image_height"             : 128, 
        "image_width"              : 800, 
        "aug_pipline"              : [],
        "fold"                     : 2,
        "init_lr"                  : 1e-3,
        "batch_size"               : 32,
        "epochs"                   : 100,
        "save_model_folder"        : "./"
    }, 

    "test": {
        "images_folder"            : "../input/severstal-steel-defect-detection/test_images",
        "sample_submission_path"   : "../input/severstal-steel-defect-detection/sample_submission.csv",
        "image_height"             : 128,
        "image_width"              : 800,
        "aug_pipline"              : [],
        "batch_size"               : 64
    }
}

import os 
import warnings 
import json 
import cv2 
import keras 
import pandas as pd 
import numpy as np 
import segmentation_models as sm 
import tensorflow as tf 
from tensorflow.python.client import device_lib 
from PIL import Image 
from matplotlib import pyplot as plt 
from glob import glob 
from keras import backend as K 
from keras.models import Model 
from keras.losses import binary_crossentropy, categorical_crossentropy 
from keras.optimizers import Adam, SGD 
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger 
from keras.utils import Sequence 
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import StratifiedKFold 


df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
df['ImageId'] = df['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
new_df = pd.DataFrame({'ImageId':df['ImageId'][::4]})
new_df['EncodedPixels_1'] = df['EncodedPixels'][::4].values
new_df['EncodedPixels_2'] = df['EncodedPixels'][1::4].values
new_df['EncodedPixels_3'] = df['EncodedPixels'][2::4].values
new_df['EncodedPixels_4'] = df['EncodedPixels'][3::4].values
new_df.reset_index(inplace=True,drop=True)
new_df.fillna('',inplace=True); 
new_df['MaskCount'] = np.sum(new_df.iloc[:,1:]!='',axis=1).values
new_df.to_csv('train_new.csv', index=False)

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
    m = mask.T.flatten()
    if m.sum() == 0:
        rle=''
    else:
        start  = np.where(m[1: ] > m[:-1])[0]+2
        end    = np.where(m[:-1] > m[1: ])[0]+2
        length = end-start
        rle = [start[0],length[0]]
        for i in range(1,len(length)):
            rle.extend([start[i],length[i]])
        rle = ' '.join([str(r) for r in rle])
    return rle

class DataGenerator(Sequence):
    def __init__(self, 
                 config, 
                 preprocessing=None,
                 n_class=4,
                 split='train', 
                 random_seed=606,
                 shuffle=False):

        super().__init__()
        self.preprocessing = preprocessing
        self.split = split
        self.n_class = n_class
        self.shuffle = shuffle
        self.data_folder = config['images_folder']
        self.height = config['image_height']
        self.width = config['image_width']
        self.batch_size = config['batch_size']
        self.aug_pipline = config['aug_pipline']
        
        if split != 'test':
            self.fold = config['fold']
            df = pd.read_csv(config['csv_file_path'])
            skf = StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True)
            train_idx, val_idx = list(skf.split(df, df['MaskCount']))[self.fold]
            if split == 'train':
                self.df = df.iloc[train_idx]
            elif split == 'val':
                self.df = df.iloc[val_idx]
            else:
                raise ValueError("Split '{}' not support now.".format(split))
            self.n_samples = len(self.df)
        else:
            self.image_file_paths = glob(self.data_folder + '/*')
            self.n_samples = len(self.image_file_paths)

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index): 
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        X = np.empty((len(indexes), self.height, self.width, 3), dtype=np.float32)
        
        if self.split != 'test':
            y = np.empty((len(indexes), self.height, self.width, self.n_class), dtype=np.uint8)
            for i, file_name in enumerate(self.df['ImageId'].iloc[indexes]):
                X[i, ] = Image.open(os.path.join(self.data_folder, file_name)).resize((self.width, self.height))
                for j in range(4):
                    y[i, :, :, j] = rle2maskResize(rle=self.df['EncodedPixels_' + str(j+1)].iloc[indexes[i]], 
                                                   d_height=self.height, d_width=self.width)
            if self.n_class == 5:
                y[:, :, :, 4] = (y[:, :, :, :4].sum(axis=-1) == 0).astype(np.uint8)
                if y.sum() != len(indexes) * self.height * self.width:
                    warnings.warn('Some pixels have not only one label is true.')
        else:
            filenames = []
            for i, file_name in enumerate([self.image_file_paths[index] for index in indexes]):
                X[i, ] = Image.open(file_name).resize((self.width, self.height))
                filenames.append(file_name.split('/')[-1])

        if self.aug_pipline != []: X = self.aug(X)
        if self.preprocessing is not None: X = self.preprocessing(X)
        
        if self.split != 'test': 
            return X, y
        else:
            return X, filenames

    def aug(self, X):
        return X

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

def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(loss, axis=1)
    return categorical_focal_loss_fixed

def postprocess(y_pred, threshold=0.5, 
                return_full_size=False, 
                filter_small_region=False):
    n_class = y_pred.shape[-1]
    if n_class == 4:
        y_pred_bi = (y_pred >= threshold).astype(int)
    elif n_class == 5:
        y_pred_bi = to_categorical(y_pred.argmax(axis=-1), num_classes=5)[:, :, :, :4]
    else:
        raise ValueError('The number of channels: {} not valid, '\
            'only support 4 or 5.'.format(n_class))
    processed_pred = np.empty(y_pred_bi.shape, dtype=np.uint8)
    for i in range(len(y_pred_bi)):
        for j in range(4):
            processed_pred[i, :, :, j] = postprocess_sigle_channel(y_pred_bi[i, :, :, j], 
                                                                   return_full_size, 
                                                                   filter_small_region)
    return processed_pred

def postprocess_sigle_channel(mask, return_full_size, filter_small_region, min_size=2048):
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

class DiceCoefCallback(keras.callbacks.Callback):
    def __init__(self, generator):
        super(DiceCoefCallback, self).__init__()
        self.generator = generator
        self.eps = 1e-7

    def on_epoch_end(self, epoch, logs={}):
        dice_coef = 0
        for X, y_true in list(self.generator):
            y_pred = self.model.predict(X)
            y_pred = postprocess(y_pred, 0.5, False, True)
            y_true = y_true[:, :, :, :4]
            inter = (y_true * y_pred).sum(1).sum(1)
            union = y_true.sum(1).sum(1) + y_pred.sum(1).sum(1)
            dice_coef_batch = (inter + self.eps) / (union + self.eps)
            dice_coef += dice_coef_batch.sum()
        dice_coef /= (self.generator.n_samples * 4)
        logs.update({'dice_coef_score': dice_coef})
        print('Epoch {} Validation Dice Coefficent Score after Postprocess: {}.\n'.format(epoch+1, dice_coef))

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
        elif self.activate == 'softmax':
            loss = categorical_focal_loss(alpha=.25, gamma=2)
            dice_coef = dice_coef_for_softmax

        self.model.compile(optimizer=Adam(lr=config['init_lr']),
                               loss=loss,
                               metrics=[sm.metrics.iou_score, dice_coef])

        # tensorboard = TensorBoard(log_dir='./logs', update_freq='batch')
        reduce_lr = ReduceLROnPlateau(patience=5, verbose=1, min_delta=1e-6)
        early_stopping = EarlyStopping(patience=8, verbose=1, min_delta=1e-6)
        dice_coef_callback = DiceCoefCallback(val_generator)

        save_weights_path = os.path.join(config['save_model_folder'], 
                                         'val_best_fold_{}_weights.h5'.format(config['fold']))
        save_model_path = os.path.join(config['save_model_folder'], 
                                         'val_best_fold_{}_model.h5'.format(config['fold']))
        csv_logger = CSVLogger('training_log.csv')
        checkpoint = ModelCheckpoint(save_weights_path, 
                                     monitor='val_loss',
                                     verbose=1,
                                     save_weights_only=True, 
                                     save_best_only=True)

        callbacks = [early_stopping, reduce_lr, csv_logger, checkpoint, dice_coef_callback]
        
        self.model.fit_generator(generator=train_generator, 
                                 epochs=config['epochs'],
                                 validation_data=val_generator, 
                                 callbacks=callbacks)

        self.model.load_weights(save_weights_path)
        self.model.save(save_model_path)
        self.model.evaluate_generator(val_generator)

sm_model = SMModel(config['model'])
sm_model.train(config['train'])