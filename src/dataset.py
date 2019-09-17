import os 
import warnings 
import json 

import numpy as np 
import pandas as pd 
import albumentations as albu 

from glob import glob 
from keras.utils import Sequence 
from PIL import Image 
from sklearn.model_selection import StratifiedKFold 
from utils import rle2maskResize 

class DataGenerator(Sequence):
    def __init__(self, 
                 config, 
                 preprocessing=None,
                 n_class=4,
                 split='train', 
                 full_size_mask=False,
                 random_seed=606,
                 shuffle=False):

        super().__init__()
        self.preprocessing = preprocessing
        self.n_class = n_class
        self.split = split
        self.full_size_mask = full_size_mask
        self.shuffle = shuffle
        self.data_folder = config['images_folder']
        self.height = config['image_height']
        self.width = config['image_width']
        self.batch_size = config['batch_size']
        self.aug_pipline = config['aug_pipline']
        if config['aug_pipline'] != []:
            self.build_aug()
        
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
        
        if self.split == 'train':
            y = y = np.empty((len(indexes), self.height, self.width, self.n_class), dtype=np.uint8)
            for i, file_name in enumerate(self.df['ImageId'].iloc[indexes]):
                X[i, ] = Image.open(os.path.join(self.data_folder, file_name)).resize((self.width, self.height))
                for j in range(4):
                    y[i, :, :, j] = rle2maskResize(rle=self.df['EncodedPixels_' + str(j+1)].iloc[indexes[i]], 
                                                   d_height=self.height, d_width=self.width)
                if self.n_class == 5:
                    y[i, :, :, 4] = (y[i, :, :, :4].sum(axis=-1) == 0).astype(np.uint8)
                    if y.sum() != len(indexes) * self.height * self.width:
                        warnings.warn('Some pixels have not only one label is true.')
                if self.aug_pipline != []:
                    augmented = self.aug(image=X[i, ], mask=y[i, :, :, 0], 
                                         mask1=y[i, :, :, 1], mask2=y[i, :, :, 2],
                                         mask3=y[i, :, :, 3])
                    X[i, ] = augmented['image']
                    y[i, :, :, 0] = augmented['mask']
                    y[i, :, :, 1] = augmented['mask1']
                    y[i, :, :, 2] = augmented['mask2']
                    y[i, :, :, 3] = augmented['mask3']
            if self.n_class == 5:
                y[:, :, :, 4] = (y[:, :, :, :4].sum(axis=-1) == 0).astype(np.uint8)
                if y.sum() != len(indexes) * self.height * self.width:
                    warnings.warn('Some pixels have not only one label is true.')

        if self.split == 'val':
            if self.full_size_mask:
                y = np.empty((len(indexes), 256, 1600, self.n_class), dtype=np.uint8)
            else:
                y = np.empty((len(indexes), self.height, self.width, self.n_class), dtype=np.uint8)
            for i, file_name in enumerate(self.df['ImageId'].iloc[indexes]):
                X[i, ] = Image.open(os.path.join(self.data_folder, file_name)).resize((self.width, self.height))
                for j in range(4):
                    if self.full_size_mask:
                        y[i, :, :, j] = rle2maskResize(rle=self.df['EncodedPixels_' + str(j+1)].iloc[indexes[i]], 
                                                       d_height=256, d_width=1600)
                    else:
                        y[i, :, :, j] = rle2maskResize(rle=self.df['EncodedPixels_' + str(j+1)].iloc[indexes[i]], 
                                                       d_height=self.height, d_width=self.width)
            if self.n_class == 5:
                y[:, :, :, 4] = (y[:, :, :, :4].sum(axis=-1) == 0).astype(np.uint8)
                if ((not self.full_size_mask and y.sum() != len(indexes) * self.height * self.width)
                    or (self.full_size_mask and y.sum() != len(indexes) * 256 * 1600)):
                    warnings.warn('Some pixels have not only one label is true.')
        
        else:
            filenames = []
            for i, file_name in enumerate([self.image_file_paths[index] for index in indexes]):
                X[i, ] = Image.open(file_name).resize((self.width, self.height))
                filenames.append(file_name.split('/')[-1])

        if self.preprocessing is not None: X = self.preprocessing(X)
        
        if self.split != 'test': 
            return X, y
        else:
            return X, filenames
    
    def build_aug(self):
        additional_targets = {
            'mask1': 'mask',
            'mask2': 'mask',
            'mask3': 'mask'
        }
        aug = []
        for aug_type in self.aug_pipline:
            if aug_type == 'Flip':
                aug += [albu.HorizontalFlip(), albu.VerticalFlip()]
            elif aug_type == 'Non-Spatial':
                aug.append(albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                ], p=0.3))
            elif aug_type == "Non-Rigid":
                aug.append(albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3))
            elif aug_type == 'Rotate':
                aug.append(albu.ShiftScaleRotate(interpolation=0))
            else:
                raise ValueError('aug type {} not support.'.format(aug_type))
        self.aug = albu.Compose(aug, p=1, additional_targets=additional_targets)


if __name__ == '__main__':
    config_path = './configs/config.json'
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    generator = DataGenerator(config['test'], None, 4, 'test')
    i = np.random.randint(0, len(generator))
    X = generator[i]
    print(X.shape, X.dtype)
    # print(y.shape, y.dtype)
    img = Image.fromarray(X[0].astype(np.uint8))
    # masks = [Image.fromarray(y[0, :, :, i]*255) for i in range(5)]
    img.show(title='image')
    # for i in range(5): 
    #     masks[i].show(title='mask[{}]'.format(i))

