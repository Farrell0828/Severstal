import os 
import warnings 
import json 

import numpy as np 
import pandas as pd 

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
            self.image_file_names = glob(self.data_folder + '/*')
            self.n_samples = len(self.image_file_names)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.n_samples / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index): 
        X = np.empty((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        if self.split != 'test':
            y = np.empty((self.batch_size, self.height, self.width, self.n_class), dtype=np.int8)
            for i, file_name in enumerate(self.df['ImageId'].iloc[indexes]):
                X[i, ] = Image.open(os.path.join(self.data_folder, file_name)).resize((self.width, self.height))
                for j in range(4):
                    y[i, :, :, j] = rle2maskResize(rle=self.df['EncodedPixels_' + str(j+1)].iloc[indexes[i]], 
                                                   d_height=self.height, d_width=self.width)
            if self.n_class == 5:
                y[:, :, :, 4] = (y[:, :, :, :4].sum(axis=-1) == 0).astype(np.uint8)
                if y.sum() != self.batch_size * self.height * self.width:
                    warnings.warn('Some pixels have not only one label is true.')
        else:
            for i, file_name in enumerate([self.image_file_names[index] for index in indexes]):
                X[i, ] = Image.open(file_name).resize((self.width, self.height))

        if self.aug_pipline != []: X = self.aug(X)
        if self.preprocessing is not None: X = self.preprocessing(X)
        
        if self.split != 'test': 
            return X, y
        else:
            return X

    def aug(self, X):
        return X

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

