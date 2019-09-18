import argparse 
import json 
import os 
import cv2 
import pandas as pd 
from tqdm import tqdm 
from model import SMModel 
from process import postprocess 
from dataset import DataGenerator 
from utils import run_length_encode 

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-c',
    '--config',
    help='Config file.'
)
argparser.add_argument(
    '-w',
    '--weights_path',
    help='Weights file path.'
)

def flip(imgs, flip_type):
    if flip_type is None:
        return imgs
    elif flip_type == 'ud':
        return imgs[:, ::-1, ...]
    elif flip_type == 'lr':
        return imgs[:, :, ::-1, ...]
    elif flip_type == 'udlr':
        return imgs[:, ::-1, ::-1, ...]
    else:
        raise ValueError('flip type {} not support.'.format(flip_type))

def _main_():
    args = argparser.parse_args()
    config_path = args.config
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    weights_path = args.weights_path
    sm_model = SMModel(config['model'])
    sm_model.model.summary()
    sm_model.model.load_weights(weights_path)
    test_generator = DataGenerator(config=config['test'],
                                   preprocessing=sm_model.preprocessing,
                                   n_class=sm_model.n_class, 
                                   split='test')
    encoded_pixels = []
    image_id_class_id = []
    for X, filenames in tqdm(list(test_generator)):
        preds = 0
        for flip_type in [None, 'ud', 'lr', 'udlr']:
            X_temp = flip(X.copy(), flip_type)
            pred_temp = sm_model.model.predict_on_batch(X_temp)
            preds += flip(pred_temp, flip_type)
        preds /= 4
        preds = postprocess(preds, config['postprocess'], True)
        for i in range(len(preds)):
            for j in range(4):
                encoded_pixels.append(run_length_encode(preds[i, :, :, j]))
                image_id_class_id.append(filenames[i] + '_{}'.format(j + 1))
    df = pd.DataFrame(data=encoded_pixels, index=image_id_class_id, columns=['EncodedPixels'])
    df.index.name = 'ImageId_ClassId'
    df.to_csv('submission.csv')

if __name__ == '__main__':
    _main_()