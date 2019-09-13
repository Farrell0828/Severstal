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

def _main_():
    args = argparser.parse_args()
    config_path = args.config
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    sm_models = [SMModel(config['model']) for i in range(5)]
    weights_paths = [
        os.path.join(config['train']['save_model_folder'], 
                     'val_best_fold_{}_weights.h5'.format(i))
        for i in range(5)]
    for i in range(5):
        sm_models[i].model.load_weights(weights_paths[i])

    test_generator = DataGenerator(config=config['test'],
                                   preprocessing=sm_models[0].preprocessing,
                                   n_class=sm_models[0].n_class, 
                                   split='test')
    encoded_pixels = []
    image_id_class_id = []
    for X, filenames in tqdm(list(test_generator)):
        preds = 0
        for i in range(5):
            preds += sm_models[i].model.predict_on_batch(X)
        preds /= 5
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