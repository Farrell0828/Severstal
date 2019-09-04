import argparse 
import json 
import os 
import cv2 
import pandas as pd 
from tqdm import tqdm 
from model import SMModel 
from process import postprocess 
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

def _main_():
    args = argparser.parse_args()
    config_path = args.config
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    weights_path = args.weights_path
    sm_model = SMModel(config['model'])
    sm_model.model.summary()
    preds, image_file_names = sm_model.predict(config['test'], weights_path)
    print(len(preds), len(image_file_names))
    encoded_pixels = []
    image_id_class_id = []
    preds = postprocess(preds, 0.5, True, False)
    for i in tqdm(range(len(preds))):
        for j in range(4):
            encoded_pixels.append(run_length_encode(preds[i, :, :, j]))
            image_id_class_id.append(image_file_names[i] + '_{}'.format(j + 1))
    df = pd.DataFrame(data=encoded_pixels, index=image_id_class_id, columns=['EncodedPixels'])
    df.index.name = 'ImageId_ClassId'
    df.to_csv('submission.csv')

if __name__ == '__main__':
    _main_()