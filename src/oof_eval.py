import argparse 
import json 
import os 
import cv2 
import pandas as pd 
from tqdm import tqdm 
from glob import glob 
from model import SMModel 
from process import postprocess 
from dataset import DataGenerator 
from utils import run_length_encode, dice_coef_score 

os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
    config['train']['image_height'] = 256
    config['train']['image_width'] = 1600
    
    oof_preds = []
    oof_true_masks = []
    

