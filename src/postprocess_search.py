import argparse 
import json 
import os 
import numpy as np 
from tqdm import tqdm 
from model import SMModel 
from process import postprocess 
from dataset import DataGenerator 
from utils import dice_coef_score 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-c',
    '--config',
    help='Config file.'
)

def oof_predict(sm_model, config):
    oof_preds = []
    oof_true_masks = []
    for i in range(5):
        config['train']['fold'] = i
        generator = DataGenerator(config=config['train'], 
                                  preprocessing=sm_model.preprocessing,
                                  n_class=sm_model.n_class, 
                                  split='val', 
                                  full_size_mask=True)
        weithts_path = os.path.join(config['train']['save_model_folder'], 
                                    'val_best_fold_{}_weights.h5'.format(i))
        sm_model.model.load_weights(weithts_path)
        print('Fold {} eval begin.'.format(i))
        for X, y in tqdm(list(generator)):
            y_preds = sm_model.model.predict(X)
            oof_preds.append(y_preds)
            y = y[:, :, :, :4]
            oof_true_masks.append(y)
    oof_preds = np.concatenate(oof_preds)
    oof_true_masks = np.concatenate(oof_true_masks)
    return oof_preds, oof_true_masks

def oof_eval(oof_preds, oof_true_masks, config):
    oof_preds = postprocess(oof_preds, config['postprocess'], True)
    cv_dice_coef = dice_coef_score(oof_true_masks, oof_preds)
    print('CV Dice Coef Score: {}'.format(cv_dice_coef))
    return cv_dice_coef

def search_min_size(oof_preds, oof_true_masks, n):
    best_score = 0
    best_config = config['postprocess']
    min_size_cadidates = [2**(n+5), 2**(n+6), 2**(n+7), 2**(n+8), 2**(n+9)]
    print('min_size Cadidates: ', min_size_cadidates)
    for min_size in min_size_cadidates:
        config['postprocess']['min_size'] = [1024, 1024, 1024, 1024]
        config['postprocess']['min_size'][n-1] = min_size
        print('min_size this round: ', config['postprocess']['min_size'])
        score = oof_eval(oof_preds, oof_true_masks, config)
        if score > best_score:
            best_config = config['postprocess'].copy()
            best_score = score
        print()
    print('Best score: ', best_score)
    print('Best score config: ', best_config)
    print()

if __name__ == '__main__':
    args = argparser.parse_args()
    config_path = args.config
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
        
    sm_model = SMModel(config['model'])
    oof_preds, oof_true_masks = oof_predict(sm_model, config)

    for i in range(4):
        print('Search for class {} begin...'.format(i+1))
        search_min_size(oof_preds, oof_true_masks, i+1)
