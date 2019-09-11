import argparse 
import json 
import os 
import numpy as np 
from tqdm import tqdm 
from model import SMModel 
from process import postprocess 
from dataset import DataGenerator 
from utils import dice_coef_score 

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
    sm_model = SMModel(config['model'])
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
            y_preds = postprocess(y_preds, config['postprocess'], True)
            oof_preds.append(y_preds)
            y = y[:, :, :, :4]
            oof_true_masks.append(y)
    oof_preds = np.concatenate(oof_preds)
    oof_true_masks = np.concatenate(oof_true_masks)
    
    cv_dice_coef = dice_coef_score(oof_true_masks, oof_preds)
    print('CV Dice Coef Score: {}'.format(cv_dice_coef))

if __name__ == '__main__':
    _main_()
