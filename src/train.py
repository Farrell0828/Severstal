import argparse 
import json 
import os 
from model import SMModel 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-c',
    '--config',
    help='Config file.'
)

argparser.add_argument(
    '-f',
    '--fold',
    help='Which fold to train.'
)

if __name__ == '__main__':

    args = argparser.parse_args()
    config_path = args.config
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    fold = int(args.fold)
    if fold == -1:
        for i in range(5):
            config['train']['fold'] = i
            sm_model = SMModel(config['model'])
            sm_model.train(config['train'])
    else:
        config['train']['fold'] = fold
        sm_model = SMModel(config['model'])
        sm_model.train(config['train'])
