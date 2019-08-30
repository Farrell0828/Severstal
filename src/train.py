import argparse 
import json 

from model import SMModel 

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-c',
    '--config',
    help='Config file.'
)

if __name__ == '__main__':

    args = argparser.parse_args()
    config_path = args.config
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    sm_model = SMModel(config['model'])
    sm_model.train(config['train'])