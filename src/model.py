
import segmentation_models as sm 
from keras.models import Model 

class SMModel(object):

    def __init__(self, config):
        self.config = config
        if config['type'] == 'Unet':
            self.model = sm.Unet(backbone_name=config['backbone'], 
                                 classes=config['n_class'],
                                 activation=config['activate']
                                 encoder_weights=config.get('encoder_weights', None), 
                                 )