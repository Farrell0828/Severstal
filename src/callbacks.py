import keras 
from process import postprocess 

class DiceCoefCallback(keras.callbacks.Callback):
    def __init__(self, generator):
        super(DiceCoefCallback, self).__init__()
        self.generator = generator
        self.eps = 1e-7
        self.postprocess_config = {
            'threshold': 0.5, 
            'filter_small_region': False, 
            'min_size': [0, 0, 0, 0]
        }

    def on_epoch_end(self, epoch, logs={}):
        dice_coef = 0
        for X, y_true in list(self.generator):
            y_pred = self.model.predict(X)
            y_pred = postprocess(y_pred, self.postprocess_config)
            y_true = y_true[:, :, :, :4]
            inter = (y_true * y_pred).sum(1).sum(1)
            union = y_true.sum(1).sum(1) + y_pred.sum(1).sum(1)
            dice_coef_batch = (2*inter + self.eps) / (union + self.eps)
            dice_coef += dice_coef_batch.sum()
        dice_coef /= (self.generator.n_samples * 4)
        logs.update({'dice_coef_score': dice_coef})
        print('Epoch {} Validation Dice Coefficent Score: {}.\n'.format(epoch+1, dice_coef))


class SWA(keras.callbacks.Callback):
    """
    Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
    Implementaton in Keras from user defined epochs assuming constant 
    learning rate
    Cyclic learning rate implementation in https://arxiv.org/abs/1803.05407 
    not implemented
    Created on July 4, 2018
    @author: Krist Papadopoulos
    """
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i]
                    /((epoch - self.swa_epoch)  + 1))

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
