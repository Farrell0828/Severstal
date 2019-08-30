from keras.callbacks import Callback 

class DiceCoefCallback(Callback):
    def __init__(self, generator):
        super(DiceCoefCallback, self)__init__()
        self.generator = generator

    def on_train_begin(self):
        return

    def on_epoch_end(self, epoch, logs={}):
        for X, y_true in list(self.generator):
            y_pred = self.model.predict(X)
            
