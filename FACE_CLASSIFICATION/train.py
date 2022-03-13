# Our source library
import tensorflow
from tensorflow.keras.applications import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Dropout, concatenate, Activation, BatchNormalization, GlobalAvgPool2D, GlobalAveragePooling2D, LeakyReLU, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant

# My source library
from utils.dataset import FaceRecognitionDataset
from backbone.network import Network


#Callback Function which stops training when accuracy reaches 98%.
class myCallback(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True


if __name__ == '__main__':
    Data = FaceRecognitionDataset(config='D:/Machine_Learning/face-recgnition/config/data.yaml')
    data, label = Data.dataloader()
    Xtrain, Xtest, Ytrain, Ytest = Data.split_data(0.8, data, label)

    model = Network(pre_train='efficientNet-b0', input_shape=(160, 160, 3), num_class=2)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # Summary of the model.
    model.summary()
    callbacks = myCallback()
    
    model.fit(Xtrain, Ytrain, batch_size=32, epochs=50, validation_data=(Xtest, Ytest), verbose=2)