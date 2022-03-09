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
    trained_model = EfficientNetB0(input_shape=(224, 224, 3),
                                include_top=False,
                                weights='imagenet')
    trained_model.summary()
    trained_model.trainable = True  # Un-Freeze all the pretrained layers of 'MobileNetV2 for Training.
    last_layer = trained_model.get_layer('top_activation')
    last_layer_output = last_layer.output  # Saves the output of the last layer of the MobileNetV2.
    x = GlobalAveragePooling2D()(last_layer_output)
    # Add a Dropout layer.
    x = Dropout(0.8)(x)
    # Add a final softmax layer for classification.
    x = Dense(2, activation='softmax')(x)

    model = Model(trained_model.input, x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # Summary of the model.
    model.summary()
    callbacks = myCallback()
    
    model.fit(Xtrain, Ytrain, batch_size=32, epochs=50, validation_data=(Xtest, Ytest), verbose=2)