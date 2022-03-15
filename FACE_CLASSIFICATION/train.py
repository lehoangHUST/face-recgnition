# Our source library
import tensorflow
import os, sys
import argparse
import numpy as np
from tensorflow.keras.applications import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Dropout, concatenate, Activation, BatchNormalization, GlobalAvgPool2D, GlobalAveragePooling2D, LeakyReLU, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant

# My source library
from utils.dataset import FaceRecognitionDataset
from utils.general import LossCallback, save_modelckpt
from backbone.network import build_model



if __name__ == '__main__':
    Data = FaceRecognitionDataset(config='D:/Machine_Learning/face-recgnition/config/data.yaml')
    data, label = Data.dataloader()
    Xtrain, Xtest, Ytrain, Ytest = Data.split_data(0.8, data, label)

    model = build_model(pretrain='efficientNet-b0', input_shape=(224, 224, 3), num_class=2)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Summary of the model.
    model.summary()
    
    model.fit(Xtrain, Ytrain, batch_size=32, epochs=50, validation_data=(Xtest, Ytest), verbose=0, callbacks=[LossCallback()])