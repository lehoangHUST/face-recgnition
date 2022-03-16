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

# ArgumentParser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='D:/Machine_Learning/face-recgnition/config/data.yaml')
    parser.add_argument('--pretrain', type=str, default='efficientNet-b0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('-epochs', type=int, default=10)
    args = parser.parse_args()
    return args


# Run train model
def run(args):
    config, pretrain, batch_size, epochs = args.config, args.pretrain, args.batch_size, args.epochs
    # Load data
    print("Loading dataset....")
    Data = FaceRecognitionDataset(config=config)
    data, label = Data.dataloader()
    print("Loaded dataset...")

    # Split dataset
    Xtrain, Xtest, Ytrain, Ytest = Data.split_data(0.8, data, label)

    # Load model
    print("Loading model...")
    model = build_model(pretrain='efficientNet-b0', input_shape=data[0].shape, num_class=2)

    # Use optimizer: adam + loss function: categorical_crossentropy + metrics: accuracy
    # Can design by yourself
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Loaed model...")
    model.summary()

    # Train
    model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[LossCallback()])

    # Evaluate
    model.evaluate(Xtest, Ytest, batch_size=batch_size, epochs=batch_size, verbose=0, callbacks=[LossCallback()])


if __name__ == '__main__':
    args = parse_args()
    run(args)

