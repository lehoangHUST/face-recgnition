import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Dropout, concatenate, Activation, BatchNormalization, GlobalAvgPool2D, GlobalAveragePooling2D, LeakyReLU, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_Model(pretrain: str,
                input_shape: tuple,
                num_class: int):
    """
    :param pretrain: Network pretrained by tensorflow.keras.applications include [EfficientNet, MobileNet, ResNet, v.v]
    :param input_shape: Image shape with (width, height, channel)
    :param num_class: Output classification
    :return: model new (transfer learning + create layer)
    """
    __MODEL_PRETRAIN__ = ['efficientNet-b0',
                          'efficientNet-b1',
                          'efficientNet-b2',
                          'mobile-netv2']
    model = {
        'efficientNet-b0': EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet'),
        'efficientNet-b1': EfficientNetB1(input_shape=input_shape, include_top=False, weights='imagenet'),
        'efficientNet-b2': EfficientNetB2(input_shape=input_shape, include_top=False, weights='imagenet'),
        'efficientNet-b3': EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet'),
        'efficientNet-b4': EfficientNetB4(input_shape=input_shape, include_top=False, weights='imagenet'),
        'efficientNet-b5': EfficientNetB5(input_shape=input_shape, include_top=False, weights='imagenet'),
        'efficientNet-b6': EfficientNetB6(input_shape=input_shape, include_top=False, weights='imagenet'),
        'mobile-netv2':    MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    }

    if pretrain in model:
        MODEL = model.get(pretrain)
    else:
        print("pre trained model not find. Defaults efficienet-b0")
        MODEL = model.get('efficientNet-b0')
    name_last_layer = MODEL.get_layer(index=len(MODEL.layers)-1).name
    last_layer = MODEL.get_layer(name_last_layer)
    last_layer_output = last_layer.output  # Saves the output of the last layer of the MobileNetV2.
    MODEL.trainable = True  # Un-Freeze all the pretrained layers of 'MobileNetV2 for Training.
    x = GlobalAveragePooling2D()(last_layer_output)
    # Add a Dropout layer.
    x = Dropout(0.8)(x)
    # Add a final softmax layer for classification.
    x = Dense(num_class, activation='softmax')(x)

    model = Model(MODEL.input, x)
    return model


if __name__ == '__main__':
    build_Model("efficientNet-b0", input_shape=(160, 160, 3), num_class=3)
