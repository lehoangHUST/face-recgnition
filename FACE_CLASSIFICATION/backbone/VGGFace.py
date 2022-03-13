import os
import gdown
from pathlib import Path

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, \
    Activation


# Build model VGGFace
def baseModel():
    # Architecture: https://www.researchgate.net/figure/VGG-Face-CNN-layers_tbl1_301270729
    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.summary()


# url = 'https://drive.google.com/uc?id=1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo'
def loadModel(url='https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5'):
    model = baseModel()
    # -----------------------------------
    output = os.getcwd() + '/.deepface/weights/vgg_face_weights.h5'
    if not os.path.isfile(output):
        print("vgg_face_weights.h5 will be downloaded...")
        gdown.download(url, output, quiet=False)

    # -----------------------------------

    model.load_weights(output)

    # -----------------------------------

    # TO-DO: why?
    descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return descriptor
