# Our source library
import tensorflow
import os
import numpy as np
from tqdm import tqdm
import argparse
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# My source library
from ..utils import augmentation
from ..utils import dataset

rows = 160
cols = 160

trained_model = MobileNetV2(input_shape=(rows, cols, 3),
                            include_top=False,
                            weights='imagenet')

trained_model.trainable = True  # Un-Freeze all the pretrained layers of 'MobileNetV2 for Training.

trained_model.summary()
