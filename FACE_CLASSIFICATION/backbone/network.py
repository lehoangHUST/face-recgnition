import tensorflow as tf
import os
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Build NETWORK
class Network(tf.keras.Model):
    def __init__(self, in_shape: tuple):
        super().__init__()
        self.trained_model = EfficientNetB0(input_shape=in_shape,
                                            include_top=False,
                                            weights='imagenet')
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(105, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self):
        self.trained_model.trainable = True  # Un-Freeze all the pretrained layers of 'MobileNetV2 for Training.
        last_layer = self.trained_model.get_layer('out_relu')
        last_layer_output = last_layer.output  # Saves the output of the last layer of the MobileNetV2.
        x = tf.keras.layers.GlobalAveragePooling2D()(last_layer_output)
        # Add a Dropout layer.
        x = self.dropout(x)
        # Add a final softmax layer for classification.
        x = self.dense1(x)
        return self.dense2(x)


if __name__ == '__main__':
    net = Network(in_shape=(160, 160, 3))