import os, sys
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import yaml
import cv2

from pathlib import Path 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

# FACE RECOGNITION
"""
    Path structure for training:
    - object1/
        + image 1 of object1
        + image 2 of object1
        + .....
    - object2/
        + image 1 of object2
        + image 2 of object2
        + .....
    - .......    
"""


class FaceRecognitionDataset:
    def __init__(self, config):
        if isinstance(config, dict):
            config = config
        elif os.path.isfile(config):
            with open(config) as f:
                config = yaml.safe_load(f)
        else:
            raise TypeError
        self.config = config
        self.data = None
        self.label = None

    # Loader data
    def dataloader(self):
        list_img = []
        list_label = []
        path = self.config['path']
        if os.path.isdir(path):
            for obj, pth in enumerate(os.listdir(path)):
                folder_sub = os.path.join(path, pth)
                for folder in tqdm(os.listdir(folder_sub), desc=f'{pth}'):
                    img = cv2.imread(os.path.join(folder_sub, folder))
                    # Convert BGR to RGB
                    img = img[:, :, ::-1]
                    img = cv2.resize(img, (self.config['resize'], self.config['resize']))
                    list_img.append(img)
                    list_label.append(obj)
            data = np.array(list_img)
            cls = to_categorical(np.array(list_label))
            return data, cls
        else:
            raise TypeError

    # Get dataset
    @staticmethod
    def split_data(ratio: float, data: np.ndarray, label: np.ndarray):
        if ratio <= 0.0 or ratio >= 1.0:
            return f"Ratio not suitable."
        else:
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, label, test_size=ratio, random_state=42)
            return Xtrain, Xtest, Ytrain, Ytest

    # Number class in dataset
    def n_class(self):
        path = self.config['path']
        return len(self.config['path'])


# Plot data image face.
def plot_image(path, ):
    pass


if __name__ == '__main__':
    dataset = FaceRecognitionDataset('D:/Machine_Learning/face-recgnition/config/data.yaml')
    dataset.dataloader()
