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

IMG_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

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
    # Init
    def __init__(self, path: str, ratio: float):
        list_img = []
        list_label = []
        self.ratio = ratio
        # Check conditional ratio
        if self.ratio >= 1.0 or self.ratio <= 0.0:
            raise ValueError
        self.classes = len(os.listdir(path))
        if os.path.isdir(path):
            for obj, pth in enumerate(os.listdir(path)):
                folder_sub = os.path.join(path, pth)
                for folder in tqdm(os.listdir(folder_sub), desc=f'{pth}'):
                    img = cv2.imread(os.path.join(folder_sub, folder))
                    # Convert BGR to RGB
                    img = img[:, :, ::-1]
                    img = cv2.resize(img, (224, 224))
                    list_img.append(img)
                list_label.append([obj + 1]*len(os.listdir(folder_sub)))
            self.features = np.array(list_img)
            self.label = np.array(list_label)
            print(self.features.shape)
            print(self.label)
            # train test split
            self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.features, self.label, test_size=self.ratio)
        else:
            raise TypeError

    # Get dataset
    def get_data(self):
        return self.Xtrain, self.Xtest, self.Ytrain, self.Ytest


if __name__ == '__main__':
    path = 'D:/Machine_Learning/Face_Recognition/105_classes_pins_dataset'
    dataset = FaceRecognitionDataset(path)