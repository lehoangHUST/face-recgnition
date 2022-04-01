# Our source library
import numpy as np
import cv2
import os
import sys
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Standardize features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# My source library
from utils.dataset import FaceRecognitionDataset
from utils.metrics import norm_p
from utils.general import LossCallback, save_modelckpt
from backbone.network import build_model
from backbone import FaceNet, VGGFace


def show_pair(img1, img2, model):
    if os.path.isfile(img1) and os.path.isfile(img2):
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        plt.figure(figsize=(8, 3))
        plt.suptitle(f'Distance between {img1} & {img2}= {norm_p(img1, img2):.2f}')
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
    else:
        raise TypeError