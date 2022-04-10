# Our source code
import cv2
import numpy as np
import os, sys
import argparse
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# My source code
from mtcnn.mtcnn import MTCNN
from FACE_DETECTION.bboxes import draw_bbox
from FACE_CLASSIFICATION.utils.dataset import FaceRecognitionDataset
from FACE_CLASSIFICATION.utils.general import LossCallback, save_modelckpt
from FACE_CLASSIFICATION.backbone.network import build_model
from FACE_CLASSIFICATION.backbone import FaceNet, VGGFace


def load_model():
    # Load model pretrain VGGFace or ArcFace
    model_pretrain = VGGFace.VGGFaceModel()
    # Load model svm for classification
    model_class = pickle.load(open('D:/model.pkl', 'rb'))
    return [model_pretrain, model_class]


if __name__ == '__main__':
    model = load_model()
    img = cv2.imread('D:/Machine_Learning/face-recgnition/Image/maxresdefault.jpg')
    detect_face = MTCNN()
    results = detect_face.detect_faces(img)
    bbox = results[0]['box']
    img = cv2.resize(img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :], (224, 224))
    img = img/255
    embedding_vector = model[0].predict(np.expand_dims(img, axis=0))[0].reshape(1, -1)
    print(model[1].predict(X))