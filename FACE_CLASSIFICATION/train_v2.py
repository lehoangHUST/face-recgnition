# Our source library
import numpy as np
import cv2
import os
import sys
import argparse
import pickle
from tqdm import tqdm

# Standardize features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Accuracy score
from sklearn.metrics import accuracy_score

# My source library
from utils.dataset import FaceRecognitionDataset
from utils.general import LossCallback, save_modelckpt
from backbone.network import build_model
from backbone import FaceNet, VGGFace

ALGORITHMS = ['svm', 'softmax']
PRETRAIN = {'vggface': VGGFace.VGGFaceModel(),
            'facenet': FaceNet.InceptionResNetV2()}


# ArgumentParser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='D:/Machine_Learning/face-recgnition/config/data.yaml')
    parser.add_argument('--algorithms', type=str, default='svm')
    parser.add_argument('--pretrain', type=str, default='vggface')
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    return args


# Train model
def run(args):
    config, algorithms, pretrain, save = args.config, args.algorithms, args.pretrain, args.save_model

    if algorithms.lower() in ALGORITHMS and pretrain.lower() in PRETRAIN:
        # Load data
        print("Loading dataset....")
        Data = FaceRecognitionDataset(config=config)
        data, label = Data.dataloader()
        print("Loaded dataset...")

        # Split dataset
        Xtrain, Xtest, Ytrain, Ytest = Data.split_data(0.2, data, label)

        # Load model pretrain
        model = PRETRAIN.get(pretrain)

        # Embedding vector train by input shape
        embeddings_train = np.zeros((Xtrain.shape[0], 2622))
        embeddings_test = np.zeros((Xtest.shape[0], 2622))
        for i, img in enumerate(Xtrain):
            embedding_vector = model.predict(np.expand_dims(img, axis=0))[0]
            embeddings_train[i] = embedding_vector

        for i, img in enumerate(Xtest):
            embedding_vector = model.predict(np.expand_dims(img, axis=0))[0]
            embeddings_test[i] = embedding_vector


        #  Train
        clf = SVC(C=5., gamma=0.001)
        clf.fit(embeddings_train, Ytrain)
        y_predict = clf.predict(embeddings_test)

        print('Accuracy for predict model is: ', accuracy_score(Ytest, y_predict))

        # Save model
        if save:
            pickle.dump(model, open('model.pkl', 'wb'))

    else:
        raise TypeError


if __name__ == '__main__':
    args = parse_args()
    run(args)