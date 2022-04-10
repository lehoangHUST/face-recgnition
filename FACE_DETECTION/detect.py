# Our source code
import cv2
import numpy as np
import os, sys
import argparse
import matplotlib.pyplot as plt

# My source code
from mtcnn.mtcnn import MTCNN
from bboxes import draw_bbox


# Add argument
parser = argparse.ArgumentParser()
parser.add_argument('--image', default='D:/Machine_Learning/face-recgnition/Image/image.jpg', type=str, help='Detect face in image ')
parser.add_argument('--images', default=None, type=str, help='Detect face in images')
parser.add_argument('--video', default=None, type=str, help='Detect in video or webcam')
parser.add_argument('--display_bbox', default=True, type=bool, help='Show rectange bbox of face humans.')
parser.add_argument('--save', default=True, type=bool, help='Save image, folder of image or video or webcam')
args = parser.parse_args()


def evalimage(detector, inp: str):
    img = cv2.imread(inp)
    results = detector.detect_faces(img)
    bboxes = []
    for result in results:
        x, y, w, h = result['box']
        bboxes.append([x, y, w, h])
    img = draw_bbox(img, bboxes)
    if not args.save:
        plt.imshow(img[:, :, ::-1])
        plt.show()
    else:
        path_save = inp.split('.')[0] + '_out.jpg'
        cv2.imwrite(path_save, img)


# Inference
def run(args):
    detector = MTCNN()

    # args.image or images or video have type: inp:output or inp
    if args.image is not None:
        inp = args.image
        evalimage(detector, inp)
    elif args.images is not None:
        for image in os.listdir(args.image):
            path_image = os.path.join(args.image, image)
            evalimage(detector, path_image)
    elif args.video is not None:
        pass


if __name__ == '__main__':
    run(args)
