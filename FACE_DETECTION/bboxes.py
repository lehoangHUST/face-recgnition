import numpy as np
import cv2


def draw_bbox(img, bboxes):
    if isinstance(bboxes, (tuple, list, np.ndarray)):
        for bbox in bboxes:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 155, 255), 2)
    return img