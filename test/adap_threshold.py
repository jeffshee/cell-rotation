import os

import cv2
import matplotlib.pyplot as plt

from utils import get_video_capture, get_video_dimension, grayscale_img

dataset_root = "../dataset"

file_list = [os.path.join(dataset_root, f) for f in os.listdir(dataset_root)]
file_list = list(filter(lambda file_path: os.path.isfile(file_path), file_list))
for video_path in file_list:
    video_capture = get_video_capture(video_path)
    video_dimension = get_video_dimension(video_path)
    width, height = video_dimension
    ret, frame = video_capture.read()

    roi = cv2.selectROI(None, frame)
    if roi == (0, 0, 0, 0):
        # Set entire image as ROI if ROI is not selected
        roi = (0, 0, width, height)
    else:
        # [x, y, w, h] -> [x1, y1, x2, y2]
        x, y, w, h = roi
        roi = x, y, x + w, y + h

    frame = grayscale_img(frame)
    frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
    mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    plt.hist(blur.ravel(), 256)
    plt.show()

    retval, img = cv2.threshold(blur, 0, 255, mode)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 10)
    # print(retval)
    plt.imshow(blur)
    plt.show()
    plt.imshow(thresh)
    plt.show()
