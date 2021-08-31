from typing import Tuple

import cv2
import numpy as np


def grayscale_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def inv_img(img):
    return cv2.bitwise_not(img)


def bgr_img(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def binarization_img(img, threshold=127, mode=cv2.THRESH_BINARY):
    img = grayscale_img(img)
    _, img = cv2.threshold(img, threshold, 255, mode)
    return img


def draw_mask_roi(roi, width: int, height: int):
    # Black in RGB
    black = np.zeros((width, height, 3), np.uint8)
    # Draw mask from ROI
    black = cv2.rectangle(black, roi[:2], roi[2:], (255, 255, 255), -1)
    return binarization_img(black)


def draw_mask_contours(contours, width: int, height: int):
    # Black in RGB
    black = np.zeros((width, height, 3), np.uint8)
    # Draw mask from ROI
    black = cv2.drawContours(black, contours, -1, (255, 255, 255), -1)
    return binarization_img(black)


def draw_label(img, pt, label: str, font_scale=0.3):
    text_size, _ = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=1)
    text_w, text_h = text_size
    img = cv2.rectangle(img, (pt[0], pt[1] - text_h), (pt[0] + text_w, pt[1]), color=(0, 0, 0), thickness=-1)
    return cv2.putText(img, label, (pt[0], pt[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                       color=(255, 255, 255), thickness=1)


def apply_mask_img(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def crop_img_rect(img, rect):
    x, y, w, h = rect
    return img[y:y + h, x:x + w]


def get_video_capture(video_path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(video_path)


def get_video_writer(output_path: str, framerate: float, dimension: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(output_path, fourcc, framerate, dimension)


def get_frame_position(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))


def set_frame_position(video_capture: cv2.VideoCapture, position: int) -> int:
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position))


def get_video_dimension(video_path: str) -> Tuple[int, int]:
    video_capture = get_video_capture(video_path)
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_length(video_path: str) -> int:
    video_capture = get_video_capture(video_path)
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_video_framerate(video_path: str) -> float:
    video_capture = get_video_capture(video_path)
    return float(video_capture.get(cv2.CAP_PROP_FPS))


def filename_append(filename: str, append: str, ext_override: str = None):
    basename, ext = filename.rsplit(".", 1)
    if ext_override is not None:
        ext = ext_override
    return f"{basename}_{append}.{ext}"
