from sys import platform
from typing import Tuple

import numpy as np
from tqdm import tqdm

import cv2
from constants import constants

"""
Video
"""


def get_frame_position(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) + 1


def set_frame_position(video_capture: cv2.VideoCapture, position: int) -> int:
    return int(video_capture.set(cv2.CAP_PROP_POS_FRAMES, position - 1))


def get_video_dimension(video_capture: cv2.VideoCapture) -> Tuple[int, int]:
    return int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def get_video_length(video_capture: cv2.VideoCapture) -> int:
    return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def get_video_framerate(video_capture: cv2.VideoCapture) -> float:
    return float(video_capture.get(cv2.CAP_PROP_FPS))


def get_video_writer(output_path: str, framerate: float, dimension: Tuple[int, int]) -> cv2.VideoWriter:
    # FFmpeg: http://ffmpeg.org/doxygen/trunk/isom_8c-source.html
    # ImageJ: Uncompressed palettized 8-bit RGBA (1987535218) fourcc: rawv
    # RGBA
    # fourcc = cv2.VideoWriter_fourcc(*"RGBA")  # Working (kind of)
    # PNG
    # fourcc = cv2.VideoWriter_fourcc(*"png ")  # Working (best compatibility) (Doesn't work on Windows!) (x)
    # Uncompressed RGB
    # fourcc = cv2.VideoWriter_fourcc(*"raw ") # Didn't encode(x)
    # Uncompressed YUV422
    # fourcc = cv2.VideoWriter_fourcc(*"yuv2")  # Encoded but didn't play(x)
    #
    # fourcc = cv2.VideoWriter_fourcc(*"XVID") # Very lossy(x)
    # fourcc = cv2.VideoWriter_fourcc(*"IYUV")  # Working on Windows
    # fourcc = -1
    if platform == "linux" or platform == "linux2":
        fourcc = cv2.VideoWriter_fourcc(*"png ")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"IYUV")
    return cv2.VideoWriter(output_path, fourcc, framerate, dimension)


"""
File
"""


def filename_append(filename: str, append: str, ext_override: str = None):
    basename, ext = filename.rsplit(".", 1)
    if ext_override is not None:
        ext = ext_override
    return f"{basename}-{append}.{ext}"


"""
Image Processing
"""


def draw_mask_contours(contours, width: int, height: int):
    # Black in RGB
    black = np.zeros((height, width, 3), np.uint8)
    # Draw mask from ROI
    black = cv2.drawContours(black, contours, -1, (255, 255, 255), -1)
    return cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)


def draw_mask_radius(cX, cY, radius, width: int, height: int):
    # Black in RGB
    black = np.zeros((height, width, 3), np.uint8)
    # Draw circle mask
    black = cv2.circle(black, (cX, cY), radius, (255, 255, 255), -1)
    return cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)


def apply_mask_img(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


def center_pad(img, target_w, target_h):
    height, width, channel = img.shape
    if target_w == width and target_h == height:
        return img
    top = (target_h - height) // 2
    left = (target_w - width) // 2
    bottom = target_h - top - height
    right = target_w - left - width
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def mask_area(mask):
    return cv2.countNonZero(mask)


def cell_crop_video(video_path: str, output_path: str, params: dict, name: str = ""):
    # Input/Output
    video_capture = cv2.VideoCapture(video_path)
    video_length = get_video_length(video_capture)
    x, y, w, h = params["roi"]
    video_framerate = get_video_framerate(video_capture)

    bar = tqdm(total=video_length, desc=f"Detecting {name}")
    set_frame_position(video_capture, 0)
    bounding_rects = []
    mask_areas = []
    masked_frames = []
    output_frames = []
    while video_capture.isOpened():
        # Progress
        bar.update(1)
        bar.refresh()

        ret, frame = video_capture.read()
        if not ret:
            break
        # Crop according ROI
        frame_roi = frame[y:y + h, x:x + w]
        frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        if constants.APPLY_GAUSSIAN_BLUR:
            frame_blur = cv2.GaussianBlur(frame_roi_gray,
                                          (constants.GAUSSIAN_BLUR_KSIZE, constants.GAUSSIAN_BLUR_KSIZE),
                                          constants.GAUSSIAN_BLUR_SIGMA)
        else:
            frame_blur = frame_roi_gray
        if params["adaptive_thresh"]:
            frame_bin = cv2.adaptiveThreshold(frame_blur, 255, params["adaptive_method"], constants.THRESH_TYPE,
                                              params["block_size"], params["C"])
        elif params["auto_thresh"]:
            threshold, frame_bin = cv2.threshold(frame_blur, 0, 255, constants.THRESH_TYPE | cv2.THRESH_OTSU)
        else:
            _, frame_bin = cv2.threshold(frame_blur, params["threshold"], 255, constants.THRESH_TYPE)

        # Filtering based on radius around the centroid
        radius = constants.FILTER_RADIUS
        M = cv2.moments(frame_bin, True)
        cX = int(M["m10"] / (M["m00"] + 1e-14))
        cY = int(M["m01"] / (M["m00"] + 1e-14))
        mask = draw_mask_radius(cX, cY, radius, w, h)
        frame_bin = apply_mask_img(frame_bin, mask)

        contours, hierarchy = cv2.findContours(frame_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        # Filter out the contours that unlikely to be a circle
        contours = [contours_i for contours_i in contours if len(contours_i) > 4]
        if len(contours) == 0:
            print("No contour found! Please adjust parameters.")
        # Get all points from contours
        ptr_list = np.concatenate(contours)
        # Find the convex hull object for all points in contours
        hull_list = [cv2.convexHull(ptr_list)]
        bounding_rects.append(cv2.boundingRect(hull_list[0]))
        # Mask unrelated region, save to frame list
        mask = draw_mask_contours(hull_list, w, h)
        masked_frames.append(apply_mask_img(frame_roi, mask))
        mask_areas.append(mask_area(mask))

    video_capture.release()
    bar.close()

    # Output cropped video
    mean_width, mean_height = [round(x) for x in np.array(bounding_rects).mean(axis=0)][-2:]
    video_writer = get_video_writer(output_path, video_framerate, (mean_width, mean_height))

    bar = tqdm(total=len(masked_frames), desc=f"Cropping {name}")
    for (x, y, w, h), frame in zip(bounding_rects, masked_frames):
        # Progress
        bar.update(1)
        bar.refresh()

        mid_x, mid_y = x + w // 2, y + h // 2
        new_x, new_y, new_w, new_h = mid_x - mean_width // 2, mid_y - mean_height // 2, mean_width, mean_height
        # Avoid negative value in slicing (slice will be invalid otherwise)
        new_x, new_y = max(0, new_x), max(0, new_y)
        frame_cropped = frame[new_y:new_y + new_h, new_x:new_x + new_w]
        # Center pad to make sure every frame has the same size
        frame_cropped = center_pad(frame_cropped, mean_width, mean_height)
        if output_frames:
            assert frame_cropped.shape == output_frames[-1].shape
        video_writer.write(frame_cropped)
        output_frames.append(frame_cropped)
    video_writer.release()
    return output_frames, bounding_rects, mask_areas
