"""
目視回転速度.txt

解析方法
目視で，細胞上の特徴となる１点を決定
１コマごとにコマ送りしてその特徴点のx, y座標を記録
フーリエ変換して周波数を抽出

U5L5
0～5秒間の平均回転速度　10.3 radian/sec
40～45秒間の平均回転速度　7.36 radian/sec

U8L8
0～5秒間の平均回転速度　5.89 radian/sec
40～45秒間の平均回転速度　4.42 radian/sec

control-U7L5
0～5秒間の平均回転速度　7.36 radian/sec
40～45秒間の平均回転速度　7.36 radian/sec

control-U7L6
0～5秒間の平均回転速度　5.89 radian/sec
40～45秒間の平均回転速度　5.89 radian/sec
"""
import os

from tqdm import tqdm

from gui import gui
from utils import *


def binarization_video(video_path: str, output_path: str, threshold=127):
    # Input/Output
    video_capture = get_video_capture(video_path)
    video_length = get_video_length(video_path)
    video_dimension = get_video_dimension(video_path)
    video_framerate = get_video_framerate(video_path)
    video_writer = get_video_writer(output_path, video_framerate, video_dimension)

    bar = tqdm(total=video_length, desc=f"Binarization {video_path}")
    set_frame_position(video_capture, 0)
    while video_capture.isOpened() and get_frame_position(video_capture) in range(video_length):
        # Progress
        bar.update(1)
        bar.refresh()
        # Read frame, binarization, write frame
        ret, frame = video_capture.read()
        if not ret:
            break
        frame = binarization_img(frame, threshold, mode=cv2.THRESH_BINARY_INV)
        video_writer.write(bgr_img(frame))  # Convert back to bgr format before writing frame

    video_capture.release()
    video_writer.release()


def cell_detect_video(video_path: str, output_path: str, threshold=127, roi=None):
    # Input/Output
    video_capture = get_video_capture(video_path)
    video_length = get_video_length(video_path)
    video_dimension = get_video_dimension(video_path)
    video_framerate = get_video_framerate(video_path)
    video_writer = get_video_writer(output_path, video_framerate, video_dimension)
    width, height = video_dimension

    if roi is None:
        roi_ratio = (0.33, 0.33, 0.66, 0.66)  # Ratio on width and height. x1,y1,x2,y2.
        roi = (int(roi_ratio[0] * width), int(roi_ratio[1] * height)),
        (int(roi_ratio[2] * width), int(roi_ratio[3] * height))

    bar = tqdm(total=video_length, desc=f"Cell Detect {video_path}")
    set_frame_position(video_capture, 0)
    while video_capture.isOpened() and get_frame_position(video_capture) in range(video_length):
        # Progress
        bar.update(1)
        bar.refresh()

        ret, frame = video_capture.read()
        if not ret:
            break
        frame_copy = frame.copy()
        frame = binarization_img(frame, threshold, cv2.THRESH_BINARY_INV)
        frame = apply_mask_img(frame, draw_mask_roi(roi, width, height))

        contours, hierarchy = cv2.findContours(frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Find the convex hull object for each contour
        # hull_list = []
        # for i in range(len(contours)):
        #     hull = cv2.convexHull(contours[i])
        #     hull_list.append(hull)

        # Filter out the contours that unlikely to be a circle
        # TODO: check cv2.approxPolyDP
        contours = [contours_i for contours_i in contours if len(contours_i) > 4]
        # Get all points from contours
        ptr_list = np.concatenate(contours)
        # Find the convex hull object for all points in contours
        hull_list = [cv2.convexHull(ptr_list)]

        # Cell detection based on contour hierarchy
        # filter_has_parent = np.where(np.squeeze(hierarchy[:, :, 3]) != -1, 1, 0)
        # filter_has_child = np.where(np.squeeze(hierarchy[:, :, 2]) != -1, 1, 0)
        # contours = [c for i, c in enumerate(contours) if filter_has_parent[i] and filter_has_child[i]]
        # contours = [c for i, c in enumerate(contours) if filter_has_parent[i]]

        # frame_copy = cv2.drawContours(frame_copy, contours=contours, contourIdx=-1, color=(0, 255, 0),
        #                               thickness=-1)
        frame_copy = cv2.drawContours(frame_copy, contours=hull_list, contourIdx=-1, color=(0, 0, 255),
                                      thickness=-1)
        frame_copy = cv2.rectangle(frame_copy, roi[:2], roi[2:], color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # Draw label on box
        frame_copy = draw_label(frame_copy, roi[:2], "ROI")
        video_writer.write(frame_copy)

    video_capture.release()
    video_writer.release()


def cell_crop_video(video_path: str, output_path: str, threshold=127, roi=None):
    # Input/Output
    video_capture = get_video_capture(video_path)
    video_length = get_video_length(video_path)
    width, height = get_video_dimension(video_path)
    video_framerate = get_video_framerate(video_path)

    if roi is None:
        roi_ratio = (0.33, 0.33, 0.66, 0.66)  # Ratio on width and height. x1,y1,x2,y2.
        roi = (int(roi_ratio[0] * width), int(roi_ratio[1] * height)),
        (int(roi_ratio[2] * width), int(roi_ratio[3] * height))

    bar = tqdm(total=video_length, desc=f"Cell Detect {video_path}")
    set_frame_position(video_capture, 0)
    bounding_rects = []
    masked_frames = []
    while video_capture.isOpened() and get_frame_position(video_capture) in range(video_length):
        # Progress
        bar.update(1)
        bar.refresh()

        ret, frame = video_capture.read()
        if not ret:
            break
        frame_copy = frame.copy()
        frame = binarization_img(frame, threshold, cv2.THRESH_BINARY_INV)
        frame = apply_mask_img(frame, draw_mask_roi(roi, width, height))

        contours, hierarchy = cv2.findContours(frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Filter out the contours that unlikely to be a circle
        contours = [contours_i for contours_i in contours if len(contours_i) > 4]
        # Get all points from contours
        ptr_list = np.concatenate(contours)
        # Find the convex hull object for all points in contours
        hull_list = [cv2.convexHull(ptr_list)]

        bounding_rects.append(cv2.boundingRect(hull_list[0]))
        # Mask unrelated region, save to frame list
        mask = draw_mask_contours(hull_list, width, height)
        masked_frames.append(apply_mask_img(frame_copy, mask))

    video_capture.release()
    bar.close()

    # Output cropped video
    mean_width, mean_height = [round(x) for x in np.array(bounding_rects).mean(axis=0)][-2:]
    video_writer = get_video_writer(output_path, video_framerate, (mean_width, mean_height))

    bar = tqdm(total=len(masked_frames), desc="Cropping")
    for (x, y, w, h), frame in zip(bounding_rects, masked_frames):
        # Progress
        bar.update(1)
        bar.refresh()

        mid_x, mid_y = x + w // 2, y + h // 2
        new_bounding_rect = (mid_x - mean_width // 2, mid_y - mean_height // 2, mean_width, mean_height)
        frame_cropped = crop_img_rect(frame, new_bounding_rect)
        video_writer.write(frame_cropped)
    video_writer.release()


if __name__ == "__main__":
    import pickle

    dataset_root = "dataset"
    output_root = "output"
    file_list = [os.path.join(dataset_root, f) for f in os.listdir(dataset_root)]
    roi_cache_path = "roi.pkl"
    if os.path.isfile(roi_cache_path):
        with open(roi_cache_path, "rb") as pkl:
            roi_cache = pickle.load(pkl)
    else:
        roi_cache = {}

    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    # threshold = 180
    # for f in file_list:
    #     f_basename = os.path.basename(f)
    #     output_path = os.path.join(output_root, filename_append(f_basename, "bin"))
    #     binarization_video(f, output_path=output_path, threshold=threshold)

    for f in file_list:
        f_basename = os.path.basename(f)
        output_path = os.path.join(output_root, filename_append(f_basename, "contr"))
        roi, threshold = gui(f, cached_roi=roi_cache.get(f, None))
        with open(roi_cache_path, "wb") as pkl:
            roi_cache[f] = roi
            pickle.dump(roi_cache, pkl)
        # cell_detect_video(f, output_path=output_path, threshold=threshold, roi=roi)
        cell_crop_video(f, output_path=output_path, threshold=threshold, roi=roi)
