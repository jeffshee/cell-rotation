import math
import os
from multiprocessing import Process

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils_v2 import *
from dp import calc_dp


def similarity(img1: np.ndarray, img2: np.ndarray):
    return np.squeeze(cv2.matchTemplate(img1, img2, constants.TM_METHOD))


def calc_stat(crop_video_path: str, bounding_rects, mask_areas, dp_result):
    video_capture = cv2.VideoCapture(crop_video_path)
    video_framerate = get_video_framerate(video_capture)
    stat = pd.DataFrame(
        columns=["bounding_rect_w", "bounding_rect_h", "area", "est_radius",
                 "cycle_frame", "est_rad/s", "est_deg/s"])
    for i, (rect, area, cycle_frame) in enumerate(zip(bounding_rects, mask_areas, dp_result)):
        _, _, rect_w, rect_h = rect
        est_radius = math.sqrt(area / math.pi)
        cycle_sec = cycle_frame / video_framerate
        est_rad_per_sec = 2 * math.pi / cycle_sec
        est_deg_per_sec = 360 / cycle_sec
        stat.loc[i] = [rect_w, rect_h, area, est_radius, cycle_frame, est_rad_per_sec, est_deg_per_sec]
    return stat


def calc_pairwise_similarity(crop_video_path: str, frame_list: list = None, name=""):
    video_capture = cv2.VideoCapture(crop_video_path)
    if not frame_list:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_list.append(frame)
    video_framerate = get_video_framerate(video_capture)
    pairwise_similarity = pd.DataFrame()
    for t1 in tqdm(range(len(frame_list)), desc=f"Calculating {name}"):
        T = min(int(t1 + constants.DELTA_T_SEC_MAX * video_framerate), len(frame_list))
        for i, t2 in enumerate(range(t1, T)):
            if T - t1 == int(constants.DELTA_T_SEC_MAX * video_framerate):
                # Drop last n sec of frames
                pairwise_similarity.loc[t1, i] = similarity(frame_list[t1], frame_list[t2])
    return pairwise_similarity


def plot_heatmap(crop_video_path: str, pairwise_similarity: pd.DataFrame, dp_result, output_path=None):
    video_capture = cv2.VideoCapture(crop_video_path)
    video_framerate = get_video_framerate(video_capture)
    plt.figure()
    plt.title(f"Heatmap of pairwise similarity (t1 and t2) \n{(os.path.basename(crop_video_path))}")
    ax = sns.heatmap(pairwise_similarity)
    ax.set_xlabel("delta [frame] (where t2 = t1 + delta, fps = {:.1f})".format(video_framerate))
    ax.set_ylabel("t1 [frame]")
    if output_path is None:
        plt.show()
    else:
        for delta_t in constants.DELTA_T_SEC_LIST:
            plt.xlim(0, int(delta_t * video_framerate))
            plt.savefig(filename_append(output_path, "{:.1f}s".format(delta_t)))
        # Plot DP result in another fig
        sns.scatterplot(x=dp_result, y=np.arange(len(dp_result)), linewidth=0)
        for delta_t in constants.DELTA_T_SEC_LIST:
            plt.xlim(0, int(delta_t * video_framerate))
            plt.savefig(filename_append(output_path, "{:.1f}s-dp".format(delta_t)))
    plt.close()


def proc(video_path: str, output_dir: str, name: str, params: dict):
    basename = os.path.basename(video_path)
    crop_video_path = os.path.join(output_dir, filename_append(basename, f"{name}-crop"))
    frame_list, bounding_rects, mask_areas = cell_crop_video(video_path, crop_video_path, params, name)
    # Calculate pairwise similarity, DP
    pairwise_similarity = calc_pairwise_similarity(crop_video_path, frame_list, name)
    dp_result = calc_dp(pairwise_similarity.to_numpy())
    csv_sim_path = os.path.join(output_dir, filename_append(basename, f"{name}-sim", "csv"))
    pairwise_similarity.to_csv(csv_sim_path)
    fig_path = filename_append(crop_video_path, "fig", "png")
    plot_heatmap(crop_video_path, pairwise_similarity, dp_result, fig_path)
    # Statistic
    stat = calc_stat(crop_video_path, bounding_rects, mask_areas, dp_result)
    csv_stat_path = os.path.join(output_dir, filename_append(basename, f"{name}-stat", "csv"))
    stat.to_csv(csv_stat_path)


def main(video_path: str, output_dir: str, gui_result: dict):
    process_list = []
    for name, params in gui_result.items():
        process_list.append(Process(target=proc, kwargs=dict(
            video_path=video_path,
            output_dir=output_dir,
            name=name,
            params=params
        )))
    print("Start processing. This might take a while...")
    if process_list:
        for p in process_list:
            p.start()
    if process_list:
        for p in process_list:
            p.join()
    print("Done.")
