import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import *

# Which method to use for template matching, default is TM_CCOEFF_NORMED.
TM_METHOD = [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED][1]

# How many frames (DELTA_T_SEC_MAX * fps) to process.
# The computational complexity is O((DELTA_T_SEC_MAX * fps) * N)
# where N is the length of the video.
DELTA_T_SEC_LIST = [1.0, 2.0, 3.0]
DELTA_T_SEC_MAX = max(DELTA_T_SEC_LIST)

"""
Deprecated, keep for future reference
"""

# from scipy.signal import find_peaks

# SLIDING_WINDOW_SEC = 3.0

# class MethodA:
#     def __init__(self, video_path: str):
#         self.frame_list = []
#         self.get_frame_list(video_path)
#         self.video_framerate = get_video_framerate(video_path)
#         self.video_length = get_video_length(video_path)
#
#     def get_frame_list(self, video_path: str):
#         video_capture = get_video_capture(video_path)
#         while video_capture.isOpened():
#             ret, frame = video_capture.read()
#             if not ret:
#                 break
#             self.frame_list.append(frame)
#
#     def x(self, t):
#         return self.frame_list[self.sec_to_frame_i(t)]
#
#     def sec_to_frame_i(self, sec):
#         return int(sec * self.video_framerate)
#
#     def frame_i_to_sec(self, frame_i):
#         return int(frame_i / self.video_framerate)
#
#     def calc(self, delta, t=SLIDING_WINDOW_SEC / 2):
#         if t < SLIDING_WINDOW_SEC / 2 or t > self.frame_i_to_sec(self.video_length) - SLIDING_WINDOW_SEC / 2:
#             return np.nan
#         taus = np.linspace(t - SLIDING_WINDOW_SEC / 2, t + SLIDING_WINDOW_SEC / 2, num=50)
#         return np.mean([similarity(self.x(tau), self.x(tau + delta)) for tau in taus])

# def plot_s_against_delta(video_path: str, output_path=None):
#     s = MethodA(video_path)
#     delta = np.linspace(0, SLIDING_WINDOW_SEC, num=50)
#     y = np.array([s.calc(d) for d in delta])
#
#     # Find peaks
#     peak_idx, _ = find_peaks(y)
#     peak_delta = delta[peak_idx]
#     peak_y = y[peak_idx]
#
#     plt.figure()
#     plt.title(f"S against delta \n{(os.path.basename(video_path))}")
#     plt.xlabel("delta [sec]")
#     plt.ylabel("S")
#     plt.plot(delta, y)
#
#     plt.scatter(peak_delta, peak_y, c="red", marker="+")
#     for peak_delta_i, peak_y_i in zip(peak_delta, peak_y):
#         plt.axvline(peak_delta_i, c="grey", linestyle="--")
#         plt.annotate("%.2f" % peak_delta_i, (peak_delta_i, peak_y_i))
#     if output_path is None:
#         plt.show()
#     else:
#         plt.savefig(output_path)
#     plt.close()

"""
Method A (Ver. 2)
"""


class MethodA2:
    def __init__(self, video_path: str = None, frame_list: list = None):
        assert video_path or frame_list
        self.frame_list = []
        if frame_list:
            print("Using raw frames")
            self.frame_list = frame_list
        else:
            self.get_frame_list(video_path)
        self.frame_shape = self.frame_list[0].shape
        self.video_length = len(self.frame_list)
        self.video_framerate = get_video_framerate(video_path)
        self.pairwise_sim_compact = pd.DataFrame()

    def get_frame_list(self, video_path: str):
        video_capture = get_video_capture(video_path)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            self.frame_list.append(frame)

    def calc(self):
        for t1 in range(len(self.frame_list)):
            T = min(int(t1 + DELTA_T_SEC_MAX * self.video_framerate), len(self.frame_list))
            for i, t2 in enumerate(range(t1, T)):
                if T - t1 == int(DELTA_T_SEC_MAX * self.video_framerate):
                    # Drop last n sec of frames
                    self.pairwise_sim_compact.loc[t1, i] = similarity(self.frame_list[t1], self.frame_list[t2])
        return self.pairwise_sim_compact


def similarity(img1: np.ndarray, img2: np.ndarray):
    return np.squeeze(cv2.matchTemplate(img1, img2, TM_METHOD))


def plot_pairwise_similarity_heatmap_compact(video_path: str, frame_list: list = None, output_path=None):
    pairwise_sim_compact = MethodA2(video_path, frame_list).calc()
    video_framerate = get_video_framerate(video_path)
    plt.figure()
    plt.title(f"Heatmap of pairwise similarity (t1 and t2) \n{(os.path.basename(video_path))}")
    ax = sns.heatmap(pairwise_sim_compact)
    ax.set_xlabel("delta [frame] (where t2 = t1 + delta, fps = {:.1f})".format(video_framerate))
    ax.set_ylabel("t1 [frame]")
    if output_path is None:
        plt.show()
    else:
        for delta_t in DELTA_T_SEC_LIST:
            plt.xlim(0, int(delta_t * video_framerate))
            plt.savefig(filename_append(output_path, "{:.1f}s".format(delta_t)))
    plt.close()


if __name__ == "__main__":
    # t1 = np.random.randint(0, 255, size=(20, 20, 1), dtype=np.uint8)
    # t2 = np.random.randint(0, 255, size=(20, 20, 1), dtype=np.uint8)
    # print(similarity(t1, t2))

    # Dummy video for testing, the rotation speed is 2*PI (rad/s)
    # Video length is 15 sec. The object spins for 15 times.
    # dummy_path = "res/rotate-qr-320.avi"
    # plot_s_against_delta(dummy_path, "res/rotate-qr-320-fig0.png")

    files = filter(lambda x: str(x).endswith("crop.avi"), os.listdir("output/#old"))
    for f in files:
        plot_pairwise_similarity_heatmap_compact(f"output/#old/{f}",
                                                 f"output/{filename_append(f, 'pairwise-compact', 'png')}")
