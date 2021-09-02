import os

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from utils import *

TM_METHOD = [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED][1]
SLIDING_WINDOW_SEC = 3.0


class MethodA:
    def __init__(self, video_path: str):
        self.frame_list = []
        self.get_frame_list(video_path)
        self.video_framerate = get_video_framerate(video_path)
        self.video_length = get_video_length(video_path)

    def get_frame_list(self, video_path: str):
        video_capture = get_video_capture(video_path)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            self.frame_list.append(frame)

    def x(self, t):
        return self.frame_list[self.sec_to_frame_i(t)]

    def sec_to_frame_i(self, sec):
        return int(sec * self.video_framerate)

    def frame_i_to_sec(self, frame_i):
        return int(frame_i / self.video_framerate)

    def calc(self, delta, t=SLIDING_WINDOW_SEC / 2):
        if t < SLIDING_WINDOW_SEC / 2 or t > self.frame_i_to_sec(self.video_length) - SLIDING_WINDOW_SEC / 2:
            return np.nan
        taus = np.linspace(t - SLIDING_WINDOW_SEC / 2, t + SLIDING_WINDOW_SEC / 2, num=50)
        return np.mean([similarity(self.x(tau), self.x(tau + delta)) for tau in taus])


def similarity(img1: np.ndarray, img2: np.ndarray):
    return np.squeeze(cv2.matchTemplate(img1, img2, TM_METHOD))


def plot_s_against_delta(video_path: str, output_path=None):
    s = MethodA(video_path)
    delta = np.linspace(0, SLIDING_WINDOW_SEC, num=50)
    y = np.array([s.calc(d) for d in delta])

    # Find peaks
    peak_idx, _ = find_peaks(y)
    peak_delta = delta[peak_idx]
    peak_y = y[peak_idx]

    plt.figure()
    plt.title(f"S against delta \n{(os.path.basename(video_path))}")
    plt.xlabel("delta [sec]")
    plt.ylabel("S")
    plt.plot(delta, y)

    plt.scatter(peak_delta, peak_y, c="red", marker="+")
    for peak_delta_i, peak_y_i in zip(peak_delta, peak_y):
        plt.axvline(peak_delta_i, c="grey", linestyle="--")
        plt.annotate("%.2f" % peak_delta_i, (peak_delta_i, peak_y_i))
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


if __name__ == "__main__":
    # t1 = np.random.randint(0, 255, size=(20, 20, 1), dtype=np.uint8)
    # t2 = np.random.randint(0, 255, size=(20, 20, 1), dtype=np.uint8)
    # print(similarity(t1, t2))

    # Dummy video for testing, the rotation speed is 2*PI (rad/s)
    # Video length is 15 sec. The object spins for 15 times.
    dummy_path = "res/rotate-qr-320.avi"
    plot_s_against_delta(dummy_path, "res/rotate-qr-320-fig0.png")
