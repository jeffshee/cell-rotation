import math

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

MIN_MATCH_COUNT = 5  # At least 4 corresponding point required to calculate homography
FILTER_MATCH = True
SCALE_FACTOR = 1.0
# https://docs.opencv.org/4.5.0/d7/d60/classcv_1_1SIFT.html
SIFT_PARAMETERS = dict(nfeatures=500, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)


def rot_angle_from_homography(M):
    # Derive rotation angle from homography
    # -180 ~ 180 (deg)
    theta = - math.atan2(M[0, 1], M[0, 0]) * 180 / math.pi
    return theta


class MethodB:
    """
    Alternative method based on feature descriptors and homography
    """

    def __init__(self, template_img: np.ndarray, video_path: str, output_path: str = None):
        self.template_img = template_img
        if SCALE_FACTOR > 1.0:
            self.template_img = scale_img(self.template_img, SCALE_FACTOR)
        self.detector = cv2.SIFT_create(**SIFT_PARAMETERS)
        self.template_kp, self.template_des = self.detector.detectAndCompute(template_img, mask=None)
        self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        self.frame_list = []
        self.get_frame_list(video_path)
        self.video_framerate = get_video_framerate(video_path)
        self.video_length = get_video_length(video_path)

        self.output_path = output_path
        self.video_writer = None

    def get_frame_list(self, video_path: str):
        video_capture = get_video_capture(video_path)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            if SCALE_FACTOR > 1.0:
                frame = scale_img(frame, SCALE_FACTOR)
            self.frame_list.append(frame)

    def sec_to_frame_i(self, sec):
        return int(sec * self.video_framerate)

    def frame_i_to_sec(self, frame_i):
        return int(frame_i / self.video_framerate)

    def calc(self):
        rot_angle_list = []
        for i, frame in enumerate(tqdm(self.frame_list, desc="Processing")):
            frame_kp, frame_des = self.detector.detectAndCompute(frame, mask=None)
            knn_matches = self.matcher.knnMatch(self.template_des, frame_des, k=2)

            # Filter matches using the Lowe's ratio test
            ratio_thresh = 0.7
            if FILTER_MATCH:
                good_matches = []
                for m, n in knn_matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
            else:
                good_matches = [m for m, _ in knn_matches]

            if len(good_matches) > MIN_MATCH_COUNT:
                src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                homography_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if homography_mat is None:
                    print("Homography Mat is None at frame {}".format(i))
                    matches_mask = None
                    rot_angle_list.append(np.nan)
                else:
                    matches_mask = mask.ravel().tolist()
                    if len(self.template_img.shape) == 3:
                        h, w, c = self.template_img.shape
                    else:
                        h, w = self.template_img.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, homography_mat)
                    frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                    rot_angle_list.append(rot_angle_from_homography(homography_mat))
            else:
                print("Not enough matches are found at frame {} - {}/{}, knn_matches {}".format(i, len(good_matches),
                                                                                                MIN_MATCH_COUNT,
                                                                                                len(knn_matches)))
                matches_mask = None
                rot_angle_list.append(np.nan)

            if self.output_path is None:
                continue

            # Output video
            draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                               singlePointColor=None,
                               matchesMask=matches_mask,  # Draw only inliers
                               flags=2)
            frame_out = cv2.drawMatches(self.template_img, self.template_kp, frame, frame_kp, good_matches, None,
                                        **draw_params)

            out_h, out_w, _ = frame_out.shape
            # Draw rotation angle at bottom
            frame_out = draw_label(frame_out, [0, out_h], "Rotation %.2f" % rot_angle_list[-1],
                                   font_scale=0.5)

            if self.video_writer is None:
                self.video_writer = get_video_writer(self.output_path, self.video_framerate, (out_w, out_h))
            self.video_writer.write(frame_out)

        if self.video_writer is not None:
            self.video_writer.release()

        ts = np.linspace(0, self.frame_i_to_sec(len(self.frame_list)), num=len(self.frame_list))
        thetas = np.array(rot_angle_list)
        return ts, thetas


def plot_theta_against_t(ts: np.ndarray, thetas: np.ndarray, output_path=None):
    plt.figure()
    plt.title(f"Theta against t")
    plt.xlabel("t [sec]")
    plt.ylabel("Theta [deg]")
    plt.plot(ts, thetas)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


def plot_angular_speed_against_t(ts: np.ndarray, thetas: np.ndarray, output_path=None):
    plt.figure()
    plt.title(f"Angular speed against t")
    plt.xlabel("t [sec]")
    plt.ylabel("Angular speed [deg/s]")
    # Convert to rad, unwrap
    thetas = np.unwrap(np.deg2rad(thetas))
    angular_speeds = np.gradient(thetas, ts)
    # Convert back to deg
    angular_speeds = np.rad2deg(angular_speeds)
    mean_angular_speeds = np.mean(angular_speeds)
    plt.plot(ts, angular_speeds)
    plt.axhline(y=mean_angular_speeds, c="grey", linestyle="--")
    plt.annotate("Mean: %.2f [deg/s]; %.2f [rad/s]" % (np.mean(angular_speeds), np.deg2rad(np.mean(angular_speeds))),
                 (0, mean_angular_speeds))
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
    print(f"Angular speed: {np.mean(angular_speeds)} [deg/s]")


if __name__ == "__main__":
    template_img = cv2.imread("res/qr-code-320.png")
    method_b = MethodB(template_img, "res/rotate-qr-320.avi", "res/rotate-qr-320-matching.avi")
    ts, thetas = method_b.calc()
    plot_theta_against_t(ts, thetas, "res/rotate-qr-320-fig1.png")
    plot_angular_speed_against_t(ts, thetas, "res/rotate-qr-320-fig2.png")
