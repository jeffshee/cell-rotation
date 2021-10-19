from utils import *


class GUI:
    def __init__(self, video_path: str, roi=None, thresh_type=cv2.THRESH_BINARY_INV,
                 adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, auto_thresh=False, adaptive_thresh=True):
        if auto_thresh and adaptive_thresh:
            raise RuntimeError("Both `auto_threshold` and `adaptive_threshold` can't be `True` at the same time")

        self.video_path = video_path
        self.roi = roi
        self.thresh_type = thresh_type
        self.adaptive_method = adaptive_method
        self.auto_thresh = auto_thresh
        self.adaptive_thresh = adaptive_thresh

        video_capture = get_video_capture(video_path)
        video_dimension = get_video_dimension(video_path)
        width, height = video_dimension
        _, self.frame = video_capture.read()
        self.frame_gray = grayscale_img(self.frame)

        """
        Params (default values)
        """
        self.threshold = 180  # cv2.threshold
        self.block_size = 7  # cv2.adaptiveThreshold
        self.C = 15  # cv2.adaptiveThreshold

        """
        Source Window
        """
        self.win_src = "Source Image"
        win_x, win_y = 100, 100
        win_offset_x, win_offset_y = 8, 40
        cv2.namedWindow(self.win_src)
        cv2.moveWindow(self.win_src, win_x, win_y)

        """
        ROI
        """
        if self.roi:
            frame_copy = self.frame.copy()
            frame_copy = cv2.rectangle(frame_copy, self.roi[:2], self.roi[2:], color=(255, 0, 0), thickness=1,
                                       lineType=cv2.LINE_AA)
            # Draw label on box
            frame_copy = draw_label(frame_copy, self.roi[:2], "ROI")
            cv2.imshow(self.win_src, frame_copy)
        else:
            self.roi = cv2.selectROI(self.win_src, self.frame)
            if self.roi == (0, 0, 0, 0):
                # Set entire image as ROI if ROI is not selected
                self.roi = (0, 0, width, height)
            else:
                # [x, y, w, h] -> [x1, y1, x2, y2]
                x, y, w, h = self.roi
                self.roi = x, y, x + w, y + h
        print("ROI (x1, y1, x2, y2):", self.roi)

        # Crop according ROI
        self.frame = crop_img_roi(self.frame, self.roi)
        self.frame_gray = crop_img_roi(self.frame_gray, self.roi)
        # Apply GaussianBlur to eliminate noises
        self.frame_gray_blur = cv2.GaussianBlur(self.frame_gray, (5, 5), 0)

        """
        Preview Windows, Params Window
        """
        self.win_bin = "Binarization (Preview)"
        self.win_cell = "Cell Area (Preview)"
        self.win_param = "Parameters"
        cv2.namedWindow(self.win_bin)
        cv2.moveWindow(self.win_bin, win_x + width + win_offset_x, win_y)
        cv2.namedWindow(self.win_cell)
        cv2.moveWindow(self.win_cell, win_x + (width + win_offset_x) * 2, win_y)
        cv2.namedWindow(self.win_param)
        cv2.moveWindow(self.win_param, win_x, win_y + height + win_offset_y)

        """
        Trackbars
        """
        if adaptive_thresh:
            cv2.createTrackbar('Block Size (val*2+3), val=', self.win_param, (self.block_size - 3) // 2, 10,
                               self._on_change_block_size)
            cv2.createTrackbar('C=', self.win_param, self.C, 50, self._on_change_c)
        elif not auto_thresh:
            cv2.createTrackbar('Threshold=', self.win_param, self.threshold, 255, self._on_change_threshold)

        # Initial call
        self.reload_preview()
        print("Adjust parameters and then press SPACE or ENTER button!")
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_result(self):
        if self.adaptive_thresh:
            ret = dict(roi=self.roi, block_size=self.block_size, c=self.C)
        else:
            ret = dict(roi=self.roi, threshold=self.threshold)
        print(ret)
        return ret

    def _on_change_block_size(self, val):
        self.block_size = val * 2 + 3
        self.reload_preview()

    def _on_change_c(self, val):
        self.C = val
        self.reload_preview()

    def _on_change_threshold(self, val):
        self.threshold = val
        self.reload_preview()

    def reload_preview(self):
        frame_copy = self.frame.copy()
        if self.adaptive_thresh:
            frame_bin = cv2.adaptiveThreshold(self.frame_gray_blur, 255, self.adaptive_method, self.thresh_type,
                                              self.block_size, self.C)
        elif self.auto_thresh:
            self.threshold, frame_bin = cv2.threshold(self.frame_gray_blur, 0, 255, self.thresh_type | cv2.THRESH_OTSU)
        else:
            _, frame_bin = cv2.threshold(self.frame_gray_blur, self.threshold, 255, self.thresh_type)
        cv2.imshow(self.win_bin, frame_bin)
        contours, hierarchy = cv2.findContours(frame_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        # Filter out the contours that unlikely to be a circle
        contours = [contours_i for contours_i in contours if len(contours_i) > 4]
        # Show message if no contours found
        if len(contours) == 0:
            print("No contour found! Please adjust parameters.")
        else:
            # Get all points from contours
            ptr_list = np.concatenate(contours)
            # Find the convex hull object for all points in contours
            hull_list = [cv2.convexHull(ptr_list)]
            frame_contours = cv2.drawContours(frame_copy, contours=hull_list, contourIdx=-1, color=(0, 0, 255),
                                              thickness=-1)
            cv2.imshow(self.win_cell, frame_contours)


if __name__ == "__main__":
    import os
    dataset_root = "dataset/new"
    file_list = [os.path.join(dataset_root, f) for f in os.listdir(dataset_root)]
    file_list = list(filter(lambda file_path: os.path.isfile(file_path), file_list))
    for file_path in file_list:
        gui = GUI(file_path)
        print(gui.get_result())

    # gui("res/rotate-qr-320.avi")
