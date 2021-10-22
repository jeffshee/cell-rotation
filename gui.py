from utils import *


def gui(video_path: str, cached_roi=None):
    def thresh_callback(thresh, frame):
        frame_copy = frame.copy()
        frame = binarization_img(frame, thresh, cv2.THRESH_BINARY_INV)
        frame = apply_mask_img(frame, draw_mask_roi(roi, width, height))

        cv2.imshow(win_bin, frame)
        contours, hierarchy = cv2.findContours(frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Filter out the contours that unlikely to be a circle
        contours = [contours_i for contours_i in contours if len(contours_i) > 4]
        # Show message if no contours found
        if len(contours) == 0:
            print("No contour found! Please adjust threshold.")
        else:
            # Get all points from contours
            ptr_list = np.concatenate(contours)
            # Find the convex hull object for all points in contours
            hull_list = [cv2.convexHull(ptr_list)]
            frame_copy = cv2.drawContours(frame_copy, contours=hull_list, contourIdx=-1, color=(0, 0, 255),
                                          thickness=-1)
            cv2.imshow(win_cell, frame_copy)

    # Get 1st frame
    video_capture = get_video_capture(video_path)
    video_dimension = get_video_dimension(video_path)
    width, height = video_dimension
    ret, frame = video_capture.read()

    win_src = "Source Image"
    win_x, win_y = 100, 100
    win_offset_x, win_offset_y = 8, 40
    cv2.namedWindow(win_src)
    cv2.moveWindow(win_src, win_x, win_y)

    """
    ROI
    """
    if cached_roi is None:
        # Getting ROI
        roi = cv2.selectROI(win_src, frame)
        if roi == (0, 0, 0, 0):
            # Set entire image as ROI if ROI is not selected
            roi = (0, 0, width, height)
        else:
            # [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = roi
            roi = x, y, x + w, y + h
    else:
        # ROI is cached, skip selecting step
        # Instead, draw the cached ROI onto the image and show
        roi = cached_roi
        frame_copy = frame.copy()
        frame_copy = cv2.rectangle(frame_copy, roi[:2], roi[2:], color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # Draw label on box
        frame_copy = draw_label(frame_copy, roi[:2], "ROI")
        cv2.imshow(win_src, frame_copy)
    print("ROI (x1, y1, x2, y2):", roi)

    """
    Threshold
    """
    win_bin = "Binarization (Preview)"
    win_cell = "Cell Area (Preview)"
    win_thresh = "Threshold"
    cv2.namedWindow(win_bin)
    cv2.moveWindow(win_bin, win_x + width + win_offset_x, win_y)
    cv2.namedWindow(win_cell)
    cv2.moveWindow(win_cell, win_x + (width + win_offset_x) * 2, win_y)
    cv2.namedWindow(win_thresh)
    cv2.moveWindow(win_thresh, win_x, win_y + height + win_offset_y)

    max_thresh = 255
    init_thresh = 180  # Initial threshold
    cv2.createTrackbar('Threshold', win_thresh, init_thresh, max_thresh,
                       lambda val: thresh_callback(val, frame))
    # Initial call
    thresh_callback(init_thresh, frame)
    print("Adjust threshold and then press SPACE or ENTER button!")
    cv2.waitKey()
    thresh = cv2.getTrackbarPos('Threshold', win_thresh)
    cv2.destroyAllWindows()
    print("Threshold", thresh)

    return roi, thresh


if __name__ == "__main__":
    gui("res/rotate-qr-320.avi")
