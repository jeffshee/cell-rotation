from multiprocessing import Process

from calc_a import *
from calc_b import *
from gui import GUI

# NOTE: Method B is not usable for cell video yet.
METHOD = "a"
# Skip loading cached ROI from roi.pkl.
IGNORE_CACHED_ROI = False


def binarization_video(video_path: str, output_path: str, threshold=127, auto_thresh=False):
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
        mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU if auto_thresh else cv2.THRESH_BINARY_INV
        if auto_thresh:
            threshold = 0
        frame = binarization_img(frame, threshold, mode=mode)
        video_writer.write(bgr_img(frame))  # Convert back to bgr format before writing frame

    video_capture.release()
    video_writer.release()


def cell_detect_video(video_path: str, output_path: str, gui_ret, thresh_type=cv2.THRESH_BINARY_INV,
                      adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, auto_thresh=False, adaptive_thresh=True):
    # Input/Output
    video_capture = get_video_capture(video_path)
    video_length = get_video_length(video_path)
    video_dimension = get_video_dimension(video_path)
    video_framerate = get_video_framerate(video_path)
    video_writer = get_video_writer(output_path, video_framerate, video_dimension)
    # width, height = video_dimension

    bar = tqdm(total=video_length, desc=f"Cell Detect {video_path}")
    set_frame_position(video_capture, 0)
    while video_capture.isOpened() and get_frame_position(video_capture) in range(video_length):
        # Progress
        bar.update(1)
        bar.refresh()

        ret, frame = video_capture.read()
        if not ret:
            break
        # Crop according ROI
        frame = crop_img_roi(frame, gui_ret["roi"])
        frame_gray = grayscale_img(frame)
        # Apply GaussianBlur to eliminate noises
        frame_gray_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        if adaptive_thresh:
            frame_bin = cv2.adaptiveThreshold(frame_gray_blur, 255, adaptive_method, thresh_type,
                                              gui_ret["block_size"], gui_ret["c"])
        elif auto_thresh:
            threshold, frame_bin = cv2.threshold(frame_gray_blur, 0, 255, thresh_type | cv2.THRESH_OTSU)
        else:
            _, frame_bin = cv2.threshold(frame_gray_blur, gui_ret["threshold"], 255, thresh_type)

        frame_copy = frame.copy()

        # frame = apply_mask_img(frame, draw_mask_roi(gui_ret["roi"], width, height))
        contours, hierarchy = cv2.findContours(frame_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Find the convex hull object for each contour
        # hull_list = []
        # for i in range(len(contours)):
        #     hull = cv2.convexHull(contours[i])
        #     hull_list.append(hull)

        # Filter out the contours that unlikely to be a circle
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
        # frame_copy = cv2.rectangle(frame_copy, roi[:2], roi[2:], color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # # Draw label on box
        # frame_copy = draw_label(frame_copy, roi[:2], "ROI")
        video_writer.write(frame_copy)

    video_capture.release()
    video_writer.release()


def cell_crop_video(video_path: str, output_path: str, gui_ret, thresh_type=cv2.THRESH_BINARY_INV,
                    adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C, auto_thresh=False, adaptive_thresh=True):
    # Input/Output
    video_capture = get_video_capture(video_path)
    video_length = get_video_length(video_path)
    # width, height = get_video_dimension(video_path)
    x1, y1, x2, y2 = gui_ret["roi"]
    width, height = x2 - x1, y2 - y1
    video_framerate = get_video_framerate(video_path)

    bar = tqdm(total=video_length, desc=f"Cell Detect {video_path}")
    set_frame_position(video_capture, 0)
    bounding_rects = []
    masked_frames = []
    output_frames = []
    while video_capture.isOpened() and get_frame_position(video_capture) in range(video_length):
        # Progress
        bar.update(1)
        bar.refresh()

        ret, frame = video_capture.read()
        if not ret:
            break
        # Crop according ROI
        frame = crop_img_roi(frame, gui_ret["roi"])
        frame_gray = grayscale_img(frame)
        # Apply GaussianBlur to eliminate noises
        frame_gray_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        if adaptive_thresh:
            frame_bin = cv2.adaptiveThreshold(frame_gray_blur, 255, adaptive_method, thresh_type,
                                              gui_ret["block_size"], gui_ret["c"])
        elif auto_thresh:
            threshold, frame_bin = cv2.threshold(frame_gray_blur, 0, 255, thresh_type | cv2.THRESH_OTSU)
        else:
            _, frame_bin = cv2.threshold(frame_gray_blur, gui_ret["threshold"], 255, thresh_type)

        frame_copy = frame.copy()

        # frame = apply_mask_img(frame, draw_mask_roi(roi, width, height))

        contours, hierarchy = cv2.findContours(frame_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

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
        output_frames.append(frame_cropped)
    video_writer.release()
    return output_frames


def cell_template_image(video_path: str, threshold=127, auto_threshold=True, roi=None, ret_binarization=False):
    # Input
    video_capture = get_video_capture(video_path)
    width, height = get_video_dimension(video_path)

    if roi is None:
        roi_ratio = (0.33, 0.33, 0.66, 0.66)  # Ratio on width and height. x1,y1,x2,y2.
        roi = (int(roi_ratio[0] * width), int(roi_ratio[1] * height)),
        (int(roi_ratio[2] * width), int(roi_ratio[3] * height))

    set_frame_position(video_capture, 0)
    ret, frame = video_capture.read()
    frame_copy = frame.copy()
    mode = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU if auto_threshold else cv2.THRESH_BINARY_INV
    if auto_threshold:
        threshold = 0
    frame = binarization_img(frame, threshold, mode)
    frame = apply_mask_img(frame, draw_mask_roi(roi, width, height))

    contours, hierarchy = cv2.findContours(frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Filter out the contours that unlikely to be a circle
    # contours = [contours_i for contours_i in contours if len(contours_i) > 4]
    # Get all points from contours
    ptr_list = np.concatenate(contours)
    # Find the convex hull object for all points in contours
    hull_list = [cv2.convexHull(ptr_list)]
    # Get boundingRect, slightly expand
    bounding_rect = expand_rect(cv2.boundingRect(hull_list[0]), pixel=2)
    return crop_img_rect(frame if ret_binarization else frame_copy, bounding_rect)


if __name__ == "__main__":
    import pickle

    dataset_root = "dataset"
    output_root = "output"
    file_list = [os.path.join(dataset_root, f) for f in os.listdir(dataset_root)]
    file_list = list(filter(lambda file_path: os.path.isfile(file_path), file_list))
    roi_cache_path = "roi.pkl"
    post_processing = []

    if os.path.isfile(roi_cache_path) and not IGNORE_CACHED_ROI:
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
        # output_path_contr = os.path.join(output_root, filename_append(f_basename, "contr"))
        output_path_bin = os.path.join(output_root, filename_append(f_basename, "bin"))

        if METHOD == "a":
            output_path_final = os.path.join(output_root, filename_append(f_basename, "crop"))
        elif METHOD == "b":
            output_path_final = os.path.join(output_root, filename_append(f_basename, "matching"))
        else:
            output_path_final = os.path.join(output_root, filename_append(f_basename, "output"))

        output_path_fig = filename_append(output_path_final, "fig", "png")
        gui = GUI(video_path=f, roi=roi_cache.get(f, None))
        ret = gui.get_result()
        with open(roi_cache_path, "wb") as pkl:
            roi_cache[f] = ret["roi"]
            pickle.dump(roi_cache, pkl)
        # cell_detect_video(f, output_path=output_path, threshold=threshold, roi=roi)

        if METHOD == "a":
            # Method A
            frame_list = cell_crop_video(f, output_path=output_path_final, gui_ret=ret)
            post_processing.append(Process(target=plot_pairwise_similarity_heatmap_compact,
                                           kwargs=dict(video_path=output_path_final,
                                                       frame_list=frame_list,
                                                       output_path=filename_append(output_path_fig, "a"))))
            # post_processing.append(Process(target=plot_pairwise_similarity_heatmap_compact,
            #                                kwargs=dict(video_path=output_path_final,
            #                                            output_path=filename_append(output_path_fig, "a"))))
            # plot_pairwise_similarity_heatmap_compact(video_path=output_path_final,
            #                                          output_path=filename_append(output_path_fig, "a"))
            # plot_s_against_delta(video_path=output_path_final, output_path=filename_append(output_path_fig, "a"))
        elif METHOD == "b":
            # Method B
            pass
            # img_template = cell_template_image(f, threshold=threshold, roi=roi)
            # method_b = MethodB(img_template, video_path=f, output_path=output_path_final)
            # ts, thetas = method_b.calc()
            # plot_theta_against_t(ts, thetas, output_path=filename_append(output_path_fig, "b1"))
            # plot_angular_speed_against_t(ts, thetas, output_path=filename_append(output_path_fig, "b2"))

            # Method B (+binarization)
            # img_template = cell_template_image(f, threshold=threshold, roi=roi, ret_binarization=True)
            # binarization_video(f, output_path=output_path_bin, threshold=threshold)
            # method_b = MethodB(img_template, video_path=output_path_bin, output_path=output_path_final)
            # ts, thetas = method_b.calc()
            # plot_theta_against_t(ts, thetas, output_path=filename_append(output_path_fig, "b1"))
            # plot_angular_speed_against_t(ts, thetas, output_path=filename_append(output_path_fig, "b2"))

    # Start post-processing
    print("Start processing. This might take a while...")
    if post_processing:
        for p in post_processing:
            p.start()
            p.join()
    print("Done.")
