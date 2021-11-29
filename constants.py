import os
import json
from pprint import pprint
from argparse import Namespace

if os.path.isfile("config.json"):
    # Read Config
    with open("config.json", "r") as f:
        config = json.load(f)
        print("[Config] Current configuration")
        pprint(config)
else:
    import cv2

    config = dict()
    # Settings
    config["APPLY_GAUSSIAN_BLUR"] = False  # Apply gaussian blur as preprocessing (Remove high-frequency noise)
    config["GAUSSIAN_BLUR_KSIZE"] = 5  # Kernel size of gaussian blur
    config["GAUSSIAN_BLUR_SIGMA"] = 0  # Sigma param of gaussian blur
    config["PREVIEW_SCALE"] = 4  # Scale factor of previews
    config["THRESH_TYPE"] = cv2.THRESH_BINARY_INV  # Base thresh type for the task
    config["FILTER_MIN_CONTOURS_LEN"] = 4
    config["FILTER_RADIUS"] = 15

    # Defaults
    config["DEFAULT_ADAPTIVE_THRESH"] = False
    config["DEFAULT_AUTO_THRESH"] = True
    config["DEFAULT_ADAPTIVE_METHOD"] = cv2.ADAPTIVE_THRESH_MEAN_C
    config["DEFAULT_THRESHOLD"] = 180  # cv2.threshold
    config["DEFAULT_BLOCK_SIZE"] = 7  # cv2.adaptiveThreshold
    config["DEFAULT_C"] = 15  # cv2.adaptiveThreshold
    # Which method to use for template matching, default is TM_CCOEFF_NORMED.
    config["TM_METHOD"] = [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED][1]

    # How many frames (DELTA_T_SEC_MAX * fps) to process.
    # The computational complexity is O((DELTA_T_SEC_MAX * fps) * N)
    # where N is the length of the video.
    config["DELTA_T_SEC_LIST"] = [1.0, 2.0, 3.0]
    config["DELTA_T_SEC_MAX"] = max(config["DELTA_T_SEC_LIST"])

    # Write Default config
    with open("config.json", "w") as f:
        print(config)
        json.dump(config, f, indent=3)

# Append config to namespace
constants = Namespace(**config)
