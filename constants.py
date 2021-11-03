import cv2

# Settings
APPLY_GAUSSIAN_BLUR = False  # Apply gaussian blur as preprocessing (Remove high-frequency noise)
GAUSSIAN_BLUR_KSIZE = 5  # Kernel size of gaussian blur
GAUSSIAN_BLUR_SIGMA = 0  # Sigma param of gaussian blur
PREVIEW_SCALE = 4  # Scale factor of previews
THRESH_TYPE = cv2.THRESH_BINARY_INV  # Base thresh type for the task
FILTER_MIN_CONTOURS_LEN = 4
FILTER_RADIUS = 15

# Defaults
DEFAULT_ADAPTIVE_THRESH = False
DEFAULT_AUTO_THRESH = True
DEFAULT_ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_MEAN_C
DEFAULT_THRESHOLD = 180  # cv2.threshold
DEFAULT_BLOCK_SIZE = 7  # cv2.adaptiveThreshold
DEFAULT_C = 15  # cv2.adaptiveThreshold
# Which method to use for template matching, default is TM_CCOEFF_NORMED.
TM_METHOD = [cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED][1]

# How many frames (DELTA_T_SEC_MAX * fps) to process.
# The computational complexity is O((DELTA_T_SEC_MAX * fps) * N)
# where N is the length of the video.
DELTA_T_SEC_LIST = [1.0, 2.0, 3.0]
DELTA_T_SEC_MAX = max(DELTA_T_SEC_LIST)
