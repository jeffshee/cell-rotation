import math

from scipy import ndimage
from tqdm import tqdm

from v1.utils import *


def generate(output_path: str):
    # Output
    width, height = 320, 180
    video_framerate = 30.0
    length = 15 * 30  # 15 sec * 30*fps
    video_writer = get_video_writer(output_path, video_framerate, (width, height))
    angular_speed = 360.0 / video_framerate  # Spin once per sec

    # QR code image
    img_qr = cv2.imread("../res/qr-code-320.png")
    img_qr_size = int(math.sqrt(0.5 * height ** 2)) - 40  # Maximum length of edge - offset
    img_qr = cv2.resize(img_qr, (img_qr_size, img_qr_size))
    bar = tqdm(total=length, desc=f"Generating {output_path}")

    for frame_idx in range(length):
        # Progress
        bar.update(1)
        bar.refresh()

        rotated = ndimage.rotate(img_qr, frame_idx * angular_speed * -1)  # -1 is for clockwise rotation
        # Add padding to rotated image
        h, w, _ = rotated.shape
        top, left = (height - h) // 2, (width - w) // 2
        bottom, right = height - top - h, width - left - w
        padded = cv2.copyMakeBorder(rotated, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        video_writer.write(padded)
    video_writer.release()


if __name__ == "__main__":
    generate("../res/rotate-qr-320.avi")
