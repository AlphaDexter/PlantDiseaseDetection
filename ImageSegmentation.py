import numpy as np
import cv2

# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

# error message when image is not colored while it should be
NOT_COLOR_IMAGE = 'NOT_COLOR_IMAGE'

def read_image(file_path, read_mode=cv2.IMREAD_COLOR):
    """
    Read image file with all preprocessing needed
    Args:
        file_path: absolute file_path of an image file
        read_mode: whether image reading mode is rgb, grayscale or somethin
    Returns:
        np.ndarray of the read image or None if couldn't read
    Raises:
        ValueError if image could not be read with message IMAGE_NOT_READ
    """
    # read image file in grayscale
    image = cv2.imread(file_path, read_mode)

    if image is None:
        raise ValueError(IMAGE_NOT_READ)
    else:
        return image

