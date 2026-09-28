import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from conf.config import Config


config = Config()
image_size = (512, 512)


def imresize(image):
    """Resize an image using PIL."""
    return np.array(Image.fromarray(image.astype(np.uint8)).resize(
        (image_size[1], image_size[0])))


def resize_image(im_path):
    """Resize the input floorplan to 512x512 RGB and save it into the run dir."""
    im = imageio.imread(im_path)

    if len(im.shape) == 2:  # Grayscale
        im = np.stack([im, im, im], axis=-1)
    elif im.shape[2] == 4:  # RGBA
        im = im[:, :, :3]
    elif im.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {im.shape}")

    im = imresize(im.astype(np.float32))
    im = np.clip(im, 0, 255).astype(np.uint8)

    print(f"Image shape after resize: {im.shape}")

    screenshot_path = os.path.join(config.work_dir, "resized_floorplan.png")
    imageio.imwrite(screenshot_path, im)

    return screenshot_path
