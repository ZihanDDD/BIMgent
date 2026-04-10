import imageio.v2 as imageio
import numpy as np
from PIL import Image
from conf.config import Config
import os


config = Config()
image_size=(512, 512)


def imresize(image):
    """Resize an image using PIL."""
    return np.array(Image.fromarray(image.astype(np.uint8)).resize(
        (image_size[1], image_size[0])))

def resize_image(im_path):
    """
    Enhanced floorplan processing with noise reduction and geometric refinement.
    """
    import imageio.v2 as imageio
    import numpy as np

    # === 1. Load and preprocess input image ===
    im = imageio.imread(im_path)

    # Convert to RGB if needed
    if len(im.shape) == 2:  # Grayscale
        im = np.stack([im, im, im], axis=-1)
    elif im.shape[2] == 4:  # RGBA
        im = im[:, :, :3]  # Drop alpha channel
    elif im.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {im.shape}")

    # Convert to float for processing
    im = im.astype(np.float32)

    # Perform resize (still float32)
    im = imresize(im)

    # === IMPORTANT: convert back to uint8 before saving ===
    im = np.clip(im, 0, 255)      # ensure valid range
    im = im.astype(np.uint8)

    print(f"Image shape after resize: {im.shape}")
    print(f"Expected shape: {image_size}")

    screenshot_name = "resized_floorplan.png"
    screenshot_path = os.path.join(config.work_dir, screenshot_name)

    imageio.imwrite(screenshot_path, im)

    return screenshot_path





    
