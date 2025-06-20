import base64
import hashlib
import pyautogui
import os
import time
import io
import cv2
import numpy as np
from typing import Any, List
from PIL import Image
import pathlib


class ScreenshotsProcessor:
    def __init__(self):
        pass

    def screenshot_capture(self, dir_path, screenshot_name):
        # Ensure dir_path exists
        os.makedirs(dir_path, exist_ok=True)

        # Validate or append .png extension
        if not screenshot_name.lower().endswith(".png"):
            screenshot_name += ".png"

        # # add a timestamp so we don't overwrite older screenshots
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # screenshot_name = f"{os.path.splitext(screenshot_name)[0]}_{timestamp}.png"

        # Build the full file path
        full_path = os.path.join(dir_path, screenshot_name)

        try:
            # Take the screenshot
            screenshot = pyautogui.screenshot()
            # Save the screenshot
            screenshot.save(full_path)
            return full_path
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            return None
    def capture_region(self, dir_path, screenshot_name, x, y, width, height):
        # Ensure dir_path exists
        os.makedirs(dir_path, exist_ok=True)

        # Validate or append .png extension
        if not screenshot_name.lower().endswith(".png"):
            screenshot_name += ".png"

        # # Add timestamp
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # screenshot_name = f"{os.path.splitext(screenshot_name)[0]}_{timestamp}.png"

        # Full save path
        full_path = os.path.join(dir_path, screenshot_name)

        try:
            # Capture only the defined region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            screenshot.save(full_path)
            return full_path
        except Exception as e:
            print(f"Failed to save region screenshot: {e}")
            return None
        
    def drawing_panel(self, image_path, x, y, w, h):
        # Read the imageWWW
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read the image at {image_path}")
            return None

        # Ensure the image has 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Original dimensions
        height, width, channels = image.shape

        # Clamp bounding box to image edges
        x = max(0, x);     y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        # Create a full white mask (same size as original image)
        masked_image = np.full_like(image, 255, dtype=np.uint8)

        # Copy ROI from the original to the mask
        masked_image[y:y+h, x:x+w] = image[y:y+h, x:x+w]

        # Construct output path
        base_dir, filename = os.path.split(image_path)
        
        output_path = os.path.join(base_dir, "masked_" + filename)

        # Save the masked image
        cv2.imwrite(output_path, masked_image)

        return output_path
    
    def extract_popup(self, prev_img_path: str,
                    curr_img_path: str,
                    out_img_path: str = "popup_only.png",
                    diff_thresh: int = 25) -> pathlib.Path:
        """
        Finds the largest changed region between two screenshots and writes an
        image that shows only that region (everything else is white).

        Parameters
        ----------
        prev_img_path : str
            Path to the *previous* screenshot.
        curr_img_path : str
            Path to the *current* screenshot (the one with the pop‑up).
        out_img_path : str, optional
            Where to save the result (PNG).  Default is 'popup_only.png'
        diff_thresh : int, optional
            Pixel‑difference threshold used to build the change mask.
            Lower values make the detector more sensitive.

        Returns
        -------
        pathlib.Path
            Path to the saved output image.
        """
        # -- read the two images --------------------------------------------------
        before = cv2.imread(prev_img_path)
        after  = cv2.imread(curr_img_path)
        if before is None or after is None:
            raise FileNotFoundError("Could not read one of the screenshots.")

        # -- build a binary mask of all changed pixels ----------------------------
        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after  = cv2.cvtColor(after,  cv2.COLOR_BGR2GRAY)

        abs_diff = cv2.absdiff(gray_before, gray_after)
        _, diff_mask = cv2.threshold(abs_diff, diff_thresh, 255, cv2.THRESH_BINARY)

        # tidy the mask (close tiny holes, join nearby blobs)
        kernel = np.ones((5, 5), np.uint8)
        diff_mask = cv2.dilate(diff_mask, kernel, 2)
        diff_mask = cv2.erode(diff_mask,  kernel, 1)

        # -- find the largest contiguous changed area (= the pop‑up) --------------
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 1920, 1080

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # -- create the output ----------------------------------------------------
        result = np.full_like(after, 255)        # start from a white canvas
        result[y:y+h, x:x+w] = after[y:y+h, x:x+w]

        cv2.imwrite(out_img_path, result)
        print(f"Saved: {out_img_path}")

        return x, y, w, h

            
def hash_text_sha256(text: str) -> str:
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()

        
def encode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64encode(payload).decode('utf-8')


def decode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64decode(payload)


def encode_image_path(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = encode_image_binary(image_file.read(), image_path)
        return encoded_image


def encode_image_binary(image_binary, image_path=None):
    encoded_image = encode_base64(image_binary)
    if image_path is None:
        image_path = '<$bin_placeholder$>'
    return encoded_image


def decode_image(base64_encoded_image):
    return decode_base64(base64_encoded_image)


def get_project_root():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.dirname(path) # get to parent, outside of project code path"
    return path

def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path
    
def encode_data_to_base64_path(data: Any) -> List[str]:
    encoded_images = []

    # Handle single input cases
    if isinstance(data, (str, Image.Image, np.ndarray, bytes)):
        data = [data]

    for item in data:
        if isinstance(item, str):
            # Check if the path exists
            if os.path.exists(assemble_project_path(item)):
                path = assemble_project_path(item)
                encoded_image = encode_image_path(path)
                image_type = path.split(".")[-1].lower()
                encoded_image = f"data:image/{image_type};base64,{encoded_image}"
                encoded_images.append(encoded_image)
            else:
                encoded_images.append(item)
            continue

        buffered = None  # Initialize buffered to avoid reference issues

        if isinstance(item, bytes):  # For raw bytes (e.g., mss grab bytes)
            image = Image.frombytes('RGB', item.size, item.bgra, 'raw', 'BGRX')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
        elif isinstance(item, Image.Image):  # PIL image
            buffered = io.BytesIO()
            item.save(buffered, format="JPEG")
        elif isinstance(item, np.ndarray):  # OpenCV image array
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = Image.fromarray(item)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
        elif item is None:
            print("Trying to encode None image! Skipping it.")
            continue

        # Encode if buffered is set
        if buffered:
            encoded_image = encode_image_binary(buffered.getvalue())
            encoded_image = f"data:image/jpeg;base64,{encoded_image}"
            encoded_images.append(encoded_image)

    return encoded_images

