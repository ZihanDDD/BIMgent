import os

import cv2
import numpy as np
import pyautogui


class ScreenshotsProcessor:

    def screenshot_capture(self, dir_path, screenshot_name):
        """Take a full-screen screenshot and save it as ``dir_path/screenshot_name``."""
        os.makedirs(dir_path, exist_ok=True)

        if not screenshot_name.lower().endswith(".png"):
            screenshot_name += ".png"

        full_path = os.path.join(dir_path, screenshot_name)

        try:
            pyautogui.screenshot().save(full_path)
            return full_path
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            return None

    def drawing_panel(self, image_path, x, y, w, h):
        """Write a copy of ``image_path`` where everything outside the
        ``(x, y, w, h)`` box is painted white. Returns the output path."""
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read the image at {image_path}")
            return None

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        height, width, _ = image.shape

        # Clamp bounding box to image edges
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        masked_image = np.full_like(image, 255, dtype=np.uint8)
        masked_image[y:y+h, x:x+w] = image[y:y+h, x:x+w]

        base_dir, filename = os.path.split(image_path)
        output_path = os.path.join(base_dir, "masked_" + filename)
        cv2.imwrite(output_path, masked_image)

        return output_path

    def extract_popup(self, prev_img_path: str,
                      curr_img_path: str,
                      out_img_path: str = "popup_only.png",
                      diff_thresh: int = 25):
        """
        Finds the largest changed region between two screenshots and writes an
        image that shows only that region (everything else is white).

        Parameters
        ----------
        prev_img_path : str
            Path to the *previous* screenshot.
        curr_img_path : str
            Path to the *current* screenshot (the one with the pop-up).
        out_img_path : str, optional
            Where to save the result (PNG).  Default is 'popup_only.png'
        diff_thresh : int, optional
            Pixel-difference threshold used to build the change mask.
            Lower values make the detector more sensitive.

        Returns
        -------
        tuple[int, int, int, int]
            ``(x, y, w, h)`` of the changed region. Falls back to the full
            1920x1080 screen when the images cannot be read or nothing changed.
        """
        before = cv2.imread(prev_img_path)
        after = cv2.imread(curr_img_path)
        if before is None or after is None:
            print("Could not read one of the screenshots.")
            return 0, 0, 1920, 1080

        gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        abs_diff = cv2.absdiff(gray_before, gray_after)
        _, diff_mask = cv2.threshold(abs_diff, diff_thresh, 255, cv2.THRESH_BINARY)

        # tidy the mask (close tiny holes, join nearby blobs)
        kernel = np.ones((5, 5), np.uint8)
        diff_mask = cv2.dilate(diff_mask, kernel, 2)
        diff_mask = cv2.erode(diff_mask, kernel, 1)

        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 1920, 1080

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        result = np.full_like(after, 255)
        result[y:y+h, x:x+w] = after[y:y+h, x:x+w]

        cv2.imwrite(out_img_path, result)
        print(f"Saved: {out_img_path}")

        return x, y, w, h
