import base64
import io
import time
import requests
import pandas as pd
from PIL import Image
from requests.exceptions import RequestException, HTTPError

API_URL = "http://localhost:8888/omni"

# --------------------------------------------------------------------------- #
# Robust wrapper around the Hugging Face endpoint
# --------------------------------------------------------------------------- #
def query(payload: dict, retries: int = 5, backoff: float = 10, timeout: int = 30):
    """
    POST the payload to the HF endpoint, retrying automatically on network
    problems or >=500 server errors.

    Parameters
    ----------
    payload : dict
    retries : int        number of additional attempts (0 ⇒ only once)
    backoff : float      seconds * (attempt index + 1) before each retry
    timeout : int        request timeout for each POST, in seconds
    """
    attempt = 0
    while True:
        try:
            response = requests.post(API_URL, json=payload, timeout=timeout)
            response.raise_for_status()                # HTTP 4xx/5xx → exception
            return response.json()

        # Network issues, time-outs, or 5xx responses trigger a retry
        except (RequestException, HTTPError, ConnectionError, TimeoutError) as err:
            return None
# --------------------------------------------------------------------------- #
def omni_process_image(path: str, out_path: str = "annotated.png") -> str:
    """
    Sends an image to the Vision-LLM endpoint, saves the returned annotation,
    and returns a double-spaced string representation of the detected bboxes.
    """

    # ---- encode the input image ------------------------------------------- #
    with open(path, "rb") as f:
        img_bytes = f.read()
    b64_in = base64.b64encode(img_bytes).decode("ascii")

    with Image.open(path) as img:
        w, h = img.size

    payload = {
        "inputs": {
            "image": f"data:image/png;base64,{b64_in}",
            "image_size": {"h": h, "w": w},
        }
    }

    # ---- call the endpoint (auto-retries inside query) -------------------- #
    result = query(payload)
    
    
    if not result or "bboxes" not in result or not result["bboxes"]:
        return None

    # ---- post-process bboxes ---------------------------------------------- #
    bboxes = result["bboxes"]
    for box in bboxes:
        if "bbox" in box:
            x1, y1, x2, y2 = box["bbox"]
            # convert normalised coords → pixels
            if all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
                box["bbox"] = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]

    # ---- save annotated image --------------------------------------------- #
    encoded_out = result["image"]
    out_bytes = base64.b64decode(encoded_out)
    Image.open(io.BytesIO(out_bytes)).save(out_path)
    print(f"Annotated image saved to {out_path}")

    # ---- make double-spaced DataFrame dump -------------------------------- #
    df = (
        pd.DataFrame(bboxes)
          .drop(columns=["type", "interactivity", "source"], errors="ignore")
    )
    double_spaced = "\n\n".join(df.to_string(index=True).splitlines())
    return double_spaced

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    image_path = r"F:/BIM_GUI_Agent/BIM_GUI_Agent/runs/37aec26b-b0b0-422e-a995-fc3180472abc/screenshots/popup_window.png"
    print("Detected bboxes:")
    bbox_report = process_image(image_path)
    print(bbox_report)
