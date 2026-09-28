import time
import base64
import io
import torch
import pandas as pd
from PIL import Image
from conf.config import Config
from BIMgent.utils.dict_utils import kget
from BIMgent.provider.omni_provider.util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model
)

config = Config()

model_path = kget(config.env_config, "models_path", default='')['omini']
model_name_or_path_Florence2 = kget(config.env_config, "models_path", default='')['Florence2']



class OmniProvider:
    def __init__(self):
        # Initialize device and load models once
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.BOX_THRESHOLD = 0.05
        self.som_model = self._load_som_model()
        self.caption_model_processor = self._load_caption_model()
        print(f'Initialized OmniProvider on {self.device}')

    def _load_som_model(self):
        """Load and initialize the YOLO model."""
        model = get_yolo_model(model_path)
        model.to(self.device)
        print(f'SOM model loaded to {self.device}')
        return model

    def _load_caption_model(self):
        """Load and initialize the caption model."""
        caption_model = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=model_name_or_path_Florence2,
            device=self.device
        )
        print("Florence2 model loaded successfully")
        return caption_model

    def process_image(self, image_path, out_path=None):
        """Process an image and return a metadata string.

        Interface matches ``omni_process_image(path, out_path)`` from the
        endpoint module so callers can swap freely.

        Returns
        -------
        str or None
            Double-spaced DataFrame string of detected bboxes, or *None* when
            nothing is detected.
        """
        image = Image.open(image_path)
        image_rgb = image.convert('RGB')
        image_width, image_height = image.size

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # Perform OCR
        start = time.time()
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        ocr_time = time.time() - start
        print(f"OCR completed in {ocr_time:.2f} seconds")

        # Perform SOM detection + labeling
        encoded_image, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path,
            self.som_model,
            BOX_TRESHOLD=self.BOX_THRESHOLD,
            output_coord_in_ratio=False,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128
        )
        caption_time = time.time() - start - ocr_time
        print(f"Caption generation completed in {caption_time:.2f} seconds")

        if not parsed_content_list:
            return None

        # Convert normalised coords to pixels if needed
        for item in parsed_content_list:
            if 'bbox' in item:
                x1, y1, x2, y2 = item['bbox']
                if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                    item['bbox'] = [
                        int(x1 * image_width),
                        int(y1 * image_height),
                        int(x2 * image_width),
                        int(y2 * image_height)
                    ]

        # Save annotated image
        if out_path and encoded_image:
            out_bytes = base64.b64decode(encoded_image)
            Image.open(io.BytesIO(out_bytes)).save(out_path)
            print(f"Annotated image saved to {out_path}")

        # Compute click targets for every bbox
        for item in parsed_content_list:
            if 'bbox' not in item:
                continue
            x1, y1, x2, y2 = item['bbox']
            item['click_x'] = x1 + (x2 - x1) * 3 // 4
            item['click_y'] = (y1 + y2) // 2

        # Build double-spaced DataFrame string
        df = (
            pd.DataFrame(parsed_content_list)
              .drop(columns=["type", "interactivity", "source"], errors="ignore")
        )
        double_spaced_str = "\n\n".join(df.to_string(index=True).splitlines())

        return double_spaced_str