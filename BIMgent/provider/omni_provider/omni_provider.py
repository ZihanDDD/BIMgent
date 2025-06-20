import time
import torch
import pandas as pd
from PIL import Image
from conf.config import Config
from copy import deepcopy
from ultralytics import YOLO
from BIMgent.provider.loop_providers.template_processor import extract_keys_from_template, check_input_keys, check_output_keys, parse_semi_formatted_text
from BIMgent.utils.dict_utils import kget

from BIMgent.memory.local_memory import LocalMemory
from BIMgent.provider.omni_provider.util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor,
    get_yolo_model
)

# If you still need Config as a global, leave this line:
config = Config()
memory = LocalMemory()

model_path = kget(config.env_config, "models_path", default='')['omini']
model_name_or_path_Florence2 =  kget(config.env_config, "models_path", default='')['Florence2']



class OmniProvider:
    def __init__(self):
        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __call__(self, image_path):
        self.image_path = image_path
        
        # Initialize models
        self.som_model = self._load_som_model()
        self.caption_model_processor = self._load_caption_model()
        
        # Configuration for bounding boxes
        self.BOX_THRESHOLD = 0.05
        self.image = Image.open(self.image_path)
        self.image_rgb = self.image.convert('RGB')
        self.image_width, self.image_height = self.image.size
        
        # Drawing configuration (for bounding boxes, text, etc.)
        box_overlay_ratio = max(self.image.size) / 3200
        self.draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        print(f'Initialized OmniProvider with image size: {self.image.size}')

    
    def _load_som_model(self):
        """Load and initialize the YOLO model."""
        model = get_yolo_model(model_path)
        model.to(self.device)
        print(f'SOM model loaded to {self.device}')
        return model
    
    def _load_caption_model(self):
        """Load and initialize the caption model."""
        try:
            # Attempt to load a Florence2 model
            caption_model = get_caption_model_processor(
                model_name="florence2", 
                model_name_or_path=model_name_or_path_Florence2, 
                device=self.device
            )
            print("Florence2 model loaded successfully")
            return caption_model
        except Exception as e:
            print(f"Error loading Florence2 model: {e}")
            # Fallback to BLIP2 if Florence2 fails
            try:
                caption_model = get_caption_model_processor(
                    model_name="blip2", 
                    model_name_or_path="F:/BIM_GUI_Agent/BIM_GUI_Agent/bim_gui_agent/provider/omni_provider/weights/weights/blip2", 
                    device=self.device
                )
                print("Fallback to BLIP2 model loaded successfully")
                return caption_model
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                raise RuntimeError("Failed to load any caption model")
    
    def process_image(self):
        """Process the image and return parsed content with absolute pixel coordinates."""
        # Perform OCR
        start = time.time()
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            self.image_path, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=True
        )
        text, ocr_bbox = ocr_bbox_rslt
        ocr_time = time.time() - start
        print(f"OCR completed in {ocr_time:.2f} seconds")
        
        # Perform SOM detection + labeling, explicitly requesting pixel-based coords
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            self.image_path, 
            self.som_model, 
            BOX_TRESHOLD=self.BOX_THRESHOLD, 
            output_coord_in_ratio=False,  # <-- Important for absolute coords
            ocr_bbox=ocr_bbox,
            draw_bbox_config=self.draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            use_local_semantics=True, 
            iou_threshold=0.7, 
            scale_img=False, 
            batch_size=128
        )
        caption_time = time.time() - start - ocr_time
        print(f"Caption generation completed in {caption_time:.2f} seconds")
        
        for item in parsed_content_list:
            if 'bbox' in item:
                x1, y1, x2, y2 = item['bbox']
                # If these are normalized (0-1), multiply by image dimensions
                if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                    item['bbox'] = [
                        int(x1 * self.image_width),
                        int(y1 * self.image_height),
                        int(x2 * self.image_width),
                        int(y2 * self.image_height)
                    ]


        # Convert to DataFrame
        df = pd.DataFrame(parsed_content_list)
        df = df.drop(columns=["type", "interactivity", "source"])
        
        df_str = df.to_string(index=True)

        # Split the output into individual lines
        lines = df_str.split("\n")

        # Join those lines with a blank line in between
        double_spaced_str = "\n\n".join(lines)
        
        
        return dino_labeled_img, double_spaced_str


class OmniPostprocessingProvider:
    
    def __init__(
        self, llm_provider):
        self.llm_provider = llm_provider
        
    def __call__(self, augmented_image, bbox):
        
        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        
        self.augmented_image = augmented_image
        self.bbox = bbox
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 
                            f"You will be provided with an image that contains bounding boxes indicating segmented regions.\
                            The detailed information is encoded in the `bbox` variable: {self.bbox}.\
                            However, the semantic labels within the bounding boxes are currently inaccurate or insufficient.\
                            Your task is to revise and correct the semantic content of each bounding box.\n\n\
                            Please follow these rules:\n\
                            1. The output must maintain the same structure and format as the original `bbox` input.\n\
                            2. You are only allowed to modify the content (i.e., labels or descriptions) within each bounding box.\
                            You should only respond strictly in the following format, and you must not output any comments, explanations, or additional information:\
                            tools:\
                            ....\
                            "
                        
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.augmented_image}"
                        }
                    }
                ]
            }
        ]

        
    
        response = self.llm_provider.client.chat.completions.create(model=self.llm_provider.llm_model,
        messages=messages,
        temperature=config.temperature,
        seed=config.seed,
        max_tokens=config.max_tokens,)
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        
        params = {
            'tools': message,
            }
        memory.update_info_history(params)
            
        return message
        
        # There are different parameters which been required in the template by <>, extract them and check
        templete, inputkey, outputkey = extract_keys_from_template(config.provider_configs['pm_provider']['template_path'])

        check_input_keys(params, inputkey)

        # Assemble prompt, encode images for the input of llms
        message_prompts = self.llm_provider.assemble_prompt(template_str=templete, params=params)

        try:
            response = self.llm_provider.create_completion(message_prompts)

        except Exception as e:
            print(f"Response is not in the correct format: {e}, retrying...")

        
        response = parse_semi_formatted_text(response)
        
        check_output_keys(response, outputkey)

        del params

        return response