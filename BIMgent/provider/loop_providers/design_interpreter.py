from copy import deepcopy
import re
import os
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List
from conf.config import Config
import json
import PIL.Image
from dotenv import load_dotenv
import base64
import httpx
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from google import genai
from google.genai import types


from abc import ABC, abstractclassmethod

from BIMgent.provider.loop_providers.template_processor import extract_keys_from_template, check_input_keys, check_output_keys, assemble_prompt, parse_semi_formatted_text
from BIMgent.memory.local_memory import LocalMemory
from BIMgent.floorplan.floorplan_processer_llm import map_floorplan_to_new_bbox

from BIMgent.utils.dict_utils import kget


# If you still need Config as a global, leave this line:
config = Config()
memory = LocalMemory()
load_dotenv()

design_panel =  kget(config.env_config, "panel_coordinates", default='')['design_panel']



# Coordinates of the design bounding box
bounding_box = {
    "top_left": ( design_panel[0],  design_panel[1]),
    "bottom_left": (design_panel[0], design_panel[3]),
    "bottom_right": (design_panel[2], design_panel[3]),
    "top_right": (design_panel[2], design_panel[1]),
}

import json
import re

@staticmethod
def extract_floorplan_json(text):
    """
    Extract and parse the floorplan room data.
    
    Args:
        text (str): The input text containing floorplan data
        
    Returns:
        str: A valid JSON string containing the room data
    """
    # Find the array start and end
    start_index = text.find('[')
    end_index = text.rfind(']')
    
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        return "[]"
        
    # Get the content between brackets
    content = text[start_index + 1:end_index].strip()
    
    # Extract individual room objects
    rooms = []
    bracket_count = 0
    current_room = ''
    
    # Process character by character to handle nested brackets
    for char in content:
        if char == '{':
            bracket_count += 1
        elif char == '}':
            bracket_count -= 1
            
        current_room += char
        
        # If we've closed a room object, add it to our list
        if bracket_count == 0 and current_room.strip():
            trimmed_room = current_room.strip()
            
            # Only add if it looks like a complete room object
            if trimmed_room.startswith('{') and trimmed_room.endswith('}'):
                try:
                    # Try to parse as valid JSON
                    room_obj = json.loads(trimmed_room)
                    rooms.append(room_obj)
                except json.JSONDecodeError:
                    # Try to fix common issues
                    fixed_json = trimmed_room.replace("'", '"')
                    # Add quotes around property names if needed
                    fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                    
                    try:
                        room_obj = json.loads(fixed_json)
                        rooms.append(room_obj)
                    except json.JSONDecodeError:
                        # Still can't parse, skip this room
                        pass
            
            # Reset for next room
            current_room = ''
    
    # Convert the list of room objects to a JSON string
    return json.dumps(rooms, indent=2)



class DesignInterpreterProvider():
    """
    The designer provider for the design of the floorplan and the other structures that should be on the design for instance the door, windows using llms etc.
    """
    def __init__(
        self,
        task_description: str,
        llm_provider,
        **kwargs,
    ):
        self.task_description = task_description
        self.llm_provider = llm_provider

        # Load and parse the template file once during initialization
    def __call__(self,image_path, *args, **kwargs) -> str:
        

        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        response = {}

        # There are different parameters which been required in the template by <>, extract them and check
        templete, inputkey, outputkey = extract_keys_from_template(config.provider_configs['design_interpreter']['template_path'])
    
        check_input_keys(params, inputkey)

        # Assemble prompt, encode images for the input of llms
        message_prompts = self.llm_provider.assemble_prompt(template_str=templete, params=params)
        image = PIL.Image.open(image_path)
        


        try:
            message = [message_prompts[0], image]
            response = self.llm_provider.create_completion(message)

        except Exception as e:
            print(f"Response is not in the correct format: {e}, retrying...")
                
        
        response = response.text
        
        response = parse_semi_formatted_text(response)
        
        check_output_keys(response, outputkey)

        del params

        return response



class DesignInterpreterPostprocessingProvider():
    def __init__(self):
        pass


    def __call__(self, response):
    
        # Floorplan

        floorplan = response
        resolution = (512,512)
        
        floorplan = floorplan.strip()
        # If the string starts and ends with '{' and '}', remove them
        if floorplan.startswith("{") and floorplan.endswith("}"):
            # Remove the first and last character (the extra braces)
            inner = floorplan[1:-1].strip()
            # If the remaining string starts with '{', assume it is a list of dictionariesz``
            if inner.startswith("{"):
                floorplan = f"[{inner}]"
                
        floorplan_data = json.loads(floorplan)[0]
        
        mapped_floorplan = map_floorplan_to_new_bbox(floorplan_data, resolution, bounding_box)
        
        print(mapped_floorplan)
                
        new_param = {'floorplan_metadata': mapped_floorplan}
    
        memory.update_info_history(new_param)
        
        del new_param
                
        return mapped_floorplan
    
    


class DesignInterpreterGeminiProvider():
    
    def __init__(
        self,
        task_description: str
    ):
        self.task_description = task_description
        self.api_key = os.getenv('Gemini_KET')
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encodes an image to a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: File not found - {image_path}")
            return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""

        # Load and parse the template file once during initialization
    def __call__(self, walls_coord, openings_coord,  *args, **kwargs) -> str:
        
        
        param = deepcopy(memory.working_area)
        response = {}
        # floorplan_meta_infor = memory.get_recent_history('floorplan')
        image_path = param.get('floorplan_path')
        seg_image_path = param.get('cleaned_floorplan_path')
        
        
        pil_image1 = PIL.Image.open(image_path)
        pil_image2 = PIL.Image.open(seg_image_path)

        # walls_coord = memory.get_recent_history('walls')
        # openings_coord = memory.get_recent_history('openings')

        
        # Initialize Anthropic client
        client = genai.Client(api_key=self.api_key)
        
        prompt = f"""You are a skilled interpreter of architectural floorplans.
        You are provided with two images representing the same floorplan:
        1. **Original Floorplan Image** : A clean architectural drawing.
        2. **Segmented Floorplan Image** : Displays the same layout with labeled wall names and index annotations placed at the approximate center of each wall.

        Additionally, you are given:
        - **`{walls_coord}`** and  **`{openings_coord}`** : floorplan metadata information which includes A list of wall names and their corresponding coordinates.A list of opening coordinates (not yet labeled as doors or windows).
        
        
        floorplan:
        The metadata information and image 2 are basically correct, and you don't need to make major changes to the current layout, such as moving a point too long. You only need to make small changes to walls and openings where necessary, according to the float rules:
        1. Use Image 1 (original floorplan) as the baseline reference. Image 2 is mostly correct — only refine walls and openings where necessary.
        2. Remove noise or structural errors, including:
            - Short isolated walls whose endpoints do not connect to any other wall and are far from other walls.    
        3. Ensure wall connectivity by adjusting coordinates:
            - Extend or trim walls along their original direction to restore intersections observed in Image 1.
            - If a wall is isolated (i.e., disconnected at both endpoints) but is very close to another wall, snap its endpoints to the nearest neighboring wall without changing its orientation (only adjust length, not angle).
        4. Ensure that the final metadata is visually and structurally consistent with Image 1, accurately representing the floorplan. If confirmed as a duplicate detection, delete the extra instance.
        5. Classify Openings, Use the original floorplan image and the provided opening coordinates to determine whether each opening is a door or a window. Update their labels accordingly. Refer to Image 1 to check if 2 opening position represent the same physical opening. If yes delete it.
        6. Classify Walls, Using the **wall coordinates** and the **segmented image** (which shows wall names at their approximate positions), classify each wall as:
            - **External Wall**: Forms the continuous outer boundary of the floorplan. The boundary **must be closed** (i.e., walls form a sealed perimeter).
            - **Internal Wall**: All walls that are not part of the external boundary.
            - Wall names must remain unchanged.
        7. Order Components. When listing the components (walls, openings, etc.), always start from the top-left of the floorplan and continue in a clockwise direction.
        8. Define Slab Midpoints. For all final external walls, calculate the midpoint of each wall segment using its coordinates [[x1, y1], [x2, y2]]. Return a list of all such midpoints, which define the slab.       
        You must respond ONLY with the requested format and no explanatory text. No additional text. You are not allowed to add extra stuff like ```json and ``` for the start and end of the floorplan response:
        floorplan: 
            [
                {{
                    "external_wall_position": [
                        "Wall1: (x1, y1) to (x2, y2)",
                        "Wall2: (x3, y3) to (x4, y4)"
                    ],
                    "internal_wall_position": [
                        "Wall1: (x1, y1) to (x2, y2)",
                        "Wall2: (x3, y3) to (x4, y4)"
                    ],
                    "slab_position": [[x7, y7], [x8, y8], ...],
                    "doors_position": [[x5, y5]],
                    "windows_position": [[x6, y6]]
                }}
            ]
        """
        
        model = "gemini-2.5-pro-preview-05-06"


        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=[prompt,
                    pil_image1, pil_image2])
        
        # Extract the response text
        response_text = response.text
        response = extract_floorplan_json(response_text)
        print("\n")
        print(f"Respond recived from {model}")
        
        floorplan = json.loads(response)
        
    
        
        # === Plot ===
        plt.figure(figsize=(6, 6))
        for item in floorplan:
            external_wall_position = item.get("external_wall_position", [])
            internal_wall_position = item.get("internal_wall_position", [])
            walls = external_wall_position + internal_wall_position 
            
            doors = item.get("doors_position", [])
            windows = item.get("windows_position", [])
            openings = doors + windows 
                        
            # Plot each wall
            for wall_str in walls:
                start, end = self.parse_coordinate_string(wall_str)
                plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
                
                # Label the wall by its ID at its midpoint
                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                plt.text(mid_x, mid_y, self.get_wall_id(wall_str), color='blue', fontsize=10, 
                        ha='center', va='center')
            
            # Plot openings if provided
            for op in openings:
                plt.plot(op[0], op[1], 'ro', markersize=6)

        plt.axis('equal')
        plt.grid(False)
        plt.gca().invert_yaxis()

        # === Save ===
        screenshot_name = "postprocessed_floorplan_visualization.png"
        screenshot_path = os.path.join(config.work_dir, screenshot_name)
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        print(f"The processed floorplan image is saved in {screenshot_path}")

        floorplan_para = {
            'final_floorplan_path': screenshot_path
        }
        memory.update_info_history(floorplan_para)

        # === Show for 3 s, then close ===
        plt.show(block=False)   # display without blocking the rest of the script
        plt.pause(3)            # keep it up for ~3 seconds
        plt.close()             # close the figure window

        return response
    
        # === Helpers ===
    def parse_coordinate_string(self, wall_str):
        """Extract start and end coordinates from a wall string like 'Wall1: (x1, y1) to (x2, y2)'."""
        coords = re.findall(r'\(([\d\.\-]+),\s*([\d\.\-]+)\)', wall_str)
        if len(coords) == 2:
            start = (float(coords[0][0]), float(coords[0][1]))
            end = (float(coords[1][0]), float(coords[1][1]))
            return start, end
        return (0, 0), (0, 0)

    def get_wall_id(self, wall_str):
        """Extract the wall ID (e.g., Wall1) from the wall string."""
        match = re.match(r'(\w+):', wall_str.strip())
        return match.group(1) if match else "Wall"
        
