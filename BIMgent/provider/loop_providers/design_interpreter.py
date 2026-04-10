from copy import deepcopy
import re
import os
import ast
import json
import time
from conf.config import Config
from PIL import Image
from string import Template
import matplotlib.pyplot as plt
from bim_gui_agent.floorplan.floorplan_processer_llm import map_floorplan_to_new_bbox
from bim_gui_agent.memory.local_memory import LocalMemory
from bim_gui_agent.utils.dict_utils import kget
from bim_gui_agent.provider.loop_providers.llm_provider import LLMProvider

config = Config()
memory = LocalMemory()

design_panel =  kget(config.env_config, "panel_coordinates", default='')['design_panel']



# Coordinates of the design bounding box
# Detail number could be configured in configuration file. conf/env_config_vectorworks.json.
#@TODO after the refinement of the Design part lets see if it could be automatic.
bounding_box = {
    "top_left": ( design_panel[0],  design_panel[1]),
    "bottom_left": (design_panel[0], design_panel[3]),
    "bottom_right": (design_panel[2], design_panel[3]),
    "top_right": (design_panel[2], design_panel[1]),
}


###----------------------------
# Staticmethod for the floorplan informaiton extraction 
###----------------------------

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



class DesignInterpreterGeminiProvider():

    def __init__(self, task_description: str):
        self.task_description = task_description
        self.llm = LLMProvider()

    def __call__(self, walls_coord, openings_coord, *args, **kwargs) -> str:

        param = deepcopy(memory.working_area)
        image_path = param.get('floorplan_path')
        seg_image_path = param.get('cleaned_floorplan_path')

        image_part1 = LLMProvider.read_image(image_path)
        image_part2 = LLMProvider.read_image(seg_image_path)

        with open("res/vectorworks/prompts/design_interpreter.prompt", "r", encoding="utf-8") as f:
            prompt_text = f.read()

        prompt = Template(prompt_text).substitute(walls_coord=walls_coord)

        response_text = self.llm.call(
            LLMProvider.MODEL_UNDERSTANDING,
            [prompt, image_part1, image_part2],
            thinking_budget=8192,
            max_output_tokens=4096,
        )

        response = extract_floorplan_json(response_text)

        floorplan = json.loads(response)

        # Fallback: if the LLM failed to return a usable floorplan structure,
        # build one from the raw wall list so downstream stages still work.
        if not floorplan:
            if isinstance(walls_coord, str):
                try:
                    wall_list = ast.literal_eval(walls_coord)
                except (ValueError, SyntaxError):
                    wall_list = []
            else:
                wall_list = list(walls_coord)
            print("Design interpreter LLM returned empty; using raw walls as fallback.")
            floorplan = [{
                "external_wall_position": wall_list,
                "internal_wall_position": [],
                "slab_position": [],
                "stair_boundingbox": [],
                "stair_start_point": [],
                "windows_position": [],
                "doors_position": [],
            }]

        # opening coordinates
        if isinstance(openings_coord, str):
            openings_coord = ast.literal_eval(openings_coord)

        window = []
        door = []


        # Open the base image once
        base_img = Image.open(image_path)
        W, H = base_img.size


        # Original coordinate space
        COORD_WIDTH = 512
        COORD_HEIGHT = 512

        # Crop size in the target image space
        crop_size = 100  # 100x100 pixel crop in the actual image

        i = 0
        for (x, y) in openings_coord:
            
            # Map coordinates from 512x512 space to actual image space
            actual_x = (x / COORD_WIDTH) * W
            actual_y = (y / COORD_HEIGHT) * H
            
            # Half of the crop size
            r = crop_size / 2
            
            # Compute a valid PIL crop box = (left, top, right, bottom)
            left   = max(0, int(actual_x - r))
            top    = max(0, int(actual_y - r))
            right  = min(W, int(actual_x + r))
            bottom = min(H, int(actual_y + r))
            
            # Skip degenerate boxes
            if right <= left or bottom <= top:
                print(f"Skip invalid box at ({x}, {y}) -> mapped to ({actual_x}, {actual_y}) -> {(left, top, right, bottom)}")
                continue
            
            # Crop the opening patch
            opening_patch = base_img.crop((left, top, right, bottom))
            
            opening_dir = os.path.join(config.work_dir, "openings")
            os.makedirs(opening_dir, exist_ok=True)
            
            opening_name = os.path.join(opening_dir, f"opening_{i}.png")
            opening_patch.save(opening_name)

            opening_part = LLMProvider.read_image(opening_name)

            prompt_opening = (
            """
            Classify the building opening in this architectural floor plan image patch. Ignore any english words on it.

            - **Door**: Has a curved arc line (door swing symbol) extending from the opening. If there is a hole on the stright line, it's door too.
            - **Window**: Only straight lines with no hole, no arc.

            Return only: 'door' or 'window'
            """
            )

            pred_text = self.llm.call(
                LLMProvider.MODEL_UNDERSTANDING,
                [prompt_opening, opening_part],
                thinking_budget=1024,
                max_output_tokens=256,
            )

            pred = pred_text.strip().lower()

            i = i + 1

            x = int(x)
            y = int(y)

            if "door" in pred and "window" not in pred:
                door.append([x, y])
            elif "window" in pred and "door" not in pred:
                window.append([x, y])
            else:
                door.append([x, y])


        # Save results
        floorplan[0]['windows_position'] = window
        floorplan[0]['doors_position'] = door

        # === Plot ===
        plt.figure(figsize=(6, 6))
        for item in floorplan:
            external_wall_position = item.get("external_wall_position", [])
            internal_wall_position = item.get("internal_wall_position", [])
            walls = external_wall_position + internal_wall_position 
            
            doors = item.get("doors_position", [])
            windows = item.get("windows_position", [])
            
            stair_boundingbox = item.get("stair_boundingbox", [])
            stair_start_point = item.get("stair_start_point", [])
                        
            # Plot each wall
            for wall_str in walls:
                start, end = self.parse_coordinate_string(wall_str)
                plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)
                
                # Label the wall by its ID at its midpoint
                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                plt.text(mid_x, mid_y, self.get_wall_id(wall_str), color='blue', fontsize=10, 
                        ha='center', va='center')
            
            # Plot doors (blue points)
            for door in doors:
                plt.plot(door[0], door[1], 'bo', markersize=6, label='Door' if doors.index(door) == 0 else '')
            
            # Plot windows (red points)
            for window in windows:
                plt.plot(window[0], window[1], 'ro', markersize=6, label='Window' if windows.index(window) == 0 else '')
            
            # Plot stair bounding box if provided
            if len(stair_boundingbox) == 2:
                # stair_boundingbox contains two corner points: [[x1, y1], [x2, y2]]
                corner1 = stair_boundingbox[0]
                corner2 = stair_boundingbox[1]
                
                # Calculate all four corners of the rectangle
                x_min = min(corner1[0], corner2[0])
                x_max = max(corner1[0], corner2[0])
                y_min = min(corner1[1], corner2[1])
                y_max = max(corner1[1], corner2[1])
                
                # Draw the rectangle
                rect_x = [x_min, x_max, x_max, x_min, x_min]
                rect_y = [y_min, y_min, y_max, y_max, y_min]
                plt.plot(rect_x, rect_y, 'g-', linewidth=2, label='Stair Bounding Box')
                
                # Optionally fill the rectangle with semi-transparent color
                plt.fill(rect_x, rect_y, color='green', alpha=0.2)
                
                # Add label at the center of the bounding box
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                plt.text(center_x, center_y, 'STAIR', color='green', fontsize=12, 
                        ha='center', va='center', fontweight='bold')
            
            # Plot stair start point (yellow points)
            for start_pt in stair_start_point:
                plt.plot(start_pt[0], start_pt[1], 'yo', markersize=8, 
                        label='Stair Start' if stair_start_point.index(start_pt) == 0 else '',
                        markeredgecolor='orange', markeredgewidth=1.5)

        plt.axis('equal')
        plt.grid(False)
        plt.gca().invert_yaxis()
        
        # Remove duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        # === Save ===
        screenshot_name = "postprocessed_floorplan_visualization.png"
        screenshot_path = os.path.join(config.work_dir, screenshot_name)
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        print(f"The processed floorplan image is saved in {screenshot_path}")

        floorplan_para = {
            'final_floorplan_path': screenshot_path
        }
        memory.update_info_history(floorplan_para)

        # Show the visualization briefly, then close it.
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        return floorplan
    
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
        


#Postprocessing
#--------function: map the generated coordinates of the floorplan to the GUI position.

class DesignInterpreterPostprocessingProvider():
    def __init__(self):
        pass


    def __call__(self, response):

        floorplan = response
        resolution = (512, 512)
        floorplan_data = floorplan



        if isinstance(floorplan_data, str):
            # Parse the string representation of a Python list/dict
            floorplan_data = json.loads(floorplan_data)

        # Upstream returns the floorplan wrapped in a one-element list
        # (``[{...}]``); ``map_floorplan_to_new_bbox`` expects a plain dict.
        if isinstance(floorplan_data, list):
            if not floorplan_data:
                raise ValueError("design_interpreter produced an empty floorplan list")
            floorplan_data = floorplan_data[0]

        mapped_floorplan = map_floorplan_to_new_bbox(floorplan_data, resolution, bounding_box)

        new_param = {'floorplan_metadata': mapped_floorplan}
    
        memory.update_info_history(new_param)
        
        del new_param
                
        return mapped_floorplan

