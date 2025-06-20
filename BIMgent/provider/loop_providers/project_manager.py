from copy import deepcopy
import re
import os
from typing import Dict, Any, List
from conf.config import Config
import json

from abc import ABC, abstractclassmethod


from BIMgent.provider.loop_providers.template_processor import extract_keys_from_template, check_input_keys, check_output_keys, parse_semi_formatted_text
from BIMgent.memory.local_memory import LocalMemory


# If you still need Config as a global, leave this line:
config = Config()
memory = LocalMemory()

pm_prompt = """"You are an assistant acting as a project manager for a building design and construction project using BIM software. You have received the complete floorplan from a designer. Your job is to guide the construction process step-by-step.

your task
<$task_description$>

A structured floorplan including coordinates and types of walls, doors, windows, etc.:
<$floorplan_metadata$>

Instructions and Hints for your decision. You should provide the current task that the following builder should do.
working_process:
1. The entire building project detail is already provided via floorplan.
2. The typical building order should be layers → External Walls → Slab → Internal Walls → Doors → Windows  → ... (loop until the final floor) → Roof
3. For each construction step, you should 
    - Based on the task description, you need to understand how many storeys are there, based on this, you need to create a new layer for the new storey.
    - Different floors are identical, for each floor, you need to manage to create components in floorplan. 
    - Every floor should have only one slab, which is based on the external walls.
    - You need to specify the component name and it's floor. 
    - You need to provide a description for the current step. For instance, if you task has specify requirement for the floor height, when you create the layer and walls, you need to set the height in the description. If there are not specific requriement just set as default.
4. You need to process all components in the floorplan. You have to finish the entire building design. manage every floor, and finally create the roof.
5. Specify the floor of the component by adding the floorx in component.
6. You should generate the agent type. If it's layer or roof, represent it's the UI agent you need to call afterwards. If it's wall, slab, door or window, represent it's the component agent.

You should respond strictly in the following format, and you must not output any comments, explanations, or additional information such as ``` are baned. Don't include anything beside the requested data represented in the following format.
working_process:
{    
    "step 1": {
        "class": "layer"
        "component": "layer_floorx",
        "description": "Detailed which design layer should be created currently for the current floor.",
    },
    "step 2": {
        "class": "external walls"
        "component": "wallx_floorx, ...",
        "description": "Detailed description of what needs to be done"
    },
    "step 3": {
        "class": "slab"
        "component": "slab_floorx",
        "description": "Detailed description of what needs to be done"
    },
    "step 4": {
        "class": "internal walls"
        "component": "wallx_floorx, ...",
        "description": "Detailed description of what needs to be done"
    },
    "step 5": {
        "class": "windows"
        "component": "windowx_floorx, ...",
        "description": "Detailed description of what needs to be done"
    },
    "step 6": {
        "class": "doors"
        "component": "doorx_floorx, ...",
        "description": "Detailed description of what needs to be done"
    },
    "step x": {
        "class": "the class of the drawing components, such as external wall/ internal wall/ slab/ layer etc. "
        "component": "the components in specific floor that should be draw here",
        "description": "Detailed description of what needs to be done"
    }
}

"""


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

class PMProvider():
    """
    The designer provider for the design of the floorplan and the other structures that should be on the design for instance the door, windows etc.
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

    def __call__(self, *args, **kwargs) -> str:

        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        response = {}

        # There are different parameters which been required in the template by <>, extract them and check
        
        
        templete = pm_prompt

        parse_input_keys = re.findall(r'<\$(.*?)\$>', templete)
        inputkey = [key.strip() for key in parse_input_keys]
        #print(f"Recommended input parameters: {input_keys}")

        start_output_line_index = templete.find('You should respond')
        output_text = templete[start_output_line_index + 1:]
        output = parse_semi_formatted_text(output_text)
        outputkey = list(output.keys())
        #print(f"Recommended output parameters: {output_keys}")

        check_input_keys(params, inputkey)

        # Assemble prompt, encode images for the input of llms
        message_prompts = self.llm_provider.assemble_prompt(template_str=templete, params=params)

        try:
            response = self.llm_provider.create_completion(message_prompts)

        except Exception as e:
            print(f"Response is not in the correct format: {e}, retrying...")
            
        print(response)

        response = parse_semi_formatted_text(response)
        
        check_output_keys(response, outputkey)

        del params
    

        return response




class PMpostprocessing():
    def __init__(self):
        pass


    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        memory.update_info_history(processed_response)

        return processed_response

