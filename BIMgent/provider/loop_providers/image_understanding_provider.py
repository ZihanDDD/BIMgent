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

import base64
# If you still need Config as a global, leave this line:
config = Config()
memory = LocalMemory()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



class ImageUnderstandingProvider:
    def __init__(
        self, llm_provider):
        self.llm_provider = llm_provider
        
    def __call__(self, current_tool):
        
        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        
        base64_image = encode_image(current_tool)

        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 
                            f"You will be provided with an image that contains the selected tool information.\
                            Your task is to understand what is select tool right now\
                            you should only reply the select tool name, one name will be enough, such as wall, slab, window, door etc.\
                            You should only respond strictly in the following format, and you must not output any comments, explanations, or additional information:\
                            selected_tool:\
                            ....\
                            "
                        
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        response = self.llm_provider.client.chat.completions.create(model=self.llm_provider.llm_model,
        messages=messages)
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")
            
        print(f"response received from {self.llm_provider.llm_model}")

        message = response.choices[0].message.content
        message = message.split(':', 1)[1].strip()
        
        params = {
            'selected_tool': message,
            }
        memory.update_info_history(params)
            
        return message
    
    def object_info(self, obj_info, class_type, length = None, mid_point = None):
        
        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        
        base64_image = encode_image(obj_info)
        
        print(class_type)
        
        prompt = f"""You will be provided with an image that contains the object information of the created object.
        
        component:
        On the top of the image, there is the information about the current finished object, you should understand the component name. And respone with the name. But keep it simple, such as if it's the window on the wall, just print window. If it's no selection just print no selection.
        
        If the component is wall, you also need to provide the meta information of the wall, which is L,A, X,Y: If the class type is not wall, you do not need to consider this step, just return None.
        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        component:
        ...
        L:
        ...
        A:
        ...
        X:
        ...
        Y:
        ...
        """

        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                        
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        response = self.llm_provider.client.chat.completions.create(model=self.llm_provider.llm_model,
        messages=messages)
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        print(f"response received from {self.llm_provider.llm_model}")

        
        def extract_values(input_string):
            try:
                # Extract values using known markers
                component = input_string.split("component:")[1].split("L:")[0].strip()
                l_value = input_string.split("L:")[1].split("A:")[0].strip()
                a_value = input_string.split("A:")[1].split("X:")[0].strip()
                x_value = input_string.split("X:")[1].split("Y:")[0].strip()
                y_value = input_string.split("Y:")[1].strip()

                return component, l_value, x_value, y_value
            except (IndexError, ValueError):
                return None
                
        component, l_value, x_value, y_value = extract_values(message)
        
        def evaluate_result(component, class_type, length, l_value, mid_point, x_value, y_value):
            
            # Normalize to lowercase for case-insensitive comparison
            if component.lower() in class_type.lower():
                # Case 1: no coordinates provided
                if l_value == 'None' and x_value == 'None' and y_value == 'None':
                    return 'success'

                try:
                    # Convert values to float
                    l_value = float(l_value)
                    x_value = float(x_value)
                    y_value = float(y_value)

                    # Compute differences
                    length_diff = abs(length - l_value)
                    x_diff = abs(mid_point[0] - x_value)
                    y_diff = abs(mid_point[1] - y_value)

                    # Check tolerances
                    if length_diff <= 300 and x_diff <= 300 and y_diff <= 300:
                        return 'success'
                    else:
                        print(f'coordinates error due to :length diff: {length_diff} x diff: {x_diff} y diff: {y_diff}')
                        return 'coordinate_fail'

                except (ValueError, TypeError):
                    return 'coordinate_fail'

            # If component not in class_type
            return 'creation_fail'
                    
        approved_value = evaluate_result(component,class_type, length, l_value, mid_point, x_value, y_value)
            
        return approved_value
    
    def layer(self, layer_panel):
        
        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        
        base64_image = encode_image(layer_panel)
        
        current_task = params.get('current_task')
        
        prompt = f"""You will be provided with an image that contains the information of the current created layer. 
        Additionally, you are provided with the current task content: {current_task}
        The current task is regarding to the creation of the new design layer. Your task is to analysis if the current created layer informaiton is correct. based on the current task.
        
        approved_value:
        1. If you find you are provided with a image contains the panel of the New Design layer, you should check:
            - If now the selected bottom is the 'Create a New Design Layer'
            - If the information next to the 'Name' under the 'Create a New Design Layer' is coorperated with the current task, basically if the floor name is correct.
            
        2. If you find you are provided with a image contaions the Edit Design Layers, you should check:
            - If the Elevation is coorperated with the current floor name.
            basically the floor 1 should have the elevation 0. Floor 2 have the elevation 3000, floor 3 have elevation 6000.
            the current floor information is provided in curernt task.
        
        
        3. If you find it's correct, simplely return success. if not return fail.
        
        Reason:
        provide the reason why you made the decision for approved_value. 

        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        approved_value:
        success/fail
        
        reason:
        1. ...
        2. ...
        
        """

        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                        
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        response = self.llm_provider.client.chat.completions.create(model=self.llm_provider.llm_model,
        messages=messages)
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        print(f"response received from {self.llm_provider.llm_model}")
        
        
        if "approved_value:" in message and "reason:" in message:
            parts = message.split("reason:")
            approved_block = parts[0]
            reason_part = parts[1].strip()

            approved_lines = approved_block.strip().splitlines()
            if len(approved_lines) >= 2:
                approved_value_part = approved_lines[1].strip()
        
        return approved_value_part, reason_part
    
    
    def layer_check(self, screent_shot_path, current_sub_step, executed_actions):
        
        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        
        base64_image = encode_image(screent_shot_path)
        
        
        prompt = f"""You will be provided with an screenshot of the current situation. 
        Additionaly, You are provided with the current task content: {current_sub_step}. and the execuded actions {executed_actions}.
        If executed_actions is an empty list, this indicates it is the first step and no actions have been taken yet. In this case, check whether the current default state already satisfies the task requirements.
        If executed_actions is not empty, it means actions have already been performed. You need to assess whether these actions have made the current state meet the task requirements.

        Based on the information provided, you must respond according to the following rules for     
        approved_value:
        1. If the task is to open a dialog, check whether the dialog appears in the screenshot
        2. If the task is to enter a name, verify that the name has been typed into the desired input field.
        3. If the task involves a confirmation action, you only need to check if the execuded actions includes the action like press enter or click ok, if yes respond with success.
        4. If the desired tool is selected, current active tool name  is in the left bottom, and in the tool panel the active tool is grey in background.
        5. If the current goal has been completed, respond with success; otherwise, respond with fail
        6. If the task is to select all, you just need to check if the select all action is been done. Do not need to check the selected components.
        
        screenshot_needed:
        1. Determine whether a screenshot is needed for the current step.
        2. If the step involves using a shortcut tool, a screenshot is not needed.
            If the step requires locating the coordinates of a specific button or UI element, a screenshot is needed.
        3. You do not need screenshot for Confirm tasks
        3. Response in True/False
        
        reasons:
        1. if failed, provide the reasons for your current decision. if success you just print success here.
        2. When generate the reasons, if there are object such as design layer, wall... You have to specify the name of the object. Such as Design layer, has the name 01-floor etc. You should mention it in the reasons.
        


        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        approved_value:
        success/fail
        
        screenshot_needed:
        True/False
        
        reasons:
        ...
        
        """

        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                        
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        response = self.llm_provider.client.chat.completions.create(model=self.llm_provider.llm_model,
        messages=messages)
        
        
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        print(f"response received from {self.llm_provider.llm_model}")
        
        
        def value_after(tag: str):
            if tag not in message:
                return None
            # text that comes after the tag
            tail = message.split(tag, 1)[1]
            # first non-blank line
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    return line
            return None

        approved_value    = value_after("approved_value:")
        screenshot_needed = value_after("screenshot_needed:")
        reasons           = value_after("reasons:")

        return approved_value, screenshot_needed, reasons
    
    
    
    def task_augmentation(self,description, executed_actions, full_path, meta_info, reasons):
        

        base64_image_seg_window = encode_image(full_path)
        
        
        prompt = f"""You are an amazing task augmentation provider. Your task is to implement low-level control functions in a way that effectively combines and sequences these actions to accomplish a given high-level subtask. Ensure the actions are context-aware, precise, and optimized for the specific tools and workflows in Vectorworks 2025.Here is some helpful information to help you make the decision.
        
        you will be provided with an image of the current screenshot image, which is already segmented, and the meta information of the provided image.
        meta_information of the labeled image:{meta_info}
        Your task:{description}
        previous actions: {executed_actions}
        reasons for the failed checking: {reasons}

        
        Based on the information provided to you, you need to response following the rules:
        Actions:
        1. Generate a workflow consisting of the necessary actions to complete the task.
        2. You are given low-level control functions — move_mouse_to(x: int, y: int), left_click(), press_enter(), shortcut(combo : str), type_name(name: str), select_all() — which must be combined to form a workflow that completes the task.
            - If the task has the shortcut to call, use the shortcut.
            - To be notice, when you generate the shortcut(combo : str), the combo should be like combo = "alt + shift + 2" using '+' to connect the bottoms. control is 'ctrl'.
        3. Your output of the actions must only be a list of actions in the following format: "['action1()', 'action2(x=..., y=...)', ...]" (string) Do not include any additional text or explanation.
        4. If you need the coordinantes of the specfic interactive bottom in the screen shot, current image's meta_information of the labeled image is provided in meta_info. When click, please make sure that you click on the middle point of the bounding box to ensure that you click on the bottom.
        5. If you still find it's hard to get the accurancy coordinate of the desired bottom, you need to defined the position by your self, to check which position you need to click to finish the task.
        5. You must only generate the solution for the current step. Do not include or assume any actions for future steps beyond what is described in the current task description. Actions such as pressing Enter or clicking Confirm are already set as separate future steps and must not be performed in advance.       
        6. If coordinates are required (e.g., for clicking on a wall), the necessary (x, y) values will be provided in the floorplan.
        7. If you find that some thing inside previous actions, you should learn from the previous actions, to improve this time.
        8. All click ok task, you should use press_enter to finish.
        
        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        Actions:       
        {{               
            "action name": "...",
            "actions": "['move_mouse_to(x=, y=)', ... ]"
        }}
        """

        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                        
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_seg_window}"
                        },
                    },
                ],
            }
        ]

        # Assemble prompt, encode images for the input of llms

        response = self.llm_provider.client.chat.completions.create(model=self.llm_provider.llm_model,
        messages=messages)
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        
        sub_tasks = message.split(':', 1)[1].strip()
        print(f"response received from {self.llm_provider.llm_model}")
        
        

        return sub_tasks
    
    
    