from copy import deepcopy
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

from conf.config import Config
from BIMgent.memory.local_memory import LocalMemory
from BIMgent.provider.loop_providers.template_processor import extract_keys_from_template, check_input_keys, check_output_keys, assemble_prompt, parse_semi_formatted_text


load_dotenv()
config = Config()
api_key = os.getenv('OA_OPENAI_KEY')
config = Config()
memory = LocalMemory()

client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

    
    
class ComponentAgentProvider():
    def __init__(self, task_description, llm_provider):
        self.task_description = task_description
        self.llm_provider = llm_provider
        pass
    
    def codePlan(self, guidance, *args, **kwds):

        params = deepcopy(memory.working_area)
        
        response = {}
        current_task = params.get('current_task')
        floorplan = params.get('floorplan_metadata')
        

        prompt = f"""You are an artificial intelligence builder integrated with Vectorworks 2025 on this computer. you can build wall, window, slab or door. Your task is to implement low-level control functions in a way that effectively combines and sequences these actions to accomplish a given high-level subtask. Ensure the actions are context-aware, precise, and optimized for the specific tools and workflows in Vectorworks 2025.Here is some helpful information to help you make the decision.

        your current task:
        {current_task}

        tool guidance:
        {guidance}
        
        floorplan:
        {floorplan}

        Based on the provided information, you should generate substeps. Here are some hints for you to help decide.
        sub_steps:
        1. Generate a potential step workflow consisting of a step-by-step action process description to complete the task. Refer to the Tool Guidance for usage details and your current task.
        2. Provide detailed steps. For example:
            - If the task involves multiple elements (e.g., several walls, windows, or doors), generate a separate sub-step for each element until all element is totally finished.
            - Each step contains the drawing of one specific component. If there are 3 walls, generate 3 steps. If there are 1 slab, generate 1 step.
            - For each step, please select the related tool first, then start drawing.
            
        3. For the mode of the tool, set them as default.
        4. For actions, Generate a workflow consisting of the necessary actions to complete the task. You are given low-level control functions — move_mouse_to(x: int, y: int), left_click(), press_enter(), shortcut(combo : str), type_name(name: str), select_all() — which must be combined to form a workflow that completes the task.
            - If the task has the shortcut to call, use the shortcut.
            - To be notice, when you generate the shortcut(combo : str), the combo should be like combo = "alt + shift + 2" using '+' to connect the bottoms. control is 'ctrl'.
            - After select the right tool, start to draw the component.
        5. Your output of the actions must only be a list of actions in the following format: "['action1()', 'action2(x=..., y=...)', ...]" (string) Do not include any additional text or explanation.
        6. Coordinates are required (e.g., for clicking on a wall), the necessary (x, y) values will be provided in the floorplan. You should replace the number based on the component name's coordinate and the coordinates are in the floorplan metadata which is provided to you.
        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        sub_steps:
        {{
        "sub_step_1": {{
            "action name": "<string>",
            "actions": ["<string>", ...],
            "coordinates": [[<number>, <number>], ...],
            "description": "<string>"
        }},
        "sub_step_2": {{
            "action name": "<string>",
            "actions": ["<string>", ...],
            "coordinates": [[<number>, <number>], ...],
            "description": "<string>"
        }}
        // add “sub_step_3”, “sub_step_4”, … as needed, following the same pattern
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
                ],
            }
        ]
        
        # Assemble prompt, encode images for the input of llms
        model = 'gpt-4.1'
        response = self.llm_provider.client.chat.completions.create(model=model,
        messages=messages)

        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        
        sub_steps = message.split(':', 1)[1].strip()


        print(f"response received from {model}")
        
        params = {
            'sub_steps' : sub_steps
        }
        memory.update_info_history(params)
        
        

        del params

        return sub_steps
    
            
    def componentVerify(self, obj_info, class_type, length = None, mid_point = None):
        
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


    
    
class UIAgentProvider():
    def __init__(self, task_description, llm_provider):
        self.task_description = task_description
        self.llm_provider = llm_provider
        pass
    
    def low_level_planning(self, guidance, *args, **kwds):

        params = deepcopy(memory.working_area)
        response = {}
        current_task = params.get('current_task')
        floorplan = params.get('floorplan_metadata')
        
        prompt = f"""You are an artificial intelligence builder integrated with Vectorworks 2025 on this computer. 
        your current task:
        {current_task}
        tool guidance:
        {guidance}
        floorplan_metadata:
        {floorplan}


        Based on the provided information, you only need to consider the current task, you should generate substeps to finish current task. Here are some hints for you to help decide.
        sub_steps:
        You have to decide two types of actions:
        Vision-Driven: Actions that coordinates are not provided yet. Including shotcut actions.
        You have to specify in the description, which shortcut to use for the action!

        1. Generate a potential workflow consisting of a step-by-step action process description to complete the task. Refer to the Tool Guidance for usage details and your current task.
        2. Follow the steps outlined in the Tool Guidance to generate the actions sequentially.
        3. Provide detailed steps. For example:
            - (e.g., opening a panel, switching a tab, adjusting settings), break them into individual steps (e.g., "open panel", "switch tab", "apply settings").
            - If the task involves multiple elements (e.g., several walls, windows, or doors), generate a separate sub-step for each element until all element is totally finished.
            - For each step, please select the related tool first, then start drawing, it should be the first action for each substeps for the element creation.
            - Each step contains the drawing of one specific component. If there are 3 walls, generate 3 steps. If there are 1 slab, generate 1 step.
            - For the mode of the tool, set them as default.

        4. Every confirmation action (e.g., Confirm, OK, Enter) must be written as a separate, additional step. For example, after typing a name, do not include pressing Enter or clicking OK in the same step — instead, generate another step explicitly for the confirmation action, as these require separate verification.
        5. In each description:
            - Clearly mention the name of the UI component being interacted with.
            - Indicate which tool, control, shortcut name is being used to perform the step, for instance you have to specify which shortcut or actions should be done for the activate of the tool. They are in guidance.
            - When you do the edit of the layer, remember you need to specify the name of the editting layer, the editting layer should be the one you've just created.
            - Example phrasing of description:
            "Move the mouse to the 'Design Layers' and click it to switch to the Design Layers settings in the Organization dialog."   
        6. For roof creation, you do not need to select external wall. simplely use select_all() function will be enough to select all component to create the roof. And the process can be stoped at the Roof Creation. You do not need the further editing of the roof.

        Pure-Action: Actions that coordinates are Provided in floorplan_metadata, such as wall creation.
        1. Generate a potential step workflow consisting of a step-by-step action process description to complete the task. Refer to the Tool Guidance for usage details and your current task.
        2. Provide detailed steps. For example:
            - If the task involves multiple elements (e.g., several walls, windows, or doors), generate a separate sub-step for each element until all element is totally finished.
            - For each step, please select the related tool first, then start drawing, it should be the first action for each substeps for the element creation.
            - Each step contains the drawing of one specific component. If there are 3 walls, generate 3 steps. If there are 1 slab, generate 1 step.
        3. For the mode of the tool, set them as default. You do NOT need to Set Wall Preferences or any tool performance
        4. For actions, Generate a workflow consisting of the necessary actions to complete the task. You are given low-level control functions — move_mouse_to(x: int, y: int), left_click(), press_enter(), shortcut(combo : str), type_name(name: str), select_all() — which must be combined to form a workflow that completes the task.
            - If the task has the shortcut to call, use the shortcut.
            - To be notice, when you generate the shortcut(combo : str), the combo should be like combo = "alt + shift + 2" using '+' to connect the bottoms. control is 'ctrl'.
            - After select the right tool, start to draw the component.
        5. Your output of the actions must only be a list of actions in the following format: "['action1()', 'action2(x=..., y=...)', ...]" (string) Do not include any additional text or explanation.
        6. Coordinates are required (e.g., for clicking on a wall, slab cooridiantes), the necessary (x, y) values will be provided in the floorplan. You should replace the number based on the component name's coordinate and the coordinates are in the floorplan metadata which is provided to you.
            - For example, the coordinates that is needed for the creation of slab is already provided in floorplan data. You need to move the mouse to the coordinate, click, until all external wall are selected. Finally create the slab.
            - You should take the coordinates firstly from  floorplan metadata:{floorplan} firstly if there are component and cooresponding coordinates. Try not to generate by your self. If it do not existing coordinates of the component then try to guess your self.
        7. The output should be string,
        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format:

        {{
            "sub_step_1": {{
                "action name": "...",
                "action_type": "Vision-Driven",
                "description": "Detailed current step's goal"
            }},
            "sub_step_2": {{
                "action name": "...",
                "action_type": "Vision-Driven",
                "description": "Detailed current step's goal"
            }},
            "sub_step_3": {{
                "action name": "...",
                "action_type": "Pure-Action",
                "actions": ["action1()", "action2(x=..., y=...)"],
                "coordinates": [[x1, y1]],
                "description": "Detailed current step's goal"
            }},
            "sub_step_x": {{
                "action name": "...",
                "action_type": "Vision-Driven",
                "description": "Detailed x step's goal"
            }}
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
                ],
            }
        ]
        
        # Assemble prompt, encode images for the input of llms
        model = 'gpt-4.1'
        response = self.llm_provider.client.chat.completions.create(model=model,
        messages=messages)

        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        

        sub_steps = message
        print(f"response received from {model}")
        
        params = {
            'sub_steps' : sub_steps
        }
        memory.update_info_history(params)
        
        print(sub_steps)
        

        del params

        return sub_steps
    
    
    def visualVerify(self, screent_shot_path, current_sub_step, executed_actions):
        
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
    
        reasons:
        1. If failed, provide the reasons for your current decision. if success you just print success here.
        2. When generate the reasons, if there are object such as design layer, wall... You have to specify the name of the object. Such as Design layer, has the name 01-floor etc. You should mention it in the reasons.
        3. You should also propose a potential solution when a failure occurs, especially when executed_actions is not empty. Analyze why the current actions failed to complete the task, and suggest how the next action should be adjusted to achieve success. Your output should be in plain text, for example:
        “Based on the current screenshot, you should click ... to close ..., then do ... again to successfully complete the task.”        
        4. Remember, you are not going to recommand press enter or click ok this kind of confrim task to finish the workflow. It's always been seperated in a single sub task. 
        
        You should respond strictly in the following format, and you must not output any comments, explanations, or additional information. Don't include anything beside the requested data represented in the following format
        approved_value:
        success/fail
        
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

        model = 'o4-mini'
        response = self.llm_provider.client.chat.completions.create(model=model,
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
        reasons           = value_after("reasons:")

        return approved_value, reasons
    
    
    
    
    
    
    
    def uiExecute(self,description, full_path, meta_info, reasons):
        

        base64_image_seg_window = encode_image(full_path)
        
        
        prompt = f"""You are an amazing task augmentation provider. Your task is to implement low-level control functions in a way that effectively combines and sequences these actions to accomplish a given high-level subtask. Ensure the actions are context-aware, precise, and optimized for the specific tools and workflows in Vectorworks 2025.Here is some helpful information to help you make the decision.
        
        you will be provided with an image of the current screenshot image, which is already segmented, and the meta information of the provided image.
        meta_information of the labeled image:{meta_info}
        Your task:{description}
        Failed checking feedback: {reasons}

        
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
        9. If the workflow involves multiple actions, do not use "press Enter" or "click OK" as the final step. These should only be used when the action is a confirmation task and the only remaining step in the workflow. 

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

        model = 'o4-mini'
        response = self.llm_provider.client.chat.completions.create(model=model,
        messages=messages)
        
        if response is None:
            print("Failed to get a response from OpenAI. Try again.")

        message = response.choices[0].message.content
        
        sub_tasks = message.split(':', 1)[1].strip()
        print(f"response received from {self.llm_provider.llm_model}")
    

        return sub_tasks
    