from copy import deepcopy
import re
from string import Template
from conf.config import Config
from BIMgent.memory.local_memory import LocalMemory
from BIMgent.provider.loop_providers.llm_provider import LLMProvider

config = Config()
memory = LocalMemory()


class VisionDrivenAgentsProvider():

    def __init__(self, task_description):
        self.task_description = task_description
        self.llm = LLMProvider()

    def oversee(self, screent_shot_path, current_sub_step, executed_actions):

        image_part = LLMProvider.read_image(screent_shot_path)

        with open("res/vectorworks/prompts/vision_driven_supervisor.prompt", "r", encoding="utf-8") as f:
            prompt_text = f.read()

        prompt = Template(prompt_text).substitute(
            current_sub_step=current_sub_step,
            executed_actions=executed_actions
        )

        message = self.llm.call(
            LLMProvider.MODEL_VISION, [prompt, image_part], top_p=0.95
        )

        print(message)

        def value_after(tag: str):
            if tag not in message:
                return None
            tail = message.split(tag, 1)[1]
            for line in tail.splitlines():
                line = line.strip()
                if line:
                    return line
            return None

        approved_value = value_after("approved_value:")
        reasons = value_after("reasons:")

        return approved_value, reasons

    def action_generate(self, description, full_path, meta_info, reasons, previous_working_process):

        with open("res/vectorworks/prompts/vision_driven_action_generator.prompt", "r", encoding="utf-8") as f:
            prompt_text = f.read()

        prompt = Template(prompt_text).substitute(
            description=description,
            meta_info=meta_info,
            reasons=reasons,
            previous_working_process=previous_working_process
        )

        image_part = LLMProvider.read_image(full_path)

        actions = self.llm.call(
            LLMProvider.MODEL_VISION, [prompt, image_part], top_p=0.95
        )

        print(actions)
        actions = re.sub(r'^```json\s*|\s*```$', '', actions.strip(), flags=re.MULTILINE)

        return actions





class PureActionProvider():
    def __init__(self, task_description):
        self.task_description = task_description
        self.llm = LLMProvider()

    def oversee(self, obj_info, class_type, length=None, mid_point=None):

        image_part = LLMProvider.read_image(obj_info)

        with open("res/vectorworks/prompts/pure_action_supervisor.prompt", "r", encoding="utf-8") as f:
            prompt_text = f.read()

        message = self.llm.call(
            LLMProvider.MODEL_VISION, [prompt_text, image_part], top_p=0.95
        )


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
