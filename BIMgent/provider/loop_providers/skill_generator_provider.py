import re
from string import Template

from BIMgent.provider.loop_providers.llm_provider import LLMProvider


class VisionDrivenAgentsProvider():
    """Vision-driven builder: generates GUI actions from a screenshot and
    verifies the outcome with a supervisor call."""

    def __init__(self, task_description):
        self.task_description = task_description
        self.llm = LLMProvider()

    def oversee(self, screenshot_path, current_sub_step, executed_actions):

        image_part = LLMProvider.read_image(screenshot_path)

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
