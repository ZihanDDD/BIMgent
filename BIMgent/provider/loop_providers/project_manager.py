from copy import deepcopy
import re
from string import Template
from BIMgent.memory.local_memory import LocalMemory
from BIMgent.provider.loop_providers.llm_provider import LLMProvider

memory = LocalMemory()


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` fences from LLM output."""
    return re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)


class PMProvider():

    def __init__(self, task_description: str, **kwargs):
        self.task_description = task_description
        self.llm = LLMProvider()

    def high_level_planner(self, *args, **kwargs) -> str:

        params = deepcopy(memory.working_area)
        floorplan_metadata = params.get('floorplan_metadata')

        with open("res/vectorworks/prompts/high_level_planner.prompt", "r", encoding="utf-8") as f:
            prompt_text = f.read()

        prompt = Template(prompt_text).substitute(
            task=self.task_description,
            floorplan=floorplan_metadata
        )

        response_content = self.llm.call(
            LLMProvider.MODEL_PLANNING, [prompt], top_p=0.95
        )

        response_content = _strip_json_fences(response_content)
        print(response_content)

        return response_content

    def low_level_planner(self, guidance, previous_working_process, *args, **kwds):

        params = deepcopy(memory.working_area)
        current_task = params.get('current_task')
        floorplan = params.get('floorplan_metadata')

        with open("res/vectorworks/prompts/low_level_planner.prompt", "r", encoding="utf-8") as f:
            prompt_text = f.read()

        prompt = Template(prompt_text).substitute(
            current_task=current_task,
            guidance=guidance,
            previous_working_process=previous_working_process,
            floorplan=floorplan
        )

        response_content = self.llm.call(
            LLMProvider.MODEL_PLANNING, [prompt], top_p=0.95
        )

        response_content = _strip_json_fences(response_content)

        memory.update_info_history({'sub_steps': response_content})
        print(response_content)

        return response_content



class PMpostprocessing():
    def __init__(self):
        pass


    def high_level_postprocessing(self, response):

        processed_response = {
            'working_process' : response
        }

        memory.update_info_history(processed_response)

        return processed_response
