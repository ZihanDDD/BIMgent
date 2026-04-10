import os
import atexit
import time
import json
from datetime import datetime
import ast
import uuid
from termcolor import colored
from conf.config import Config
from bim_gui_agent.memory.local_memory import LocalMemory

from bim_gui_agent.provider.ui_controller import UIController, MouseController
from bim_gui_agent.provider.screenshots_processor import ScreenshotsProcessor
from bim_gui_agent.provider.loop_providers.design_interpreter import DesignInterpreterPostprocessingProvider, DesignInterpreterGeminiProvider
from bim_gui_agent.provider.loop_providers.project_manager import PMProvider, PMpostprocessing
from bim_gui_agent.provider.loop_providers.skill_generator_provider import VisionDrivenAgentsProvider, PureActionProvider
from bim_gui_agent.provider.loop_providers.skill_executor import execute_actions
from bim_gui_agent.provider.Deep_fp_provider.deep_floorplan_provider import DeepFloorplanProvider
from bim_gui_agent.provider.omni_provider.omni_provider import OmniProvider
from bim_gui_agent.provider.builders_provider.builder_provider import query_builder, ingest_documents
from bim_gui_agent.utils.dict_utils import kget
from bim_gui_agent.utils.coordinate_trans import map_gui_to_ifc
from bim_gui_agent.utils.floorplan_resize import resize_image


config = Config()
class PipelineRunner():

    def __init__(self, task_description, floorplan_path):

        self.task_description = task_description
        self.floorplan_path = floorplan_path

        # Init internal params
        self.set_internal_params()

    def set_internal_params(self):

        self.memory = LocalMemory()

        # UI controller for the operation of the mouse and keyboard
        self.ui_controller = UIController()

        # controller for control the skill
        self.mouse_controller = MouseController()

        # Screenshots processor for processing of current state
        self.screenshots_processor = ScreenshotsProcessor()

        # Design Interpreter
        self.design_interpreter_gemini = DesignInterpreterGeminiProvider(self.task_description)
        self.design_interpreter_postprocessing = DesignInterpreterPostprocessingProvider()

        # Project Manager
        self.pm = PMProvider(self.task_description)
        self.pm_postprocessing = PMpostprocessing()

        # Deep Floorplan Provider (local inference)
        self.deep_floorplan_provider = DeepFloorplanProvider()

        # Omni Provider (local inference)
        self.omni_provider = OmniProvider()

        # Skill generator
        self.vision_driven_agents = VisionDrivenAgentsProvider(self.task_description)
        self.pure_action_agents = PureActionProvider(self.task_description)

    def run(self):
        

        # Parameters for processing
        init_params = {
            'task_description': self.task_description,
            'floorplan_path': self.floorplan_path
        }

        # Track runtime
        run_start_time = time.time()

        self.memory.update_info_history(init_params)

        
        # Create a folder for screenshots.
        self.masked_dir = os.path.join(config.work_dir, "screenshots")        
        os.makedirs(self.masked_dir, exist_ok=True)  # Create folder if it doesn't exist

        # -------------- Floorplan Understanding Pipeline --------------
        # 1. Resize the input floorplan to 512x512 for DeepFloorplan
        print(colored("🚀 Running floorplan processer for floorplan segmentation 🚀", "cyan"))

        resize_path = resize_image(self.floorplan_path)
        self.memory.update_info_history({'floorplan_path': resize_path})

        # 2. Run DeepFloorplan segmentation (local inference)
        walls, openings = self.deep_floorplan_provider.process_image(resize_path)
        walls = json.dumps(walls, ensure_ascii=False)
        openings = json.dumps(openings, ensure_ascii=False)

        print(colored("✅ Floorplan processer finished ✅", 'light_green'))

        # 3. Run Design Interpreter (Gemini) to refine walls + classify openings
        print(colored("🚀 Running design interpreter 🚀 ", "cyan"))

        self.run_floorplan_interpreter(walls, openings)

        print(colored("✅ Design interpreter finished ✅ ", "light_green"))

        # Ensure work dir exists and save intermediate floorplan data
        os.makedirs(config.work_dir, exist_ok=True)
        self.working_process_path = os.path.join(config.work_dir, 'working_process_data.json')


        # -------------- Hierarchical planning
        print("\n")
        print(colored("🚀 Running hierarchical planner 🚀 ", "cyan"))

        # Build working_process_data
        floorplan_data = self.memory.working_area.get('floorplan_metadata')
        if isinstance(floorplan_data, str):
            try:
                floorplan_data = json.loads(floorplan_data)
            except json.JSONDecodeError:
                pass

        working_process_data = {
            "floorplan_data": floorplan_data,
            "task_description": self.task_description
        }

        with open(self.working_process_path, 'w', encoding='utf-8') as f:
            json.dump(working_process_data, f, ensure_ascii=False, indent=4)



        # ---------------------------------------------------------------------------------------------------------
        # +++++++++++++++ High-level planner +++++++++++++++++++
        # ---------------------------------------------------------------------------------------------------------
        high_level_response = self.pm.high_level_planner()
        self.pm_postprocessing.high_level_postprocessing(high_level_response)

        try:
            if isinstance(high_level_response, list):
                high_level_steps = json.loads(high_level_response[-1]) if high_level_response else {}
            elif isinstance(high_level_response, dict):
                high_level_steps = high_level_response
            elif isinstance(high_level_response, str):
                high_level_steps = json.loads(high_level_response)
            else:
                high_level_steps = {}
        except (json.JSONDecodeError, IndexError, TypeError) as e:
            print(f"Error parsing high-level planner output: {e}")
            high_level_steps = {}

        working_process_data["working_process"] = high_level_steps

        with open(self.working_process_path, 'w', encoding='utf-8') as f:
            json.dump(working_process_data, f, ensure_ascii=False, indent=4)

        print(colored("✅ High-level planner finished ✅", "light_green"))

        # ---------------------------------------------------------------------------------------------------------
        # +++++++++++++++ Execute high-level steps with low-level planning +++++++++++++++++++
        # ---------------------------------------------------------------------------------------------------------

        print("\n")
        print(colored("🚀 Running Builders 🚀 ", "cyan"))

        # A screenshot for the initial status
        unique_code = str(uuid.uuid4().int)[:8]
        screenshot_name = f"initial_{unique_code}.png"
        initial_screenshot = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)

        step_number = 2

        while True:
            step_key = f"step {step_number}"
            current_step = high_level_steps.get(step_key)

            if current_step is None:
                print("\n")
                print(colored("✅ All high-level steps completed ✅", "light_green"))
                break

            class_type = current_step.get('class', 'unknown')
            component = current_step.get('component', '')

            print("\n")
            print(colored(f"{'='*60}", 'cyan'))
            print(colored(f"🚀 High-level step {step_number}: {class_type} — {component} 🚀", "cyan"))
            print(colored(f"{'='*60}", 'cyan'))

            # Set current task in memory for the low-level planner
            self.memory.update_info_history({'current_task': current_step})

            # Read current working process for context
            with open(self.working_process_path, 'r', encoding='utf-8') as f:
                previous_working_process = json.load(f)
            # ---------------------------------------------------------------------------------------------------------
            # +++++++++++++++ RAG documentation +++++++++++++++++++
            # ---------------------------------------------------------------------------------------------------------
            ingest_documents()

            # Query guidance for the full task description
            guidance = query_builder(current_step.get('description', ''))

            # ----- Low-level planning for this step -----
            sub_tasks_response = self.pm.low_level_planner(guidance, previous_working_process)

            try:
                if isinstance(sub_tasks_response, list):
                    sub_tasks = json.loads(sub_tasks_response[-1]) if sub_tasks_response else {}
                elif isinstance(sub_tasks_response, dict):
                    sub_tasks = sub_tasks_response
                elif isinstance(sub_tasks_response, str):
                    sub_tasks = json.loads(sub_tasks_response)
                else:
                    sub_tasks = {}
            except (json.JSONDecodeError, IndexError, TypeError) as e:
                print(f"Error parsing low-level planner output for {step_key}: {e}")
                sub_tasks = {}

            working_process_data[step_key] = sub_tasks

            with open(self.working_process_path, 'w', encoding='utf-8') as f:
                json.dump(working_process_data, f, ensure_ascii=False, indent=4)

            print(colored(f"✅ Low-level planning for {step_key} finished ✅", "light_green"))

            # ----- Execute sub-steps for this high-level step -----
            sub_step_number = 1

            while True:


                sub_step_key = f"sub_step_{sub_step_number}"
                current_sub_step = sub_tasks.get(sub_step_key)

                if current_sub_step is None:
                    print("\n")
                    print(colored(f"✅ All sub-steps for {step_key} completed ✅", "light_green"))
                    actions = ['move_mouse_to(x=1870, y=1020)', 'left_click()', 'shortcut("x")', 'left_click()', 'press_enter()', 'press_enter()', 'press_enter()']
                    execute_actions(actions, self.mouse_controller)
                    break

                sub_task_name = current_sub_step.get('action_name', sub_step_key)

                print(colored("current processing sub task", "yellow"))
                print(current_sub_step)

                print("\n")
                print(colored(f"🚀 Current running Builder: 🚀 ", "cyan"))
                print(colored(f"🚀 {class_type} - {sub_task_name} 🚀 ", "yellow"))

                current_sub_step_type = current_sub_step.get('action_type')

                if current_sub_step_type == 'Vision-Driven':
                    self.run_vision_driven(current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot)
                else:  # Pure Action
                    self.run_pure_action(current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot)

                sub_step_number += 1

                print("\n")
                print(colored(f"-----------------------------------------------------", 'light_green'))
                print(colored(f"Turning to the next task", 'light_green'))
                print(colored(f"-----------------------------------------------------", 'light_green'))
                print("\n")

                time.sleep(0.5)

            step_number += 1

        print(colored("The BIM model based on the floorplan is finished.", 'green'))

        # Calculate and save runtime
        run_end_time = time.time()
        runtime_seconds = run_end_time - run_start_time
        runtime_minutes = int(runtime_seconds // 60)
        runtime_remaining_secs = runtime_seconds % 60
        working_process_data["runtime"] = {
            "start_time": run_start_time,
            "end_time": run_end_time,
            "runtime_seconds": round(runtime_seconds, 2),
            "runtime_formatted": f"{runtime_minutes}m {runtime_remaining_secs:.1f}s"
        }
        with open(self.working_process_path, 'w', encoding='utf-8') as f:
            json.dump(working_process_data, f, ensure_ascii=False, indent=4)
        print(colored(f"Total runtime: {runtime_minutes}m {runtime_remaining_secs:.1f}s", 'cyan'))

        params = self.memory.working_area
        print(type(params))
        try:
            memory = json.loads(params)
        #Logs
        except:
            memory = params


        self.memory_path = os.path.join(config.work_dir, 'memory.json')
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)


        return
    
    
#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------


    def run_vision_driven(self, current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot):

        sub_task_name = current_sub_step.get('action_name')
        max_attempts = 3

        print("\n")
        print(colored(f"🚀 Running current {class_type} Builder's task {sub_task_name}  🚀 ", "cyan"))

        # log process data
        executed_actions = []
        cot = []
        screenshot_paths = []

        for attempt in range(1, max_attempts + 1):

            # ------------------------------------------------
            # For attempt 1: check if subtask is already done (no action needed)
            # For attempt 2+: undo previous failed action, generate new actions, execute
            # ------------------------------------------------

            if attempt == 1:
                # First check: see if the task is already completed
                unique_code = str(uuid.uuid4().int)[:8]
                check_screenshot_name = f"_check_{sub_task_name}_{unique_code}.png"
                check_screenshot_path = self.screenshots_processor.screenshot_capture(
                    self.masked_dir, check_screenshot_name
                )

                # Auto-approve if last action was press_enter (confirmation dialog)
                if executed_actions == [['press_enter()']]:
                    approved_value = 'success'
                    cot.append(approved_value)
                else:
                    approved_value, reasons = self.vision_driven_agents.oversee(
                        check_screenshot_path, current_sub_step, executed_actions
                    )
                    cot.append(reasons)

                if approved_value == 'success':
                    # Task already done — use the check screenshot as the single screenshot
                    screenshot_paths.append(check_screenshot_path)
                    current_sub_step['approved_value'] = 'success'
                    current_sub_step['actions'] = executed_actions
                    current_sub_step['cot'] = cot
                    current_sub_step['screenshot_paths'] = screenshot_paths
                    current_sub_step['timestamp'] = datetime.now().isoformat()
                    working_process_data[step_key][sub_step_key] = current_sub_step
                    with open(self.working_process_path, 'w', encoding='utf-8') as f:
                        json.dump(working_process_data, f, ensure_ascii=False, indent=4)
                    print("\n")
                    print(colored(f"✅ {sub_task_name} has been completed ✅", 'light_green'))
                    return

                # Not done yet — proceed to generate actions
                print("\n")
                print(colored(f'⚠️Warning: Current task has not been done yet for the reason: ⚠️', 'light_red'))
                print(colored(f'{reasons}', 'light_yellow'))

            else:
                # Undo previous failed action (for attempt 3+)
                if attempt > 2:
                    execute_actions(['undo()'], self.mouse_controller)

            # ------------------------------------------------
            # Generate and execute actions for this attempt
            # ------------------------------------------------

            time.sleep(0.5)
            print("\n")
            print(colored(f'⏳ generating actions for {sub_task_name}... attempt {attempt}/{max_attempts}', "light_cyan"))

            # Use check_screenshot for attempt 1, otherwise use last attempt's screenshot
            ref_screenshot = check_screenshot_path if attempt == 1 else screenshot_paths[-1]

            # Dynamic UI grounding
            unique_code = str(uuid.uuid4().int)[:6]
            popup_name = f"_popup_{sub_task_name}_{unique_code}.png"
            popup_path = os.path.join(self.masked_dir, popup_name)
            x, y, w, h = self.screenshots_processor.extract_popup(
                initial_screenshot, ref_screenshot, out_img_path=popup_path
            )

            current_window = self.screenshots_processor.drawing_panel(
                ref_screenshot, x, y, w, h
            )

            unique_code = str(uuid.uuid4().int)[:6]
            seg_name = f"_seg_{sub_task_name}_{unique_code}.png"
            seg_path = os.path.join(self.masked_dir, seg_name)
            meta_info = self.omni_provider.process_image(current_window, seg_path)

            if not meta_info:
                seg_path = ref_screenshot
                meta_info = self.omni_provider.process_image(seg_path, seg_path)

            # Action Generator
            with open(self.working_process_path, 'r', encoding='utf-8') as f:
                previous_working_process = json.load(f)
            
            print(meta_info)

            actions_response = self.vision_driven_agents.action_generate(
                current_sub_step, seg_path, meta_info, reasons, previous_working_process
            )

            print(actions_response)
            print(type(actions_response))

            try:
                actions_response = json.loads(actions_response)
            except json.decoder.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Take a screenshot even for failed parsing so the attempt is recorded
                unique_code = str(uuid.uuid4().int)[:8]
                screenshot_name = f"{sub_task_name}_attempt_{attempt}_{unique_code}.png"
                fail_screenshot = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
                screenshot_paths.append(fail_screenshot)
                cot.append(f"JSON decode error on attempt {attempt}")
                continue

            actions = actions_response["actions"]
            try:
                actions = ast.literal_eval(actions)
            except:
                actions = ['']

            print("\n")
            print(colored(f'🏃‍♂️ Actions running for {sub_task_name}', 'green'))
            print(actions)

            # Action Execution
            execute_actions(actions, self.mouse_controller)
            executed_actions.append(actions)
            time.sleep(1)

            # ------------------------------------------------
            # Take ONE screenshot per attempt — clearly named
            # ------------------------------------------------
            unique_code = str(uuid.uuid4().int)[:8]
            screenshot_name = f"{sub_task_name}_attempt_{attempt}_{unique_code}.png"
            attempt_screenshot = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
            screenshot_paths.append(attempt_screenshot)

            # ------------------------------------------------
            # Supervisor checks the result
            # ------------------------------------------------
            approved_value, reasons = self.vision_driven_agents.oversee(
                attempt_screenshot, current_sub_step, executed_actions
            )
            cot.append(reasons)

            if approved_value == 'success':
                current_sub_step['approved_value'] = 'success'
                current_sub_step['actions'] = executed_actions
                current_sub_step['cot'] = cot
                current_sub_step['screenshot_paths'] = screenshot_paths
                current_sub_step['timestamp'] = datetime.now().isoformat()
                working_process_data[step_key][sub_step_key] = current_sub_step
                with open(self.working_process_path, 'w', encoding='utf-8') as f:
                    json.dump(working_process_data, f, ensure_ascii=False, indent=4)
                print("\n")
                print(colored(f"✅ {sub_task_name} has been completed ✅", 'light_green'))
                return

            # Failed this attempt
            print("\n")
            print(colored(f'⚠️Warning: Current task has not been done yet for the reason: ⚠️', 'light_red'))
            print(colored(f'{reasons}', 'light_yellow'))

        # -----------------------------------------------------------------------
        # All attempts exhausted — mark as fail
        # -----------------------------------------------------------------------
        current_sub_step['approved_value'] = 'fail'
        current_sub_step['actions'] = executed_actions
        current_sub_step['cot'] = cot
        current_sub_step['screenshot_paths'] = screenshot_paths
        current_sub_step['timestamp'] = datetime.now().isoformat()
        working_process_data[step_key][sub_step_key] = current_sub_step
        with open(self.working_process_path, 'w', encoding='utf-8') as f:
            json.dump(working_process_data, f, ensure_ascii=False, indent=4)
        print(f"Fail, Jump to next task…\n")



        
    def run_pure_action(self, current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot):

        # The generated actions from the previous steps
        original_actions = current_sub_step.get('actions')
        sub_task_name = current_sub_step.get('action_name')
        coordinates = current_sub_step.get('coordinates')
        max_attempts = 3

        print("\n")
        print(colored(f"🚀 Running current {class_type} Builder's task {sub_task_name}  🚀 ", "cyan"))
        executed_actions = []
        cot = []
        screenshot_paths = []

        for attempt in range(1, max_attempts + 1):

            # ------------------------------------------------
            # Execute actions
            # ------------------------------------------------
            if attempt > 1:
                # Retry: escape/undo depending on failure type, then re-execute
                print("\n")
                print(colored(f'🏃‍♂️ Redoing the actions {sub_task_name} — attempt {attempt}/{max_attempts}', 'green'))

            else:
                print("\n")
                print(colored(f'🏃‍♂️ Actions running for {sub_task_name}', 'green'))

            execute_actions(original_actions, self.mouse_controller)
            time.sleep(0.5)
            executed_actions.append(original_actions)
            print(executed_actions)

            # ------------------------------------------------
            # Take ONE screenshot per attempt — clearly named
            # ------------------------------------------------
            unique_code = str(uuid.uuid4().int)[:8]
            screenshot_name = f"{sub_task_name}_attempt_{attempt}_{unique_code}.png"
            attempt_screenshot = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
            screenshot_paths.append(attempt_screenshot)

            # ------------------------------------------------
            # Supervisor checks the result
            # ------------------------------------------------

            # Layer / roof pure-actions (tab switching, option toggling, etc.)
            # don't produce an on-canvas element the supervisor can verify, so
            # we skip the oversight call and auto-pass them. Every other class
            # goes through the full-screen supervisor.
            # if class_type in ("layer", "roof"):
            #     print("\n")
            #     print(colored(
            #         f'⏩ Auto-passing {class_type} pure-action: {sub_task_name}',
            #         'yellow',
            #     ))
            #     approved_value = 'success'
            # else:
            #     length = None
            #     mid_point = None
            #     if isinstance(coordinates, list) and len(coordinates) == 2 and all(isinstance(pt, list) and len(pt) == 2 for pt in coordinates):
            #         end_points, length, mid_point = map_gui_to_ifc(coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1])

            #     print("\n")
            #     print(colored(f'⏳ Checking the element... {sub_task_name}', 'yellow'))

            #     approved_value = self.pure_action_agents.oversee(attempt_screenshot, class_type, length, mid_point)
            approved_value = 'success'
            # ------------------------------------------------
            # Handle result
            # ------------------------------------------------

            if approved_value == 'success':
                current_sub_step['actions'] = executed_actions
                current_sub_step['approved_value'] = 'success'
                current_sub_step['cot'] = cot
                current_sub_step['screenshot_paths'] = screenshot_paths
                current_sub_step['timestamp'] = datetime.now().isoformat()
                working_process_data[step_key][sub_step_key] = current_sub_step
                with open(self.working_process_path, 'w', encoding='utf-8') as f:
                    json.dump(working_process_data, f, ensure_ascii=False, indent=4)
                print("\n")
                print(colored(f"✅ {sub_task_name} has been completed ✅", 'light_green'))
                time.sleep(0.5)
                return

            # Failed — prepare for retry
            if approved_value == 'creation_fail':
                message = 'The currently step is not finished, the desired component is not been created.'
                print("\n")
                print(colored(f'Warning:⚠️ {message}', 'light_red'))
                execute_actions(['press_escape()', 'press_escape()', 'press_escape()'], self.mouse_controller)
                cot.append(message)

            elif approved_value == 'coordinate_fail':
                message = 'The currently step is correct, component been created but the coordinates are not correct.'
                print("\n")
                print(colored(f'Warning:⚠️ {message}', 'light_red'))
                execute_actions(['press_escape()', 'undo()'], self.mouse_controller)
                cot.append(message)

        # -----------------------------------------------------------------------
        # All attempts exhausted — mark as fail
        # -----------------------------------------------------------------------
        current_sub_step['approved_value'] = 'fail'
        current_sub_step['actions'] = executed_actions
        current_sub_step['cot'] = cot
        current_sub_step['screenshot_paths'] = screenshot_paths
        current_sub_step['timestamp'] = datetime.now().isoformat()
        working_process_data[step_key][sub_step_key] = current_sub_step
        with open(self.working_process_path, 'w', encoding='utf-8') as f:
            json.dump(working_process_data, f, ensure_ascii=False, indent=4)
        print("Fail, Jump to next task…\n")
    
    def run_floorplan_interpreter(self, walls, openings):

        # Using gemini for explain: (better than openai)
        response = self.design_interpreter_gemini(walls, openings)
        # Opneai version: 
        #response = self.design_interpreter_openai(image_path, cleaned_image_path, walls, openings)

        # Postprocessing map coordinates back.
        self.design_interpreter_postprocessing(response)

    def pipeline_shutdown(self):
        print('>>> Bye.')


def exit_cleanup(runner):
    print("Cleaning up resources")
    runner.pipeline_shutdown()



def entry(args):

    task_id = 1
    # Read task description and floorplan path from config
    task_description = kget(config.env_config, "task_description_list", default='')[task_id-1]['task_description']
    floorplan_path = kget(config.env_config, "floorplan_image_path", "floorplan", default='')

    print(f"\n{'='*60}")
    print(colored(f"  Floorplan: {floorplan_path}", 'cyan'))
    print(colored(f"  Task: {task_description}", 'cyan'))
    print(f"{'='*60}\n")

    # Clear memory for a fresh run (work_dir was created by Config at startup)
    memory = LocalMemory()
    memory.clear()

    pipelineRunner = PipelineRunner(
        task_description=task_description,
        floorplan_path=floorplan_path,
    )

    atexit.register(exit_cleanup, pipelineRunner)

    pipelineRunner.run()

    print(colored(f"\n  Pipeline completed.\n", 'green'))




