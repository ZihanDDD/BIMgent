import os 
import atexit
import time
import json
import ast
from termcolor import colored
from conf.config import Config
from BIMgent.memory.local_memory import LocalMemory
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Providers
from BIMgent.provider.ui_controller import UIController, MouseController
from BIMgent.provider.screenshots_processor import ScreenshotsProcessor

from BIMgent.provider.loop_providers.design_interpreter import DesignInterpreterPostprocessingProvider, DesignInterpreterGeminiProvider
from BIMgent.provider.loop_providers.designer_provider import  DesignerLLMProvider
from BIMgent.provider.loop_providers.project_manager import PMProvider,PMpostprocessing
from BIMgent.provider.loop_providers.image_understanding_provider import ImageUnderstandingProvider
from BIMgent.provider.loop_providers.skill_generator_provider import  UIAgentProvider, ComponentAgentProvider
from BIMgent.provider.loop_providers.skill_executor import execute_actions
from BIMgent.provider.Deep_fp_provider.deep_floorplan_provider import DeepFloorplanProvider, DeepFloorplanPostprocessingProvider
from BIMgent.provider.omni_provider.omni_provider import OmniProvider, OmniPostprocessingProvider
from BIMgent.provider.builders_provider.builder_provider import query_builder, ingest_documents

#LLM providers
from BIMgent.provider.llm_provider.openai_provider import OpenAIProvider

#utils
from BIMgent.utils.dict_utils import kget
from BIMgent.utils.coordinate_trans import map_gui_to_ifc
from BIMgent.utils.result_phaser import JSONAnalyzer


config = Config()

class PipelineRunner():
    def __init__(self,task_description, floorplan_path):

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

        # ------------------------------------------------------------------------- 
        # LLMs providers
        
        self.llm_provider_gpt4o = OpenAIProvider('gpt-4o') 
        self.llm_provider_gpt4_5 = OpenAIProvider('gpt-4.5-preview')
        self.llm_provider_gpt41 = OpenAIProvider('gpt-4.1')
        self.llm_provider_gpto4mini = OpenAIProvider('o4-mini')

        # -------------------------------------------------------------------------
        # Providers

        #Designer
        self.designer_llm = DesignerLLMProvider(self.task_description)
                
        #Designer Interpreter
        self.design_interpreter_postprocessing = DesignInterpreterPostprocessingProvider() 
        self.design_interpreter_gemini = DesignInterpreterGeminiProvider(self.task_description)

        #Project Manager
        self.pm = PMProvider(self.task_description, self.llm_provider_gpt41)
        self.pm_postprocessing = PMpostprocessing()
        
        # Omni provider
        self.omini_provider = OmniProvider()
        self.omini_provider_postprocessing = OmniPostprocessingProvider(self.llm_provider_gpt4o)
        
        # Deep Floorplan Provider
        self.deep_floorplan = DeepFloorplanProvider()
        self.dfp_postprocessing = DeepFloorplanPostprocessingProvider(self.task_description)

        #Skill generator module launching
        self.ui_agent = UIAgentProvider(self.task_description, self.llm_provider_gpto4mini)
        self.component_agent = ComponentAgentProvider(self.task_description, self.llm_provider_gpto4mini)
        
        # Image understanding
        self.image_understanding = ImageUnderstandingProvider(self.llm_provider_gpto4mini)
    
    def show_image_temporarily(self, img, title="Image", seconds=3):

        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.draw()
        plt.pause(seconds)
        plt.close()
        

    def run(self):
        
        self.running_step_number = 0
        #Time cunting
        start_time = time.time()
        
        self.api_calling_times = 0
        
        # Parameters for processing
        init_params = {
            'task_description': self.task_description,
            'floorplan_path': self.floorplan_path,
        }

        self.memory.update_info_history(init_params)
        
        # Create a folder for screenshots.
        self.masked_dir = os.path.join(config.work_dir, "screenshots")        
        os.makedirs(self.masked_dir, exist_ok=True)  # Create folder if it doesn't exist

        # -------------- Running designer, to based on the task description call the designer. Return the floorplan parameters

        
        if self.floorplan_path == "no path":
            print(colored("🚀 Running designer to generate the floorplan based on the task description  🚀 ", 'cyan'))
            print("\n")
            image_path = self.designer_llm.text_to_floorplan()
        else:
            print(colored("🚀 Running designer to generate the floorplan based on the existing floorplan  🚀 ", 'cyan'))
            print("\n")
            image_path = self.designer_llm.process_existing_floorplan()
            

        # Load the image file into a NumPy array and show.
        img = mpimg.imread(image_path)
        self.show_image_temporarily(img)
        
        
        print(colored("✅ Designer finished ✅", 'light_green'))
        print("\n")
        
        # -------------------------------------------------------------------------
        

        # -------------- Running deep floorplan, for better understanding the floorplan components
        # Process including floorplan segmentation and an algorithm for noise moving and refinement.
        print(colored("🚀 Running floorplan processer for floorplan segmentation 🚀", "cyan"))
        print("\n")
        print(colored("floorplan components:", "yellow"))
        
        
        # Local running:
        seg_image_path, walls, openings = self.deep_floorplan.process_image(image_path)
        # Deployed server running:
        #walls, openings = run_deepfloorplan()
        init_params = {
            'cleaned_floorplan_path': seg_image_path
        }

        self.memory.update_info_history(init_params)
        
        
        print(colored("✅ Floorplan processer finished ✅", 'light_green'))
        print("\n")
        # -------------------------------------------------------------------------



        # -------------- Running Design Interpreter -- Process the segmented floorplan image, structure the compnents. Output with the component's coordinates.                
        print("\n")  
        print(colored("🚀 Running design interpreter 🚀 ", "cyan"))
        print("\n")
        print(colored("Interpreted floorplan components:", "yellow"))
        
        
        # Using gemini for explain: (better than openai)
        self.run_floorplan_interpreter(walls, openings)
        
        
        print("\n")
        print(colored("✅ Design interpreter finished ✅ ", "light_green"))
        # -------------------------------------------------------------------------


        
        # -------------- Running pm, based on the genearted floorplan name, coordinates and the connections to generate the process for each room design.
        print("\n")
        print(colored("🚀 Running project manager 🚀 ", "cyan"))
        
        # High-level planner
        self.run_pm()
        self.api_calling_times = self.api_calling_times + 1

        print("\n")
        print(colored("✅ Project manager Finished ✅", "light_green"))
        # -------------------------------------------------------------------------
    
    
        # -------------- Process each generated steps setup
        print("\n")
        print(colored("🚀 Running Builders 🚀 ", "cyan"))
        
        
        # Load the working process.                
        working_process_str = self.memory.get_recent_history('working_process')[0]
        working_process_data = json.loads(working_process_str)
        step_number = 1
        # -------------------------------------------------------------------------
        
        
        # load vector database
        ingest_documents()
        self.working_process_path = os.path.join(config.work_dir, 'working_process_data.json')
        with open(self.working_process_path, 'w', encoding='utf-8') as f:
            json.dump(working_process_data, f, ensure_ascii=False, indent=4)
            

        
        # -------------- steps strart
        while True:
            actions = ['move_mouse_to(x=1870, y=1020)', 'left_click()']
            execute_actions(actions, self.mouse_controller)

            step_key = f"step {step_number}"
            current_task = working_process_data.get(step_key)
            

            
            #load current task to the memory
            if current_task is None:
                print("\n")
                print(colored("✅ All Builders finished their tasks✅", "light_green"))
                break
            
            params = {
                 'current_task': current_task,
            }
            self.memory.update_info_history(params)
            
            
            class_type = current_task.get('class')
            description = current_task.get('description')
            agent_type = current_task.get('agent_type')


            
            print("\n")
            print(colored(f"🚀 Current running Builder: 🚀 ", "cyan"))
            print(colored(f"🚀 {class_type} 🚀 ", "yellow"))
            
            description = current_task.get('description')
            guidance = query_builder(description)
            
            # Generate the plan for the further steps Low-level planner
            sub_tasks = self.ui_agent.low_level_planning(guidance)
            
            print(sub_tasks)
            try:
                sub_tasks = json.loads(sub_tasks)
            except:
                print("not loaded to json, retry")
                pass

            self.api_calling_times = self.api_calling_times + 1

            # A screenshot for the initial status
            screenshot_name = f"initial_{class_type}"
            initial_screenshot = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
            
            sub_step_number = 1
            
            while True:

                sub_step_key = f"sub_step_{sub_step_number}"

                current_sub_step = sub_tasks.get(sub_step_key)
                
                if current_sub_step is None:
                    print("\n")
                    print(colored(f"✅ {class_type} finished their tasks✅", "light_green"))
                    break
                current_sub_step_type = current_sub_step.get('action_type')
                

                # 2 Execution Workflows
                if current_sub_step_type == 'Vision-Driven':
                    self.run_ui_agent(current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot)
                else:
                    self.run_component_agent(current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot)
                    
                sub_step_number = sub_step_number + 1
                        
                print("\n")
                print(colored(f"-----------------------------------------------------", 'light_green'))
                print(colored(f"Turning to the next task", 'light_green'))
                print(colored(f"-----------------------------------------------------", 'light_green'))
                print("\n")
                time.sleep(0.5)
                
            step_number = step_number + 1
            
        print(colored("The BIM model based on the floorplan is finished.", 'green'))
        
        # Time
        end_layer = time.time()
        run_time = end_layer - start_time
        print(f"Layer section ran in {end_layer - start_time:.4f} seconds")    
    
    
        params = {
            'run_time': run_time,
            'running_step_number': self.running_step_number,
            'api_calling_times': self.api_calling_times
        }
        self.memory.update_info_history(params)
        
           
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
            
        analyzer = JSONAnalyzer(self.memory_path, self.working_process_path)
        analyzer.load_and_process()
            
        
        return
    
#-----------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------


    def run_ui_agent(self, current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot):
                

        sub_task_name =  current_sub_step.get('action name')
        
        print("\n")
        print(colored(f"🚀 Running current {class_type} Builder's task {sub_task_name}  🚀 ", "cyan")) 
        executed_actions = []
        cot = []
        redo_times = 0
        
        while True:
                
            screenshot_name = f"{sub_task_name}_{redo_times}_times"
            current_screent_shot_path = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
            print(executed_actions)

            if executed_actions == [['press_enter()']]:
                approved_value = 'success'
                cot.append(approved_value)
            else:
                approved_value, reasons = self.ui_agent.visualVerify(current_screent_shot_path, current_sub_step, executed_actions)
                cot.append(reasons)
                self.api_calling_times = self.api_calling_times + 1
            if approved_value == 'success':
                current_sub_step['approved_value'] = 1 
                current_sub_step['actions'] = executed_actions
                current_sub_step['cot'] = cot
                working_process_data[step_key][sub_step_key] = current_sub_step

                with open(self.working_process_path, 'w', encoding='utf-8') as f:
                    json.dump(working_process_data, f, ensure_ascii=False, indent=4)
                
                screenshot_name = f"{sub_task_name}_{redo_times}_times"
                self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
                print("\n")
                print(colored(f"✅ {sub_task_name} has been completed ✅", 'light_green'))
                break
    

            print("\n")
            print(colored(f'⚠️Warning: Current task has not been done yet for the reason: ⚠️', 'light_red'))
            print(colored(f'{reasons}', 'light_yellow'))
            
            redo_times = redo_times + 1 
            
            if redo_times > 2:
                current_sub_step['approved_value'] = 0 
                current_sub_step['actions'] = executed_actions
                current_sub_step['cot'] = cot
                working_process_data[step_key][sub_step_key] = current_sub_step

                with open(self.working_process_path, 'w', encoding='utf-8') as f:
                    json.dump(working_process_data, f, ensure_ascii=False, indent=4)
    
                print("Fail, Jump to next task…\n")
                redo_times = 0
                break             
    

                        
            time.sleep(0.5)                          
            print("\n")
            print(colored(f'⏳ generating actions for {sub_task_name}... for the {redo_times} time', "light_cyan"))
            
            if redo_times > 1:
                actions = ['undo()']
                execute_actions(actions, self.mouse_controller)
                
            screenshot_name = "popup_window.png"
            screenshot_path = os.path.join(self.masked_dir, screenshot_name)
            x, y, w, h = self.screenshots_processor.extract_popup(initial_screenshot, current_screent_shot_path, out_img_path = screenshot_path)

            current_window = self.screenshots_processor.drawing_panel(
                current_screent_shot_path,
                x,               # x
                y,               # y
                w,  # width
                h   # height
            )


            image_name = "seg_image.png"
            full_path = os.path.join(self.masked_dir, image_name)
            
            self.omini_provider(current_window)
            image, meta_info = self.omini_provider.process_image()
            
            
            
            if not meta_info:
                self.omini_provider(current_screent_shot_path)
                image, meta_info = self.omini_provider.process_image()
            
            image.save(full_path)

            actions = self.ui_agent.uiExecute(current_sub_step, full_path, meta_info, reasons)
            
            
            try:
                actions = json.loads(actions)
            except json.decoder.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            
            actions = actions["actions"]
            try:
                actions = ast.literal_eval(actions) 
            except:
                actions = ['']
            
            print("\n")
            print(colored(f'🏃‍♂️ Actions running for {sub_task_name}','green'))
            print(actions)
            
            execute_actions(actions, self.mouse_controller)   
            self.running_step_number = self.running_step_number + 1   
            self.api_calling_times = self.api_calling_times + 1   
            executed_actions.append(actions)
            time.sleep(1)
            


        
    def run_component_agent(self,current_sub_step, class_type, working_process_data, step_key, sub_step_key, initial_screenshot):
        
        
    
        actions = current_sub_step.get('actions')
        sub_task_name =  current_sub_step.get('action name')
        coordinates = current_sub_step.get('coordinates')

        print("\n")
        print(colored(f"🚀 Running current {class_type} Builder's task {sub_task_name}  🚀 ", "cyan")) 
        executed_actions = []
        cot=[]
        redo_times = 1
        
        print("\n")
        print(colored(f'🏃‍♂️ Actions running for {sub_task_name}','green'))
        

        execute_actions(actions, self.mouse_controller)
        self.running_step_number = self.running_step_number + 1      
        time.sleep(3)
        executed_actions.append(actions)
        print(executed_actions)
        
        while True:
            
            screenshot_name = "object_info_check.png"
            screenshot_path = self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
            tool_panel =  kget(config.env_config, "panel_coordinates", default='')['object_info']
            x, y = tool_panel[0], tool_panel[1]
            width = tool_panel[2] - tool_panel[0]
            height = tool_panel[3] - tool_panel[1]
            # Capture screenshot directly from that region
            object_info_screenshot_path = self.screenshots_processor.capture_region(
                dir_path=self.masked_dir,
                screenshot_name=screenshot_name,
                x=x,
                y=y,
                width=width,
                height=height
            )
            length = None
            mid_point = None
            if isinstance(coordinates, list) and len(coordinates) == 2 and all(isinstance(pt, list) and len(pt) == 2 for pt in coordinates):
                end_points, length, mid_point = map_gui_to_ifc(coordinates[0][0],coordinates[0][1],coordinates[1][0], coordinates[1][1])
                
            print("\n") 
            print(colored(f'⏳ Checking the element... {sub_task_name}','yellow'))
            
            
            approved_value = self.component_agent.componentVerify(object_info_screenshot_path, class_type, length, mid_point)
            self.api_calling_times = self.api_calling_times + 1


            if redo_times > 1:
                current_sub_step['approved_value'] = 0
                current_sub_step['actions'] = executed_actions
                current_sub_step['cot'] = cot             
                working_process_data[step_key][sub_step_key] = current_sub_step

                with open(self.working_process_path, 'w', encoding='utf-8') as f:
                    json.dump(working_process_data, f, ensure_ascii=False, indent=4)
                    
                print("Fail,  Jump to next task…\n")
                redo_times = 0                     # reset counter so we pause again after 5 more steps
                break        

            
        
            if approved_value == 'creation_fail':
                
                redo_times = redo_times + 1
                
                message = 'The currently step is not finished, the desired component is not been created.'
                print("\n")
                print(colored(f'Warning:⚠️ {message}', 'light_red'))
                actions = ['press_escape()', 'press_escape()', 'press_escape()']
                execute_actions(actions, self.mouse_controller)
                
                print("\n")
                print(colored(f'🏃‍♂️ Redoing the actions {sub_task_name}','green'))
                actions = current_sub_step.get('actions')
                execute_actions(actions, self.mouse_controller)
                self.running_step_number = self.running_step_number + 1      
                executed_actions.append(actions)
                print(executed_actions)
                cot.append(message)
                continue
            
            elif approved_value == 'coordinate_fail':
                
                redo_times = redo_times + 1
                
                message = 'The currently step is correct, component been created but the coordinates are not correct.'
                print("\n")
                print(colored(f'Warning:⚠️ {message}', 'light_red'))
                actions = ['press_escape()', 'undo()']
                execute_actions(actions, self.mouse_controller)
                
                
                print("\n")
                print(colored(f'🏃‍♂️ Redoing the actions {sub_task_name}','green'))
                actions = current_sub_step.get('actions')
                execute_actions(actions, self.mouse_controller)
                executed_actions.append(actions)
                print(executed_actions)
                self.running_step_number = self.running_step_number + 1
                cot.append(message)
                continue
        
            else:
                
                current_sub_step['actions'] = executed_actions
                current_sub_step['approved_value'] = 1
                current_sub_step['cot'] = cot
                working_process_data[step_key][sub_step_key] = current_sub_step

                with open(self.working_process_path, 'w', encoding='utf-8') as f:
                    json.dump(working_process_data, f, ensure_ascii=False, indent=4)
                                    
                screenshot_name = f"{sub_task_name}"
                self.screenshots_processor.screenshot_capture(self.masked_dir, screenshot_name)
                actions = ['press_escape()']
                execute_actions(actions, self.mouse_controller)
                
                print("\n")
                print(colored(f"✅ {sub_task_name} has been completed ✅", 'light_green'))
                

                time.sleep(0.5)
                break
                

    
    def run_floorplan_interpreter(self, walls, openings):
        # Using gemini for explain: (better than openai)
        response = self.design_interpreter_gemini(walls, openings)
        self.design_interpreter_postprocessing(response)

    def run_pm(self):
        # Call llm api for the gui information reading
        response = self.pm()
        # add the responce into the current memory
        self.pm_postprocessing(response)

    def pipeline_shutdown(self):
        print('>>> Bye.')


def exit_cleanup(runner):
    print("Cleaning up resources")
    runner.pipeline_shutdown()



def entry(args):

    task_description = "No Task"

    task_id = 1
    # Read end to end task description from config file
    task_description = kget(config.env_config, "task_description_list", default='')[task_id-1]['task_description']
    
    floorplan_path = kget(config.env_config, "floorplan_image_path", default='')['floorplan']
    
    pipelineRunner = PipelineRunner(task_description=task_description, floorplan_path=floorplan_path)

    atexit.register(exit_cleanup,pipelineRunner)

    pipelineRunner.run()





