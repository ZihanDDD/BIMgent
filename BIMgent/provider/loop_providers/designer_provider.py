from copy import deepcopy
import os
from conf.config import Config
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from BIMgent.memory.local_memory import LocalMemory
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

config = Config()
memory = LocalMemory()

def show_image_temporarily(img, title="Image", seconds=3):

    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.draw()
    plt.pause(seconds)
    plt.close()

#---------------------------------------------------------------------------------------------------------------------------------------
#Here are the designer provider based on the llm generation: Gemini and @TODO OpenAI


class DesignerLLMProvider():
    """
    The designer provider for the design of the floorplan and the other structures that should be on the design for instance the door, windows using llms etc.
    """
    def __init__(
        self,
        task_description: str,
        **kwargs,
    ):
        load_dotenv()
        self.task_description = task_description
        self.api_key = os.getenv('OA_OPENAI_KEY')

        # Load and parse the template file once during initialization
    def text_to_floorplan(self, *args, **kwargs) -> str:

        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        client = OpenAI(api_key=self.api_key)
        
        prompt = f"""
            You are an experienced architect who can design floorplans image based on the user’s needs. You will use your extensive architectural knowledge to expand and supplement the user's original description and ultimately express your design in image format.
            Users description: {self.task_description}
            
            Please refer to basic architectural rules:
            1. The output image should be 512*512, the image only contains walls, doors and windows
            2. Wall Configuration: Arrange walls to define the building’s perimeter and internal spaces. Ensure that load-bearing walls are adequately spaced and placed to distribute the weight of the structure evenly. 
            3. Window Placement: Install windows strategically to provide natural light and ventilation to rooms. Ensure window locations are proportionate to the room size.
            4. Door Placement: Position doors for easy access to different rooms and areas. Main entrance to the building should be prominent and easy to locate, with interior doors facilitating smooth movement.
            5. Structural Integrity: Ensure all elements (walls, opening) are securely connected and stable.
            6. Compliance: Avoid clashing/overlapping building components, such as overlapping partitions between different areas and overlapping window and door locations. Adjacent rooms can share internal partitions. Rooms can also utilize exterior walls.
            7. Make your design spatially and geometrically rational. Use millimeter units. Minimize other prose.
            8. The generated image should not contain dimensional annotations. No annotations are allowed and only floorplan image containing walls, windows and doors are allowed to be generated.        
            9. The wall thickness in the response image should be small. This is indicated by the black colour and the thickness is in small size.

        """
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        screenshot_name = "input_floorplan.png"
        screenshot_path = os.path.join(config.work_dir, screenshot_name)

        # Save the image to a file
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
        # Add floorplan path to memory
        
        floorplan_para = {
            'floorplan_path': screenshot_path
        }
        memory.update_info_history(floorplan_para)

        del params

        return screenshot_path
    
    def process_existing_floorplan(self):
        # Deep copy memory to avoid unintended side effects
        params = deepcopy(memory.working_area)
        client = OpenAI(api_key=self.api_key)
        floorplan_path = params.get('floorplan_path')
        
        existing_floorplan = "existing_floorplan.png" 
        existing_floorplan_path = os.path.join(config.work_dir, existing_floorplan)
        shutil.copy2(floorplan_path, existing_floorplan_path)
        # Load the image file into a NumPy array
        img = mpimg.imread(floorplan_path)
        show_image_temporarily(img, 'Existing floorplan')
        
        
        prompt = f"""
            You are an experienced architect who can edit and enhance the existing floorplan image based on the user’s needs. Please generate a floorplan image, redraw the layout accurately
            
            Here are some rule you should follow on:
            1. Given a rough floorplan, generate a clean, architectural-style floorplan image. Redraw the layout accurately based on the hand-drawn sketch, but only include walls, doors, and windows. Ignore any furniture, kitchen details, or decorative elements. The final result should be neat, clear, and simplified for architectural use.
            2. The wall thickness in the response image should be small. This is indicated by the black colour and the thickness is in small size.
            3. The generated image should not contain dimensional annotations. No annotations are allowed and only floorplan image containing walls, windows and doors are allowed to be generated.        
        """
        
        result = client.images.edit(
        model="gpt-image-1",
        image=open(floorplan_path, 'rb'),
        prompt=prompt)
        
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        screenshot_name = "input_floorplan.png"
        screenshot_path = os.path.join(config.work_dir, screenshot_name)

        # Save the image to a file
        with open(screenshot_path, "wb") as f:
            f.write(image_bytes)
            
        floorplan_para = {
            'floorplan_path': screenshot_path
        }
        memory.update_info_history(floorplan_para)

        del params

        return screenshot_path




