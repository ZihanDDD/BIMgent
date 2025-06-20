import requests
import base64
import io
import json
import os
import matplotlib.pyplot as plt

from copy import deepcopy
from PIL import Image
from bim_gui_agent.memory.local_memory import LocalMemory
from conf.config import Config
memory = LocalMemory()
config = Config()


def decode_data_uri(data_uri: str):
    header, b64 = data_uri.split(',', 1)
    img_data = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_data))

def show_image_temporarily(img, title="Image", seconds=3):

    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.draw()
    plt.pause(seconds)
    plt.close()


def run_deepfloorplan():

    params = deepcopy(memory.working_area)
    
    img_path = params.get('floorplan_path')
    
    output_name = "cleaned.png"
    seg_output_path = os.path.join(config.work_dir, output_name)
    output_name = "segmented.png"
    output_path = os.path.join(config.work_dir, output_name)
    
    url="http://localhost:8888/deepfloorplan"
    with open(img_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(url, files=files)
    resp.raise_for_status()
    data = resp.json()

    seg_img = decode_data_uri(data["segmented_image"])
    vis_img = decode_data_uri(data["visualization"])

    show_image_temporarily(vis_img, title="Visualization", seconds=3)
    vis_img.save(seg_output_path)
    show_image_temporarily(seg_img, title="Segmented Image", seconds=3)
    seg_img.save(output_path)
    

    print("\nWalls:")
    walls    = json.dumps(data["cleaned"]["walls"],    ensure_ascii=False)
    
    print(walls)
    
    print("\nOpenings:")
    openings = json.dumps(data["cleaned"]["openings"], ensure_ascii=False)
    print(openings)
    
    floorplan_para = {
        'cleaned_floorplan_path': seg_output_path
    }
    memory.update_info_history(floorplan_para)

    return walls, openings
