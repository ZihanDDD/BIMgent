# BIM_GUI_agent
This repository contains the **BIM GUI Agent**. The basic workflow is illustrated below:

![Workflow Diagram](docs/BIMgent_images/general_workflow1.png)


## Setup

### Prepare API Keys

This project uses APIs from OpenAI, Anthropic (Claude), and Google (Gemini).  
You will need to update the `.env` file with your own API keys:
```
env OA_OPENAI_KEY="your_openai_api_key" # OpenAI API key 
Gemini_KEY="your_gemini_api_key" # Google Gemini API key 
```


### Dependencies
Please set up your Python environment and install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Get Started

Please update the configuration file at `conf/env_config_vectorworks.json` before running the agent:

1. If you plan to migrate to other BIM authoring tools, simply modify the `"env_path"` field to point to the desired software's launch path.

    You do **not** need to change the following fields:
    - `"env_name"`
    - `"sub_path"`
    - `"env_short_name"`

    These are reserved for future support of additional BIM tools, but that functionality is not yet implemented.

2. Modify the `task_description_list` to reflect the customer's specific requirements.  
   > ⚠️ Note: This feature is not currently active, as OpenAI GPT-4o’s original image generation functionality has not been released.

3. Currently, I manually generate a floorplan image using ChatGPT, download it, and update the path in `floorplan_image_path`.  
   Please replace this path with your own floorplan image as needed.

4. ownload Required Models

You need to download three sets of models and update the `models_path` configuration to point to the correct directories:

### Model Structure
After downloading, your models directory should contain:
```
models/
├── deep_floorplan/          # DeepFloorplan model files
├── Florence2/               # Florence2 model files  
└── omini/                   # OmniParser model file
    └── model.pt
```

### Download Instructions

#### 1. DeepFloorplan Model
- **Repository**: [DeepFloorplan](https://github.com/zlzeng/DeepFloorplan)
- **Download**: Follow the model download instructions in the repository
- **Target folder**: `deep_floorplan/`

#### 2. OmniParser Model
- **Repository**: [OmniParser](https://github.com/microsoft/OmniParser)
- **Download**: Follow the model download instructions in the repository
- **Target file**: `omini/model.pt` (or similar `.pt` file)


5. Update `panel_coordinates` to match your tool panel and design panel layout.  
   The format is `[x1, y1, x2, y2]` where:
   - `x1, y1` = top-left corner  
   - `x2, y2` = bottom-right corner

6. If you want to update the prompts directly, you can find them in:
   `res/vectorworks/propt/templates/`

   This folder contains prompt templates for:
   - **Floorplan Design**
   - **Project Manager**
   - **Builders**

   Prompts for **Floorplan Interpretation** and **OmniParser** are currently written directly in the script:
   `bim_gui_agent/provider/loop_providers/design_interpreter.py`


## Run

Use the following command to run the agent:

```bash
python agent_runner.py
```

To view the logic please check the script:
```
runner/vectorworks_runner.py
```
