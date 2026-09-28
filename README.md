# BIMgent: Towards Autonomous Building Modeling via Computer-use Agents

🔗 **Website**: [https://tumcms.github.io/BIMgent.github.io/](https://tumcms.github.io/BIMgent.github.io/)
📄 **Paper on arXiv**: [arXiv:2506.07217](https://arxiv.org/abs/2506.07217)

![BIMgent overview](docs/overview.png)

**BIMgent** is an agentic framework that lets Large Language Models autonomously perform architectural building modeling in a real BIM authoring tool. Instead of generating BIM files through custom APIs, BIMgent drives the *actual* GUI of [Vectorworks](https://www.vectorworks.net/) — it looks at the screen, plans like a human modeler, and acts through the mouse and keyboard. Given a floorplan image and the number of floors, the agent produces a complete 3D BIM model end-to-end.

## 🎥 Demo

**From a floorplan image to a finished BIM model in Vectorworks, fully driven by the agent.**

![BIMgent demo](docs/BIMgent_demo.gif)

More demo videos are available on the project website:
🔗 [https://tumcms.github.io/BIMgent.github.io/](https://tumcms.github.io/BIMgent.github.io/)

## 🧠 Framework Overview

![BIMgent framework: floorplan understanding, hierarchical planning, action execution, generated models](docs/graphic_abstract.png)

BIMgent implements the three-stage pipeline described in the paper:

1. **Floorplan Understanding** — the input floorplan image is resized and segmented by **DeepFloorplan** to extract wall and opening candidates. A **Design Interpreter** (Gemini) then refines wall geometry and classifies openings into doors and windows, producing structured floorplan metadata.

2. **Hierarchical Planning (Project Manager)** — an LLM-based planner decomposes the modeling task into:
   - a **high-level plan** (ordered list of modeling steps such as *create layers → build walls → insert doors → insert windows → add slabs → add roof*), and
   - a **low-level plan** per step, grounded in **retrieval-augmented builder documentation** (`res/vectorworks/builders/*.md`) so the agent follows the correct tool-specific workflow.

3. **Builders with Closed-Loop GUI Control** — each sub-step is executed by one of two builder agents:
   - **Vision-Driven Agent** — for tasks that require on-screen reasoning. It takes a screenshot, uses **OmniParser** for dynamic UI grounding, generates mouse/keyboard actions, executes them, and lets a **supervisor** LLM verify the outcome with a retry loop.
   - **Pure-Action Agent** — for tasks where coordinates and keystrokes are precomputed from the floorplan metadata (e.g. drawing walls along known endpoints).

All interactions with Vectorworks happen through the `MouseController` (PyAutoGUI-based) and a `ScreenshotsProcessor` that captures full-screen screenshots and masks them down to the changed pop-up / design-panel region.

## 🗂️ Repository Structure

```
BIMgent/
├── agent_runner.py                 # Entry point — prompts for floorplan + floors, then runs the pipeline
├── .env.example                    # Template for API keys (copy to .env)
├── conf/
│   ├── config.py                   # Global Config (singleton)
│   └── env_config_vectorworks.example.json # Template env config (copy to env_config_vectorworks.json)
├── BIMgent/                        # Core package
│   ├── runner/
│   │   └── vectorworks_runner.py   # Main pipeline orchestration
│   ├── provider/
│   │   ├── ui_controller.py        # Mouse / keyboard control
│   │   ├── screenshots_processor.py# Screenshot capture + popup / panel masking
│   │   ├── Deep_fp_provider/       # DeepFloorplan wall/opening segmentation + geometric post-processing
│   │   ├── omni_provider/          # OmniParser UI element grounding
│   │   ├── builders_provider/      # RAG over builder documentation
│   │   └── loop_providers/
│   │       ├── design_interpreter.py     # Gemini-based floorplan interpreter
│   │       ├── project_manager.py        # High-level + low-level planners
│   │       ├── skill_generator_provider.py # Vision-driven builder (action generator + supervisor)
│   │       ├── skill_executor.py         # Executes generated actions
│   │       └── llm_provider.py           # Single entry point for all Gemini calls
│   ├── memory/                     # Working-area memory shared across stages
│   ├── floorplan/                  # Maps floorplan coordinates onto the Vectorworks design panel
│   └── utils/                      # Config helpers, image resizing, Gemini retry wrapper
├── res/vectorworks/
│   ├── prompts/                    # Prompt templates for every LLM stage
│   └── builders/                   # Markdown docs retrieved via RAG (wall, door, window, slab, roof, stair, layer)
├── mini_building_benchmark/        # Mini benchmark + evaluation scripts used in the paper
└── docs/                           # Figures and demo media
```

## 🔧 Setup

### 1. Install Vectorworks
The current release targets the BIM authoring tool **Vectorworks 2025**. A valid license is required. Install it, then create your local environment config from the template and set `env_path` to the Vectorworks executable:
```bash
cp conf/env_config_vectorworks.example.json conf/env_config_vectorworks.json
```
`conf/env_config_vectorworks.json` is git-ignored because it holds machine-specific paths and is rewritten by `agent_runner.py` on every run.

### 2. Python environment
Create a Python **3.10** environment and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. API keys
BIMgent uses Google Gemini for planning, vision and design interpretation, and OpenAI embeddings for the builder-documentation RAG. Copy the template and fill in your keys:
```bash
cp .env.example .env
```
```env
OA_OPENAI_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```
`.env` is git-ignored — never commit it.

### 4. Download external models
BIMgent reuses two pretrained models. Download them and update the paths under `models_path` in `conf/env_config_vectorworks.json`.

| Model | Purpose | Repository |
| --- | --- | --- |
| **DeepFloorplan** | Wall / opening segmentation from the input floorplan image | [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) |
| **OmniParser** | Dynamic UI grounding of Vectorworks panels & dialogs (icon detector + Florence-2 captioner) | [microsoft/OmniParser](https://github.com/microsoft/OmniParser) |

Expected entries in the config:
```json
"models_path": {
  "deep_floorplan": "<path>/deep_floorplan/pretrained",
  "omini":          "<path>/omni/weights/icon_detect/model.pt",
  "Florence2":      "<path>/omni/weights/icon_caption_florence"
}
```

### 5. Calibrate panel coordinates
Vectorworks' panels must be mapped so the agent knows where to look and click. Update `panel_coordinates` in `conf/env_config_vectorworks.json` to match your screen layout:

```json
"panel_coordinates": {
  "tool_panel":   [x1, y1, x2, y2],   // Left-hand tool list
  "design_panel": [x1, y1, x2, y2],   // Central modeling canvas
  "object_info":  [x1, y1, x2, y2],   // Right-hand object info panel
  "whole_panel":  [0, 0, 1920, 1080]  // Full screen
}
```
Each rectangle is `[top-left x, top-left y, bottom-right x, bottom-right y]`. A quick way to read screen coordinates is to hover the mouse and run:
```bash
python -c "import pyautogui, time; time.sleep(3); print(pyautogui.position())"
```

## 🚀 Running the Agent

Launch Vectorworks with an empty document, arrange the panels to match your `panel_coordinates`, then run:

```bash
python agent_runner.py
```

You will be prompted for:
1. **The path to a floorplan image** (a real photo, a rendered floorplan, or a hand-drawn sketch).
2. **The number of floors** the building has.

Non-interactive usage:
```bash
python agent_runner.py \
  --envConfig ./conf/env_config_vectorworks.json \
  --floorplanPath /path/to/floorplan.png \
  --taskDescription "The building has 2 floors based on the provided layout."
```

During execution the agent will:
1. Run DeepFloorplan + the Design Interpreter to produce floorplan metadata.
2. Call the high-level planner, then the low-level planner (per step, with RAG guidance).
3. Take control of mouse and keyboard to model the building in Vectorworks, with screenshot-based supervision and retries.

**⚠️ Note:** Once the agent starts acting, keep your hands off the mouse and keyboard — the agent is driving the real desktop. Intermediate screenshots, planner outputs, and per-step logs are saved under the run's work directory (`memory.json`, `working_process_data.json`, and `screenshots/`).

The main orchestration logic lives in [`BIMgent/runner/vectorworks_runner.py`](BIMgent/runner/vectorworks_runner.py).

## 📊 Mini Building Benchmark

The `mini_building_benchmark/` folder contains the small-scale benchmark used in the paper for evaluating floorplan understanding and end-to-end modeling:
- `ori_images/` and `scaled_images/` — source and resized floorplans,
- `GT/` — ground-truth annotations,
- `run_floorplan_understanding.py` — runs DeepFloorplan + the Design Interpreter over all 45 cases and writes predictions to `floorplan_understanding/`,
- `eva_matric.py` — compares those predictions against `GT/` and writes `evaluation_results.xlsx` (requires `openpyxl`).

```bash
python mini_building_benchmark/run_floorplan_understanding.py
python mini_building_benchmark/eva_matric.py
```

## 📎 Appendices: All 45 Evaluation Cases

The complete set of floorplans, task specifications, and generated model visualizations for all 45 evaluation cases did not fit in the paper and is provided in [`appendices/`](appendices/). Click any page to open it at full resolution.

<details>
<summary><b>Appendix B — Input floorplans and task specifications (Cases 1–45)</b></summary>
<br>

Each case lists the input floorplan image together with its task specification (number of floors).

<table>
  <tr>
    <td align="center"><b>Cases 1–10</b><br><a href="appendices/appendix_B_page_1.png"><img src="appendices/appendix_B_page_1.png" width="400" alt="Appendix B, Cases 1-10"></a></td>
    <td align="center"><b>Cases 11–20</b><br><a href="appendices/appendix_B_page_2.png"><img src="appendices/appendix_B_page_2.png" width="400" alt="Appendix B, Cases 11-20"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 21–30</b><br><a href="appendices/appendix_B_page_3.png"><img src="appendices/appendix_B_page_3.png" width="400" alt="Appendix B, Cases 21-30"></a></td>
    <td align="center"><b>Cases 31–40</b><br><a href="appendices/appendix_B_page_4.png"><img src="appendices/appendix_B_page_4.png" width="400" alt="Appendix B, Cases 31-40"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 41–45</b><br><a href="appendices/appendix_B_page_5.png"><img src="appendices/appendix_B_page_5.png" width="400" alt="Appendix B, Cases 41-45"></a></td>
    <td></td>
  </tr>
</table>

</details>

<details>
<summary><b>Appendix C — Generated BIM models: BIMgent vs. Claude-Sonnet-4.5 and GPT-5.4 computer-use (Cases 1–45)</b></summary>
<br>

For each case, the shaded and wireframe views of the model generated by BIMgent (Ours) are shown next to those produced by the Claude-Sonnet-4.5 and GPT-5.4 computer-use baselines.

<table>
  <tr>
    <td align="center"><b>Cases 1–3</b><br><a href="appendices/appendix_C_page_1.png"><img src="appendices/appendix_C_page_1.png" width="400" alt="Appendix C, Cases 1-3"></a></td>
    <td align="center"><b>Cases 4–6</b><br><a href="appendices/appendix_C_page_2.png"><img src="appendices/appendix_C_page_2.png" width="400" alt="Appendix C, Cases 4-6"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 7–9</b><br><a href="appendices/appendix_C_page_3.png"><img src="appendices/appendix_C_page_3.png" width="400" alt="Appendix C, Cases 7-9"></a></td>
    <td align="center"><b>Cases 10–12</b><br><a href="appendices/appendix_C_page_4.png"><img src="appendices/appendix_C_page_4.png" width="400" alt="Appendix C, Cases 10-12"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 13–15</b><br><a href="appendices/appendix_C_page_5.png"><img src="appendices/appendix_C_page_5.png" width="400" alt="Appendix C, Cases 13-15"></a></td>
    <td align="center"><b>Cases 16–18</b><br><a href="appendices/appendix_C_page_6.png"><img src="appendices/appendix_C_page_6.png" width="400" alt="Appendix C, Cases 16-18"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 19–21</b><br><a href="appendices/appendix_C_page_7.png"><img src="appendices/appendix_C_page_7.png" width="400" alt="Appendix C, Cases 19-21"></a></td>
    <td align="center"><b>Cases 22–24</b><br><a href="appendices/appendix_C_page_8.png"><img src="appendices/appendix_C_page_8.png" width="400" alt="Appendix C, Cases 22-24"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 25–27</b><br><a href="appendices/appendix_C_page_9.png"><img src="appendices/appendix_C_page_9.png" width="400" alt="Appendix C, Cases 25-27"></a></td>
    <td align="center"><b>Cases 28–30</b><br><a href="appendices/appendix_C_page_10.png"><img src="appendices/appendix_C_page_10.png" width="400" alt="Appendix C, Cases 28-30"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 31–33</b><br><a href="appendices/appendix_C_page_11.png"><img src="appendices/appendix_C_page_11.png" width="400" alt="Appendix C, Cases 31-33"></a></td>
    <td align="center"><b>Cases 34–36</b><br><a href="appendices/appendix_C_page_12.png"><img src="appendices/appendix_C_page_12.png" width="400" alt="Appendix C, Cases 34-36"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 37–39</b><br><a href="appendices/appendix_C_page_13.png"><img src="appendices/appendix_C_page_13.png" width="400" alt="Appendix C, Cases 37-39"></a></td>
    <td align="center"><b>Cases 40–42</b><br><a href="appendices/appendix_C_page_14.png"><img src="appendices/appendix_C_page_14.png" width="400" alt="Appendix C, Cases 40-42"></a></td>
  </tr>
  <tr>
    <td align="center"><b>Cases 43–45</b><br><a href="appendices/appendix_C_page_15.png"><img src="appendices/appendix_C_page_15.png" width="400" alt="Appendix C, Cases 43-45"></a></td>
    <td></td>
  </tr>
</table>

</details>

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{deng2025bimgentautonomousbuildingmodeling,
  title         = {BIMgent: Towards Autonomous Building Modeling via Computer-use Agents},
  author        = {Zihan Deng and Changyu Du and Stavros Nousias and André Borrmann},
  year          = {2025},
  eprint        = {2506.07217},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2506.07217}
}
```
