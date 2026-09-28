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
├── check_setup.py                  # Pre-flight check: keys, model paths, panel coordinates, packages
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

## ⚡ Quick Start

```bash
git clone https://github.com/ZihanDDD/BIMgent_private_sourcecode.git BIMgent && cd BIMgent
py -3.10 -m venv .venv && .venv\Scripts\activate      # Windows; on macOS/Linux: python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                                                  # 1. add your API keys
cp conf/env_config_vectorworks.example.json conf/env_config_vectorworks.json   # 2. set model paths + panel coordinates
python check_setup.py                                                 # 3. verify everything before touching Vectorworks

# Open Vectorworks 2025 with an empty document, then:
python agent_runner.py --floorplanPath mini_building_benchmark/scaled_images/1floor/cubicasa1.png \
                       --taskDescription "The building has 1 floor based on the provided layout."
```

`check_setup.py` validates the Python version, `.env` keys, model paths, panel coordinates, screen size and package imports, and prints exactly what still needs fixing. The detailed steps behind each line are below.

## 🔧 Setup

**Requirements at a glance**

| | |
| --- | --- |
| OS | Windows 10/11 (the agent drives the Vectorworks desktop GUI via PyAutoGUI; shortcuts and dialogs are Windows-specific) |
| Display | 1920×1080, Windows display scaling **100%** — all click coordinates are absolute screen pixels |
| Python | 3.10 |
| GPU | Optional but recommended (CUDA). DeepFloorplan falls back to CPU automatically; OmniParser (YOLO + Florence-2) runs on CPU too but is several times slower per screenshot |
| Accounts | Vectorworks 2025 license, Google Gemini API key, OpenAI API key |
| Network | Needed at runtime for the LLM calls, and on the **first run** to download the Florence-2 processor and PaddleOCR models |

### 1. Install Vectorworks
The current release targets **Vectorworks 2025**. Install it; the agent does not launch Vectorworks itself, so you will start it manually with an empty document before each run.

### 2. Python environment
```bash
# Windows (Python 3.10 must be installed; `py -0` lists available versions)
py -3.10 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```
`.venv/` is git-ignored.
The heavy dependencies are TensorFlow (DeepFloorplan), PyTorch + transformers + ultralytics (OmniParser) and PaddleOCR. If you want GPU inference, install a CUDA build of `torch`/`torchvision` for your driver first (see [pytorch.org](https://pytorch.org/get-started/locally/)), then run the `pip install` above.

### 3. API keys
BIMgent uses **Google Gemini** for planning, vision and floorplan interpretation, and **OpenAI embeddings** (`text-embedding-3-large`) to index the builder documentation for RAG. Copy the template and fill in both keys:
```bash
cp .env.example .env
```
```env
OA_OPENAI_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key      # GOOGLE_API_KEY is accepted as a fallback
```
`.env` is git-ignored — never commit it. The Gemini model ids used for each stage are class attributes on [`LLMProvider`](BIMgent/provider/loop_providers/llm_provider.py) (`MODEL_PLANNING`, `MODEL_VISION`, `MODEL_UNDERSTANDING`); change them there if you want to swap models.

### 4. Environment config
```bash
cp conf/env_config_vectorworks.example.json conf/env_config_vectorworks.json
```
Edit `conf/env_config_vectorworks.json`:

| Key | What to put there |
| --- | --- |
| `env_path` | Path to `Vectorworks2025.exe` (informational — the agent expects Vectorworks to be already running, see below) |
| `models_path` | Paths to the downloaded weights (step 5) |
| `panel_coordinates` | Screen rectangles of the Vectorworks panels (step 6) |
| `task_description_list`, `floorplan_image_path` | Filled in automatically by `agent_runner.py` on every run — leave as is |

This file is git-ignored because it holds machine-specific paths and is rewritten on each run.

### 5. Download external models
BIMgent reuses two pretrained models. Download them and update `models_path` in the config.

| Model | Purpose | Where to get it | Expected files |
| --- | --- | --- | --- |
| **DeepFloorplan** | Wall / opening segmentation of the input floorplan | [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) → pretrained checkpoint | a directory containing `pretrained_r3d.meta`, `pretrained_r3d.index`, `pretrained_r3d.data-*` |
| **OmniParser icon detector** | YOLO model that finds clickable UI elements in Vectorworks screenshots | [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) → `icon_detect/model.pt` | single `.pt` file |
| **OmniParser Florence-2 captioner** | Describes each detected element so the LLM can pick the right one | [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0) → `icon_caption_florence/` | a directory with `config.json` + weights |

```json
"models_path": {
  "deep_floorplan": "D:/models/deep_floorplan/pretrained",
  "omini":          "D:/models/omni/weights/icon_detect/model.pt",
  "Florence2":      "D:/models/omni/weights/icon_caption_florence"
}
```
Two more models are fetched automatically on first use and cached by their libraries: the Florence-2 *processor* (`microsoft/Florence-2-base`, from the Hugging Face Hub) and the PaddleOCR English text models.

### 6. Calibrate panel coordinates
The agent needs to know where the Vectorworks panels sit on screen. Open Vectorworks, arrange the workspace the way you will run it (the defaults in the example config assume a maximised window at 1920×1080 with the tool palette on the left and the Object Info palette on the right), then set `panel_coordinates`:

```json
"panel_coordinates": {
  "tool_panel":   [x1, y1, x2, y2],   // Left-hand tool palette
  "design_panel": [x1, y1, x2, y2],   // Central drawing area — the floorplan is scaled to fit inside this box
  "object_info":  [x1, y1, x2, y2],   // Right-hand Object Info palette
  "whole_panel":  [0, 0, 1920, 1080]  // Full screen
}
```
Each rectangle is `[top-left x, top-left y, bottom-right x, bottom-right y]` in screen pixels. `design_panel` matters most: floorplan coordinates are mapped into it, so it must be the empty drawing canvas with no palettes overlapping. To read a coordinate, hover the mouse over the point and run:
```bash
python -c "import pyautogui, time; time.sleep(3); print(pyautogui.position())"
```

### 7. Verify
```bash
python check_setup.py
```
Fix every `[FAIL]` line it prints. `[WARN]` lines (no CUDA, screen size mismatch, Vectorworks path) are informational but worth reading.

## 🚀 Running the Agent

Always run from the repository root — prompts, builder docs and the RAG index (`chroma_db/`) are resolved relative to the working directory.

1. Launch Vectorworks 2025 with an **empty document** and the workspace arranged as calibrated in step 6.
2. Run the agent:
   ```bash
   python agent_runner.py
   ```
   You will be prompted for the path to a floorplan image (a real photo, a rendered floorplan, or a hand-drawn sketch) and the number of floors. Non-interactive:
   ```bash
   python agent_runner.py \
     --envConfig ./conf/env_config_vectorworks.json \
     --floorplanPath /path/to/floorplan.png \
     --taskDescription "The building has 2 floors based on the provided layout."
   ```
3. **Keep your hands off the mouse and keyboard** once the builders start — the agent is driving the real desktop. To abort, press `Ctrl+C` in the terminal; the agent yields control between actions.

What happens during a run:
1. **Floorplan understanding** — the image is resized to 512×512, segmented by DeepFloorplan, geometrically cleaned, and refined by the Gemini Design Interpreter. A few matplotlib windows pop up for ~3 s each so you can sanity-check the extracted walls and openings.
2. **Hierarchical planning** — the high-level planner produces the step list; for each step the builder documentation is retrieved from `chroma_db/` (embedded automatically on the first run, takes a few seconds) and the low-level planner produces sub-steps.
3. **Execution** — each sub-step is either replayed as precomputed actions or handled by the vision-driven builder (screenshot → OmniParser grounding → action generation → supervisor check, up to 3 attempts).

Every run writes to `runs/run_<timestamp>/`:

| File | Content |
| --- | --- |
| `resized_floorplan.png`, `segmented_floorplan.png`, `cleaned.png`, `postprocessed_floorplan_visualization.png` | Floorplan-understanding stages |
| `openings/` | Crops of every detected opening sent to the door/window classifier |
| `working_process_data.json` | Floorplan metadata, high-/low-level plans, per-sub-step actions, supervisor reasoning, screenshot paths and total runtime |
| `memory.json` | Final shared working area |
| `screenshots/` | Every screenshot taken, plus the masked / OmniParser-annotated variants |

The main orchestration logic lives in [`BIMgent/runner/vectorworks_runner.py`](BIMgent/runner/vectorworks_runner.py).

### Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `No Gemini API key found` at startup | `.env` missing or key still a placeholder — run `check_setup.py` |
| `Pretrained model not found in '...'` | `models_path.deep_floorplan` must point at the directory that contains `pretrained_r3d.*` |
| `FileNotFoundError: res/vectorworks/prompts/...` | Not running from the repository root |
| Clicks land in the wrong place | Display scaling is not 100%, or `panel_coordinates` no longer match the Vectorworks layout |
| Long pause on the first vision-driven step | Florence-2 processor / PaddleOCR models being downloaded and loaded; subsequent steps are faster |
| Supervisor keeps rejecting a step | Check the `_seg_*.png` screenshot in `screenshots/` — if OmniParser found no elements, the popup diff probably masked the wrong region |

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
