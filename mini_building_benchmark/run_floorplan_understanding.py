"""
Batch floorplan understanding evaluation.

Processes all 45 floorplans (1floor/2floor/3floor × cubicasa1-15):
  1. Resize image to 512×512
  2. Run DeepFloorplan segmentation (server at localhost:8888)
  3. Run Gemini design interpreter for wall/opening refinement + classification
  4. Save results (JSON + images) to mini_building_benchmark/floorplan_understanding/

Prerequisites:
  - DeepFloorplan server running at localhost:8888
  - GEMINI_API_KEY (or GOOGLE_API_KEY) set in .env

Usage:
  python mini_building_benchmark/run_floorplan_understanding.py              # run all 45
  python mini_building_benchmark/run_floorplan_understanding.py --floor 1floor  # run one floor
  python mini_building_benchmark/run_floorplan_understanding.py --floor 1floor --case cubicasa1  # single
  python mini_building_benchmark/run_floorplan_understanding.py --no-resume   # re-run everything
"""

import os
import sys
import json
import time
import traceback

# Use non-interactive backend for batch processing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Disable plt.pause sleeping in batch mode (saves ~7 min across 45 floorplans)
plt.pause = lambda *a, **kw: None

# Project root so imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from conf.config import Config
from bim_gui_agent.memory.local_memory import LocalMemory
from bim_gui_agent.utils.floorplan_resize import resize_image
from bim_gui_agent.provider.Deep_fp_provider.deepfloorplan_endpoint import run_deepfloorplan
from bim_gui_agent.provider.loop_providers.design_interpreter import (
    DesignInterpreterGeminiProvider,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORI_IMAGES_DIR = os.path.join(PROJECT_ROOT, "mini_building_benchmark", "ori_images")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "mini_building_benchmark", "floorplan_understanding")

FLOORS = ["1floor", "2floor", "3floor"]
NUM_CASES = 15  # cubicasa1 .. cubicasa15

# Delay between Gemini API calls to avoid 429 rate-limit errors (seconds)
API_CALL_DELAY = 2


def is_already_done(out_dir: str) -> bool:
    """Check if this case was already completed (working_process_data.json exists)."""
    return os.path.isfile(os.path.join(out_dir, "working_process_data.json"))


def process_single_floorplan(floor: str, case_name: str):
    """Run the full floorplan understanding pipeline for one image."""

    config = Config()
    memory = LocalMemory()

    image_path = os.path.join(ORI_IMAGES_DIR, floor, f"{case_name}.png")
    if not os.path.exists(image_path):
        print(f"  [SKIP] Image not found: {image_path}")
        return False

    # Output directory for this case
    out_dir = os.path.join(OUTPUT_ROOT, floor, case_name)
    os.makedirs(out_dir, exist_ok=True)

    # --- Reset singleton state for this run ---
    config.work_dir = out_dir
    memory.clear()
    memory.memory_path = out_dir

    # 1. Resize image to 512×512
    memory.update_info_history({"floorplan_path": image_path})
    resize_path = resize_image(image_path)
    memory.update_info_history({"floorplan_path": resize_path})

    # 2. Deep Floorplan segmentation (calls server at localhost:8888)
    walls, openings = run_deepfloorplan()

    # 3. Design interpreter (Gemini 3.1 Pro) — refine walls + classify openings
    interpreter = DesignInterpreterGeminiProvider(task_description="floorplan evaluation")
    floorplan_result = interpreter(walls, openings)

    # 4. Save result JSON
    if isinstance(floorplan_result, list) and len(floorplan_result) > 0:
        result_data = floorplan_result[0]
    else:
        result_data = floorplan_result

    json_path = os.path.join(out_dir, "working_process_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"  [OK] Saved -> {out_dir}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch floorplan understanding evaluation")
    parser.add_argument("--floor", type=str, default=None,
                        help="Process only this floor (e.g. '1floor')")
    parser.add_argument("--case", type=str, default=None,
                        help="Process only this case (e.g. 'cubicasa1')")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-process even if results already exist")
    args = parser.parse_args()

    floors = [args.floor] if args.floor else FLOORS
    cases = [args.case] if args.case else [f"cubicasa{i}" for i in range(1, NUM_CASES + 1)]
    resume = not args.no_resume

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    total = 0
    success = 0
    skipped = 0
    failed = []
    n_total = len(floors) * len(cases)

    for floor in floors:
        for case_name in cases:
            total += 1
            out_dir = os.path.join(OUTPUT_ROOT, floor, case_name)

            # Resume: skip already-completed cases
            if resume and is_already_done(out_dir):
                print(f"  [{total}/{n_total}] {floor}/{case_name} -- already done, skipping")
                skipped += 1
                success += 1
                continue

            print(f"\n{'='*60}")
            print(f"Processing {floor}/{case_name}  ({total}/{n_total})")
            print(f"{'='*60}")

            try:
                ok = process_single_floorplan(floor, case_name)
                if ok:
                    success += 1
                else:
                    failed.append(f"{floor}/{case_name}")
            except Exception as e:
                print(f"  [FAIL] {floor}/{case_name}: {e}")
                traceback.print_exc()
                failed.append(f"{floor}/{case_name}")

            # Small delay between cases to avoid API rate limits
            time.sleep(API_CALL_DELAY)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE — {success}/{n_total} succeeded ({skipped} skipped/resumed)")
    if failed:
        print(f"Failed cases ({len(failed)}): {failed}")
    print(f"Results saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
