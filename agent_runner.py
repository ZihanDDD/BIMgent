from __future__ import annotations

# Silence TensorFlow / oneDNN / abseil noise. Must be set before tensorflow import.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path
import argparse
import importlib
import json
from typing import Optional, Tuple

from termcolor import colored

from conf.config import Config

config = Config()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prompt_for_task_description() -> Tuple[str, str]:
    """Interactively obtain the floorplan image path and floor count.

    Returns
    -------
    Tuple[str, str]
        ``(floorplan_path, description)`` — the path to an existing floorplan
        image, and a task description stating how many floors the building has.
    """
    print(
        colored(
            "\nThis agent builds a BIM model from an existing floorplan image.",
            "light_yellow",
        )
    )

    # ── Floorplan image path ────────────────────────────────────────────────
    while True:
        path = input(
            colored("Enter the file path to the existing floorplan image:\n> ", "light_green")
        ).strip().strip('"').strip("'")
        if path:
            break
        print(colored("The file path can't be empty. Let's try again.", "light_red"))

    # ── Number of floors ────────────────────────────────────────────────────
    while True:
        floors_raw = input(
            colored("How many floors does the building have (based on the layout)?\n> ", "light_green")
        ).strip()
        try:
            n_floors = int(floors_raw)
            if n_floors >= 1:
                break
        except ValueError:
            pass
        print(colored("Please enter a positive integer (e.g. 1, 2, 3).", "light_red"))

    desc = f"The building has {n_floors} floor{'s' if n_floors > 1 else ''} based on the provided layout."
    return path, desc


def write_task_details(env_cfg_path: Path, desc: str, floorplan_path: Optional[str]) -> None:
    """Persist *desc* (and optionally *floorplan_path*) to the env‑config JSON.

    If *floorplan_path* is provided, the value of
    ``floorplan_image_path.floorplan`` is **replaced**.
    """
    with env_cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure task_description_list exists and is long enough (task_id=1)
    task_list = data.setdefault("task_description_list", [])
    if not task_list:
        task_list.append({})
    task_list[0]["task_description"] = desc

    # Update (or create) floorplan_image_path if a path was supplied
    if floorplan_path:
        data.setdefault("floorplan_image_path", {})["floorplan"] = floorplan_path

    with env_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Determine and run the environment‑specific *runner* module."""
    runner_key = (
        config.env_shared_runner.lower() if config.env_shared_runner else config.env_short_name.lower()
    )
    runner_module = importlib.import_module(f"bim_gui_agent.runner.{runner_key}_runner")
    runner_module.entry(args)


def get_args_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser("BIM‑GUI Agent Runner")
    parser.add_argument(
        "--envConfig",
        type=str,
        default="./conf/env_config_vectorworks.json",
        help="Path to the environment‑config JSON file.",
    )
    parser.add_argument(
        "-t",
        "--taskDescription",
        type=str,
        help="Task description to store in the env‑config. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "-p",
        "--floorplanPath",
        type=str,
        help="Optional path to an existing floor‑plan image (overrides interactive prompt).",
    )
    return parser


# ---------------------------------------------------------------------------
# Script entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    env_cfg_path = Path(args.envConfig).expanduser()

    # Determine floorplan path and task description
    if args.floorplanPath:
        floorplan_path = args.floorplanPath
        task_desc = args.taskDescription or "Please create a building based on the provided floorplan"
    else:
        # Interactive prompt
        floorplan_path, task_desc = prompt_for_task_description()
        if not task_desc:
            task_desc = "Please create a building based on the provided floorplan"

    # Persist details to the JSON config
    write_task_details(env_cfg_path, task_desc, floorplan_path)

    # Load the updated configuration
    config.load_env_config(str(env_cfg_path))
    config.set_fixed_seed()

    print(colored("\n✔  Configuration updated", "cyan"))
    print(colored(f"   Floorplan: {floorplan_path}", "cyan"))
    print(colored(f"   Task: {task_desc}", "cyan"))
    print(colored("🚀  BIM‑GUI AGENT STARTING...", "cyan"))

    # Hand off to the environment-specific runner
    main(args)
