from __future__ import annotations
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

def prompt_for_task_description() -> Tuple[Optional[str], str]:
    """Interactively obtain a task description.
    """
    while True:
        print(  # Top‑level choice
            colored(
                "\nWould you like to generate a new floorplan from a text "
                "description, or start from an existing plan?",
                "light_yellow",
            )
        )
        mode = (
            input(colored("Type 'text' or 'existing': ", 'light_green'))
            .strip()
            .lower()
        )

        # ── Text description path ────────────────────────────────────────────
        if mode == "text":
            desc = input(
                colored("Describe the building you have in mind:\n> ", "light_green")
            ).strip()
            path = "no path"
            if desc:
                return path, desc
            print(colored("The description can't be empty. Let's try again.", "light_red"))

        # ── Existing plan path ───────────────────────────────────────────────
        elif mode == "existing":
            path = input(
                colored("Enter the file path to the existing floor plan:\n> ", "light_green")
            ).strip()
            desc = input(
                colored(
                    "Add any additional requirements or constraints "
                    "(press Enter to skip):\n> ",
                    "light_green",
                )
            ).strip()
            if path:
                return path, desc
            print(colored("The file path can't be empty. Let's try again.", "light_red"))

        # ── Invalid choice ───────────────────────────────────────────────────
        else:
            print(
                colored("Sorry, I didn't understand. Please type 'text' or 'existing'.", "light_red")
            )


def write_task_details(env_cfg_path: Path, desc: str, floorplan_path: Optional[str]) -> None:
    """
    If floorplan_path is provided, the value of
    floorplan_image_path.floorplan is replaced.
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
    """Determine and run the environment‑specific runner module."""
    runner_key = (
        config.env_shared_runner.lower() if config.env_shared_runner else config.env_short_name.lower()
    )
    runner_module = importlib.import_module(f"BIMgent.runner.{runner_key}_runner")
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
        help="Optional path to an existing floorplan image (overrides interactive prompt).",
    )
    return parser


# ---------------------------------------------------------------------------
# Script entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # Obtain description + (optional) path from CLI or prompt
    if args.taskDescription:
        floorplan_path = args.floorplanPath
        task_desc = args.taskDescription
    else:
        floorplan_path, task_desc = prompt_for_task_description()

    # Persist details to the JSON config *before* loading it
    env_cfg_path = Path(args.envConfig).expanduser()
    write_task_details(env_cfg_path, task_desc, floorplan_path)

    # Load the updated configuration and continue
    config.load_env_config(str(env_cfg_path))
    config.set_fixed_seed()

    print(colored("\n✔  Configuration updated", "cyan"))
    print(colored("🚀  BIM‑GUI AGENT STARTING...", "cyan"))

    # Hand off to the environment‑specific runner
    main(args)
