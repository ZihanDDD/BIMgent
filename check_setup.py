"""Pre-flight check for BIMgent.

Run this from the repository root before `python agent_runner.py`. It only
uses the standard library plus whatever optional packages are installed, so
it works even on a half-configured environment and tells you what is missing.

    python check_setup.py
    python check_setup.py --envConfig conf/env_config_vectorworks.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
_problems = 0


def report(status: str, msg: str) -> None:
    global _problems
    if status == FAIL:
        _problems += 1
    print(f"[{status}] {msg}")


def check_python() -> None:
    v = sys.version_info
    if v.major == 3 and v.minor == 10:
        report(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        report(WARN, f"Python {v.major}.{v.minor}.{v.micro} — the project targets 3.10; "
                     "TensorFlow/PaddleOCR wheels may not be available for other versions")


def check_cwd() -> None:
    if Path("res/vectorworks/prompts").is_dir() and Path("agent_runner.py").is_file():
        report(OK, f"Running from repository root: {os.getcwd()}")
    else:
        report(FAIL, "Not running from the repository root. Prompts and the RAG DB are "
                     "resolved relative to the CWD — `cd` into the repo first.")


def check_env_keys() -> None:
    env = Path(".env")
    if not env.is_file():
        report(FAIL, ".env not found — `cp .env.example .env` and fill in your keys")
        return
    values: dict[str, str] = {}
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")

    placeholders = {"", "your_openai_api_key", "your_gemini_api_key"}
    for key, purpose in (("OA_OPENAI_KEY", "OpenAI embeddings for the builder RAG"),
                         ("GEMINI_API_KEY", "Gemini planning / vision / interpretation")):
        val = values.get(key) or (values.get("GOOGLE_API_KEY") if key == "GEMINI_API_KEY" else None)
        if not val or val in placeholders:
            report(FAIL, f"{key} missing or still a placeholder in .env ({purpose})")
        else:
            report(OK, f"{key} set ({val[:4]}…, {len(val)} chars)")


def check_env_config(path: Path) -> dict | None:
    if not path.is_file():
        report(FAIL, f"{path} not found — `cp conf/env_config_vectorworks.example.json {path}`")
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        report(FAIL, f"{path} is not valid JSON: {e}")
        return None
    report(OK, f"Loaded {path}")

    exe = cfg.get("env_path", "")
    if exe and Path(exe).is_file():
        report(OK, f"Vectorworks executable found: {exe}")
    else:
        report(WARN, f"Vectorworks executable not found at env_path={exe!r} "
                     "(informational only — start Vectorworks manually before running the agent)")

    models = cfg.get("models_path", {})
    checks = {
        "deep_floorplan": ("dir", ["pretrained_r3d.meta", "pretrained_r3d.index"]),
        "omini": ("file", []),
        "Florence2": ("dir", ["config.json"]),
    }
    for key, (kind, required) in checks.items():
        p = models.get(key, "")
        if not p or "<path>" in p:
            report(FAIL, f"models_path.{key} still has the placeholder value — point it at the downloaded weights")
            continue
        pp = Path(p)
        if kind == "file" and not pp.is_file():
            report(FAIL, f"models_path.{key}: file not found: {p}")
        elif kind == "dir" and not pp.is_dir():
            report(FAIL, f"models_path.{key}: directory not found: {p}")
        else:
            missing = [f for f in required if not (pp / f).exists()]
            if missing:
                report(FAIL, f"models_path.{key}: {p} is missing {missing}")
            else:
                report(OK, f"models_path.{key}: {p}")

    panels = cfg.get("panel_coordinates", {})
    for name in ("design_panel", "tool_panel", "object_info", "whole_panel"):
        box = panels.get(name)
        if not (isinstance(box, list) and len(box) == 4 and box[0] < box[2] and box[1] < box[3]):
            report(FAIL, f"panel_coordinates.{name} must be [x1, y1, x2, y2] with x1<x2, y1<y2 (got {box})")
    return cfg


def check_screen(cfg: dict | None) -> None:
    try:
        import pyautogui  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        report(WARN, f"pyautogui not importable ({e}); cannot verify screen size")
        return
    w, h = pyautogui.size()
    whole = (cfg or {}).get("panel_coordinates", {}).get("whole_panel", [0, 0, 1920, 1080])
    if (w, h) == (whole[2], whole[3]):
        report(OK, f"Screen size {w}x{h} matches whole_panel")
    else:
        report(WARN, f"Screen size {w}x{h} but whole_panel is {whole[2]}x{whole[3]} — "
                     "re-calibrate panel_coordinates (and set Windows display scaling to 100%)")


def check_imports() -> None:
    mods = [
        ("google.genai", "google-genai"), ("dotenv", "python-dotenv"), ("termcolor", "termcolor"),
        ("pyautogui", "pyautogui"), ("cv2", "opencv-python"), ("PIL", "Pillow"), ("imageio", "imageio"),
        ("skimage", "scikit-image"), ("matplotlib", "matplotlib"), ("tensorflow", "tensorflow"),
        ("torch", "torch"), ("torchvision", "torchvision"), ("transformers", "transformers"),
        ("ultralytics", "ultralytics"), ("supervision", "supervision"), ("paddleocr", "paddleocr"),
        ("einops", "einops"), ("timm", "timm"), ("langchain_openai", "langchain_openai"),
        ("langchain_chroma", "langchain_chroma"), ("langchain_community", "langchain_community"),
    ]
    missing = []
    for mod, pkg in mods:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        report(FAIL, "Missing packages: " + ", ".join(missing) + "  →  pip install -r requirements.txt")
    else:
        report(OK, f"All {len(mods)} required packages import")

    try:
        import transformers
        major = int(transformers.__version__.split(".")[0])
        if major >= 5:
            report(FAIL, f"transformers {transformers.__version__} — Florence-2 remote code needs 4.x: "
                         'pip install "transformers>=4.49,<5"')
        else:
            report(OK, f"transformers {transformers.__version__}")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            report(OK, f"CUDA available for OmniParser: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
        else:
            report(WARN, f"No CUDA device (torch {torch.__version__}) — OmniParser (YOLO + Florence-2) will run on CPU "
                         "and be slow. For NVIDIA GPUs: pip uninstall torch torchvision && "
                         "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="BIMgent pre-flight check")
    parser.add_argument("--envConfig", default="conf/env_config_vectorworks.json")
    args = parser.parse_args()

    check_python()
    check_cwd()
    check_env_keys()
    cfg = check_env_config(Path(args.envConfig))
    check_screen(cfg)
    check_imports()

    print()
    if _problems:
        print(f"{_problems} blocking problem(s) found — fix the [FAIL] lines above, then re-run.")
        return 1
    print("Ready. Open Vectorworks with an empty document, then run: python agent_runner.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
