import json
import csv
import os
from conf.config import Config

config = Config()

class JSONAnalyzer:
    def __init__(self, memory_path, working_process_path):
        self.memory_path = memory_path
        self.working_process_path = working_process_path
        self.output_dir = config.work_dir
        self.output_csv = os.path.join(self.output_dir, "summary.csv")

        # Execution metadata
        self.time = None
        self.api_calls = None

        # New counters
        self.vision_driven_total = 0
        self.vision_driven_approved_1 = 0
        self.vision_driven_approved_0 = 0

        self.pure_action_total = 0
        self.pure_action_approved_1 = 0
        self.pure_action_approved_0 = 0

    def load_and_process(self):
        self._load_memory()
        self._parse_working_process()
        self._write_summary_to_csv()

    def _load_memory(self):
        with open(self.memory_path, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        self.time = memory_data.get("run_time")
        self.api_calls = memory_data.get("api_calling_times")

    def _parse_working_process(self):
        with open(self.working_process_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for step in data.values():
            for key, substep in step.items():
                if not key.startswith("sub_step") or not isinstance(substep, dict):
                    continue

                approved = substep.get("approved_value")
                action_type = substep.get("action_type")

                if action_type == "Vision-Driven":
                    self.vision_driven_total += 1
                    if approved == 1:
                        self.vision_driven_approved_1 += 1
                    elif approved == 0:
                        self.vision_driven_approved_0 += 1

                elif action_type == "Pure-Action":
                    self.pure_action_total += 1
                    if approved == 1:
                        self.pure_action_approved_1 += 1
                    elif approved == 0:
                        self.pure_action_approved_0 += 1

    def _write_summary_to_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        file_exists = os.path.isfile(self.output_csv)

        fieldnames = [
            "location", "time", "api_calls",
            "vision_driven_total", "vision_driven_approved_1", "vision_driven_approved_0",
            "pure_action_total", "pure_action_approved_1", "pure_action_approved_0"
        ]

        with open(self.output_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "location": self.output_dir,
                "time": self.time,
                "api_calls": self.api_calls,
                "vision_driven_total": self.vision_driven_total,
                "vision_driven_approved_1": self.vision_driven_approved_1,
                "vision_driven_approved_0": self.vision_driven_approved_0,
                "pure_action_total": self.pure_action_total,
                "pure_action_approved_1": self.pure_action_approved_1,
                "pure_action_approved_0": self.pure_action_approved_0,
            })

# Example usage:
# analyzer = JSONAnalyzer("memory.json", "working_process.json")
# analyzer.load_and_process()
