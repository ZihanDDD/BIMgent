import os
from datetime import datetime

from dotenv import load_dotenv

from BIMgent.utils.singleton import Singleton
from BIMgent.utils.json_utils import load_json
from BIMgent.utils.dict_utils import kget


load_dotenv(verbose=True)


class Config(metaclass=Singleton):
    """Process-wide configuration: the loaded env-config JSON plus the run directory."""

    def __init__(self):
        self.env_config = None
        self.env_name = "-"
        self.env_short_name = "-"
        self._set_dirs()

    def load_env_config(self, env_config_path):
        """Load environment-specific configuration from a JSON file."""
        if not os.path.exists(env_config_path):
            raise FileNotFoundError(f"Config file not found: {env_config_path}")

        self.env_config = load_json(env_config_path)
        self.env_name = kget(self.env_config, 'env_name', default='')
        self.env_short_name = kget(self.env_config, 'env_short_name', default='')

    def _set_dirs(self) -> None:
        """Create the per-run working directory (named with a timestamp)."""
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(os.getcwd(), 'runs', run_name)
        os.makedirs(run_dir, exist_ok=True)
        self.work_dir = run_dir
