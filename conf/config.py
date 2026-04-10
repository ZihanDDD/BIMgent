import os
import json
from datetime import datetime
from bim_gui_agent.utils.singleton import Singleton
from dotenv import load_dotenv
from bim_gui_agent.utils.json_utils import load_json
from bim_gui_agent.utils.dict_utils import kget


load_dotenv(verbose=True)



# Using Singleton pattern for config to have always the same object across all modules
class Config(metaclass = Singleton):
    """Configuration class for managing environment-specific settings."""

    DEFAULT_FIXED_SEED_VALUE = 42
    DEFAULT_FIXED_TEMPERATURE_VALUE = 0.0

    def __init__(self):
        """Initialize the configuration object with default values."""
        self.env_config = None  # Initialize env_config attribute
        self.env_name = "-"
        self.env_sub_path = "-"
        self.env_short_name = "-"
        self.env_shared_runner = None


        # Default parameters
        self.fixed_seed = False
        self.seed = None
        self.temperature = 1.0
        self.max_recent_steps = 5
        
        if self.fixed_seed:
            self.set_fixed_seed()


        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))


        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))



        self._set_dirs()

    def set_env_name(self, env_name: str) -> None:
        """Set the environment name."""
        self.env_name = env_name

    def load_env_config(self, env_config_path):
        """Load environment-specific configuration from a JSON file."""
        if not os.path.exists(env_config_path):
            raise FileNotFoundError(f"Config file not found: {env_config_path}")
        
        # Load JSON and set env_config
        self.env_config = load_json(env_config_path)

        # Extract environment-specific parameters
        self.env_name = kget(self.env_config, 'env_name', default='')
        self.env_sub_path = kget(self.env_config, 'sub_path', default='')
        self.env_short_name = kget(self.env_config, 'env_short_name', default='')
        self.provider_configs = kget(self.env_config, 'provider_configs', default={})


    def set_fixed_seed(self, is_fixed=True, seed=None, temperature=None):
        """Set fixed seed and temperature for reproducibility."""
        self.fixed_seed = is_fixed
        self.seed = seed or self.DEFAULT_FIXED_SEED_VALUE
        self.temperature = temperature or self.DEFAULT_FIXED_TEMPERATURE_VALUE

    def _set_dirs(self) -> None:
        """Create the per-run working directory (named with a timestamp)."""
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(os.getcwd(), 'runs', run_name)
        os.makedirs(run_dir, exist_ok=True)
        self.work_dir = run_dir

    def set_run_dir(self, folder_name: str) -> None:
        """Override the run directory with a custom name (e.g. batch evals)."""
        run_dir = os.path.join(os.getcwd(), 'runs', folder_name)
        os.makedirs(run_dir, exist_ok=True)
        self.work_dir = run_dir
