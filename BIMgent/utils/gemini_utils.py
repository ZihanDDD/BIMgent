import os
import time
import random
from google import genai
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 10
INITIAL_WAIT = 2.0
MULTIPLIER = 2.0
MAX_WAIT = 120.0
JITTER_RANGE = 1.0


def _load_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(name)
        if v:
            return v
    raise RuntimeError(
        "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env"
    )


class GeminiKeyManager:
    """Single-key Gemini client holder (kept as a thin shim for backwards compat)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._api_key = _load_api_key()
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client


def get_gemini_key_manager() -> GeminiKeyManager:
    return GeminiKeyManager()


def _is_rate_limit_error(err_str: str) -> bool:
    return (
        "429" in err_str
        or "RESOURCE_EXHAUSTED" in err_str
        or "Resource_exhausted" in err_str
    )


def gemini_call_with_retry(client, model, contents, config, max_retries=MAX_RETRIES):
    """Call Gemini with exponential backoff on 429 / quota errors."""
    active_client = client or get_gemini_key_manager().client

    wait = INITIAL_WAIT
    for attempt in range(max_retries):
        try:
            return active_client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            err = str(e)
            if _is_rate_limit_error(err) and attempt < max_retries - 1:
                jitter = random.uniform(0, JITTER_RANGE)
                total_wait = min(wait + jitter, MAX_WAIT)
                print(
                    f"Gemini rate-limited; waiting {total_wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(total_wait)
                wait *= MULTIPLIER
            elif _is_rate_limit_error(err):
                print(f"Rate limited (429). All {max_retries} attempts exhausted.")
                return None
            else:
                print(f"Gemini call failed: {e}")
                return None

    return None
