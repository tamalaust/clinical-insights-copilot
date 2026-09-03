"""
Required env vars for the agentic app layer. Same "no fallback
defaults, fail loudly" pattern as etl/config.py — kept as a small
separate helper here rather than importing etl's private _require_env,
since these are two independently-runnable parts of the monolith.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and set it."
        )
    return value


OLLAMA_BASE_URL = _require_env("OLLAMA_BASE_URL")
OLLAMA_MODEL = _require_env("OLLAMA_MODEL")
FASTAPI_BASE_URL = _require_env("FASTAPI_BASE_URL")
REDIS_URL = _require_env("REDIS_URL")
