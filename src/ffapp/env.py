"""Loads secrets and config flags (FFAPP_OFFLINE, FFAPP_CACHE_STRICT, API keys) from
.env into os.environ. CLAUDE.md: secrets come from os.environ only, loaded from .env.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from ffapp.config import REPO_ROOT

DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_env(path: Path = DEFAULT_ENV_PATH) -> None:
    if path.exists():
        load_dotenv(path)
