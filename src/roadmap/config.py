"""Shared settings, read from environment variables and the repo-level `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load `.env` once on import. Existing environment variables win over the file.
load_dotenv(REPO_ROOT / ".env")

DEFAULT_CHAT_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for talking to a local Ollama server."""

    chat_model: str
    ollama_host: str


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(
        chat_model=os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL),
        ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    )
