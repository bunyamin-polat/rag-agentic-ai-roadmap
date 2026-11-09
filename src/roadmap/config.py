"""Shared settings, read from environment variables and the repo-level `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load `.env` once on import. Existing environment variables win over the file.
load_dotenv(REPO_ROOT / ".env")

DEFAULT_CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PROVIDER = "auto"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for talking to Hugging Face Inference Providers."""

    hf_token: str | None
    chat_model: str
    provider: str


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        chat_model=os.getenv("HF_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        provider=os.getenv("HF_PROVIDER", DEFAULT_PROVIDER),
    )
