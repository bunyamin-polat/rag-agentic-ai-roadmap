"""Shared helpers for the RAG & Agentic AI roadmap notebooks and projects."""

from roadmap.config import get_settings
from roadmap.llm import (
    OLLAMA_ERRORS,
    ask,
    explain_ollama_error,
    get_chat_model,
    get_client,
    get_embeddings,
)

__all__ = [
    "OLLAMA_ERRORS",
    "ask",
    "explain_ollama_error",
    "get_chat_model",
    "get_client",
    "get_embeddings",
    "get_settings",
]
