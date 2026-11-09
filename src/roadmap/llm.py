"""Small helpers for calling local chat models through Ollama.

Two entry points, matching how the roadmap progresses:

* `ask()` sends a prompt with the plain `ollama` client and returns the reply text.
  Stage 1 uses it to focus on prompts before introducing any framework.
* `get_chat_model()` returns a LangChain chat model, used from the LangChain
  lessons onwards.

Reasoning models such as `gpt-oss` "think" before they answer. Thinking tokens
count toward the token limit, so the helpers ask for a low thinking effort by
default and keep the limit generous.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Literal

import ollama

from roadmap.config import get_settings

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

Message = dict[str, str]
Think = bool | Literal["low", "medium", "high"] | None


def get_client() -> ollama.Client:
    """Return an `ollama.Client` for the configured server."""
    return ollama.Client(host=get_settings().ollama_host)


@cache
def supports_thinking(model: str) -> bool:
    """Return True if the model can produce separate "thinking" output."""
    capabilities = get_client().show(model).capabilities or []
    return "thinking" in capabilities


def ask(
    prompt: str | Sequence[Message],
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    think: Think = "low",
    **options,
) -> str:
    """Send a prompt (or a full message list) to a chat model and return the reply text.

    Args:
        prompt: A user prompt string, or a list of `{"role": ..., "content": ...}` messages.
        system: Optional system message, added when `prompt` is a string.
        model: Ollama model name. Defaults to `CHAT_MODEL` (`gpt-oss:20b`).
        temperature: Sampling temperature. 0 gives the most repeatable output.
        max_tokens: Upper limit on generated tokens, thinking included.
        think: Thinking effort for reasoning models ("low", "medium", "high"),
            or True/False. Ignored for models that cannot think.
        **options: Extra Ollama options such as `top_p` or `seed`.
    """
    model = model or get_settings().chat_model
    if isinstance(prompt, str):
        messages: list[Message] = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
    else:
        messages = list(prompt)

    response = get_client().chat(
        model=model,
        messages=messages,
        think=think if supports_thinking(model) else None,
        options={"temperature": temperature, "num_predict": max_tokens, **options},
    )
    return (response.message.content or "").strip()


def get_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    think: Think = "low",
    **kwargs,
) -> ChatOllama:
    """Return a LangChain `ChatOllama` model for the configured Ollama server."""
    from langchain_ollama import ChatOllama

    settings = get_settings()
    model = model or settings.chat_model
    return ChatOllama(
        model=model,
        base_url=settings.ollama_host,
        temperature=temperature,
        num_predict=max_tokens,
        reasoning=think if supports_thinking(model) else None,
        **kwargs,
    )
