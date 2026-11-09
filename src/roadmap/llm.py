"""Small helpers for calling chat models on Hugging Face Inference Providers.

Two entry points, matching how the roadmap progresses:

* `ask()` sends a prompt through the plain `huggingface_hub.InferenceClient`.
  Stage 1 uses it to focus on prompts before introducing any framework.
* `get_chat_model()` returns a LangChain chat model, used from the LangChain
  lessons onwards.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from huggingface_hub import InferenceClient

from roadmap.config import get_settings

if TYPE_CHECKING:
    from langchain_huggingface import ChatHuggingFace

Message = dict[str, str]


def get_client(model: str | None = None, provider: str | None = None) -> InferenceClient:
    """Return an `InferenceClient` bound to a chat model and provider."""
    settings = get_settings()
    return InferenceClient(
        model=model or settings.chat_model,
        provider=provider or settings.provider,  # pyright: ignore[reportArgumentType]
        token=settings.hf_token,
    )


def ask(
    prompt: str | Sequence[Message],
    *,
    system: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **params,
) -> str:
    """Send a prompt (or a full message list) to a chat model and return the reply text.

    Args:
        prompt: A user prompt string, or a list of `{"role": ..., "content": ...}` messages.
        system: Optional system message, prepended when `prompt` is a string.
        model: Model id on the Hugging Face Hub. Defaults to `HF_CHAT_MODEL`.
        provider: Inference provider name, or "auto". Defaults to `HF_PROVIDER`.
        max_tokens: Upper bound on generated tokens.
        temperature: Sampling temperature. 0 gives the most deterministic output.
        **params: Extra chat-completion parameters such as `top_p` or `seed`.
    """
    if isinstance(prompt, str):
        messages: list[Message] = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
    else:
        messages = list(prompt)

    response = get_client(model, provider).chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **params,
    )
    return (response.choices[0].message.content or "").strip()


def get_chat_model(
    model: str | None = None,
    provider: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    **kwargs,
) -> ChatHuggingFace:
    """Return a LangChain `ChatHuggingFace` model backed by Inference Providers."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    settings = get_settings()
    endpoint = HuggingFaceEndpoint(
        repo_id=model or settings.chat_model,
        provider=provider or settings.provider,
        task="conversational",
        temperature=temperature,
        max_new_tokens=max_tokens,
        huggingfacehub_api_token=settings.hf_token,
        **kwargs,
    )
    return ChatHuggingFace(llm=endpoint)
