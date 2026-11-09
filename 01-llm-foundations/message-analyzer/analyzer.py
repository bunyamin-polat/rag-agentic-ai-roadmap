"""Analyse a customer message with a local LLM and get back structured fields.

The model must answer with JSON that follows the `Analysis` schema. Ollama forces the
output to match the schema while the model writes it (structured output), and Pydantic
checks the result before we use it.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import ollama
from pydantic import BaseModel, Field, ValidationError

from roadmap import get_client, get_settings

MAX_MESSAGE_CHARS = 4000


class Analysis(BaseModel):
    """The fields we want for every customer message."""

    summary: str = Field(description="One short sentence that says what the customer wants.")
    sentiment: Literal["positive", "neutral", "negative"]
    sentiment_score: int = Field(
        ge=0, le=100, description="0 = very negative, 50 = neutral, 100 = very positive."
    )
    category: Literal["billing", "delivery", "product", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    reply: str = Field(description="A friendly reply to the customer, 2 to 4 sentences.")


SYSTEM_PROMPT = f"""You help a customer support team. Read the customer's message and answer
with JSON that follows this schema:

{json.dumps(Analysis.model_json_schema(), indent=2)}

Rules:
- summary: one short sentence.
- sentiment_score: 0 is very negative, 50 is neutral, 100 is very positive.
- urgency is high when the customer is blocked, loses money, or wants to leave.
- reply: 2 to 4 friendly sentences. Do not invent names, order details, phone numbers,
  or email addresses, and do not use placeholders such as [Name].
"""


class AnalysisError(RuntimeError):
    """The model did not return a valid analysis."""


def thinking_setting(client: ollama.Client, model: str) -> str | None:
    """Ask reasoning models for short thinking; send nothing to other models."""
    capabilities = client.show(model).capabilities or []
    return "low" if "thinking" in capabilities else None


def analyze(
    message: str, model: str | None = None, client: ollama.Client | None = None
) -> dict[str, Any]:
    """Analyse one message with one model.

    Returns the `Analysis` fields plus the model name, the time it took, and the number of
    generated tokens.
    """
    client = client or get_client()
    model = model or get_settings().chat_model

    start = time.perf_counter()
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        format=Analysis.model_json_schema(),
        think=thinking_setting(client, model),
        options={"temperature": 0},
    )
    seconds = time.perf_counter() - start

    try:
        analysis = Analysis.model_validate_json(response.message.content or "")
    except ValidationError as error:
        raise AnalysisError(f"{model} did not return a valid analysis.") from error

    return {
        "model": model,
        "seconds": round(seconds, 1),
        "output_tokens": response.eval_count,
        **analysis.model_dump(),
    }


def list_chat_models(client: ollama.Client | None = None) -> list[str]:
    """Return the installed Ollama models that can chat (embedding models are skipped)."""
    client = client or get_client()
    names = sorted(model.model for model in client.list().models if model.model)
    return [name for name in names if "completion" in (client.show(name).capabilities or [])]
