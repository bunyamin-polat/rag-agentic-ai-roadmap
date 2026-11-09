"""Tests for analyzer.py with a fake Ollama client (no server needed)."""

import json
from types import SimpleNamespace

import pytest
from analyzer import SYSTEM_PROMPT, Analysis, AnalysisError, analyze, list_chat_models

VALID_ANALYSIS = {
    "summary": "The customer was charged twice and wants a refund.",
    "sentiment": "negative",
    "sentiment_score": 10,
    "category": "billing",
    "urgency": "high",
    "reply": "We are sorry about the double charge. We will refund it today.",
}


class FakeClient:
    def __init__(self, content: str, capabilities=("completion", "thinking")):
        self.content = content
        self.capabilities = list(capabilities)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.content), eval_count=42)

    def show(self, model):
        return SimpleNamespace(capabilities=self.capabilities)


def test_analyze_returns_fields_and_timing():
    client = FakeClient(json.dumps(VALID_ANALYSIS))

    result = analyze("I was charged twice!", model="test-model", client=client)

    assert result["model"] == "test-model"
    assert result["output_tokens"] == 42
    assert result["seconds"] >= 0
    for key, value in VALID_ANALYSIS.items():
        assert result[key] == value


def test_analyze_sends_schema_and_prompts():
    client = FakeClient(json.dumps(VALID_ANALYSIS))

    analyze("Where is my package?", model="test-model", client=client)

    call = client.calls[0]
    assert call["format"] == Analysis.model_json_schema()
    assert call["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Where is my package?"},
    ]
    assert call["think"] == "low"
    assert call["options"] == {"temperature": 0}


def test_analyze_skips_thinking_for_models_that_cannot_think():
    client = FakeClient(json.dumps(VALID_ANALYSIS), capabilities=("completion",))

    analyze("Hello", model="test-model", client=client)

    assert client.calls[0]["think"] is None


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        json.dumps({**VALID_ANALYSIS, "sentiment_score": 150}),
        json.dumps({**VALID_ANALYSIS, "category": "weather"}),
    ],
)
def test_analyze_rejects_invalid_output(content):
    with pytest.raises(AnalysisError):
        analyze("Hello", model="test-model", client=FakeClient(content))


def test_list_chat_models_skips_embedding_models():
    class ListingClient:
        def list(self):
            names = ["qwen3:1.7b", "nomic-embed-text:latest", "gpt-oss:20b"]
            return SimpleNamespace(models=[SimpleNamespace(model=name) for name in names])

        def show(self, model):
            capabilities = ["embedding"] if "embed" in model else ["completion"]
            return SimpleNamespace(capabilities=capabilities)

    assert list_chat_models(ListingClient()) == ["gpt-oss:20b", "qwen3:1.7b"]
