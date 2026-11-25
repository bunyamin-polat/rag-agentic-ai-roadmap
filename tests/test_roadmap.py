"""Offline tests for the shared `roadmap` helpers. No Ollama server is needed."""

from types import SimpleNamespace

import ollama
import pytest

from roadmap import config, llm


class FakeClient:
    """Records `chat()` calls and returns a fixed reply."""

    def __init__(self, reply: str = "Paris"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=f"  {self.reply}\n"))


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: client)
    monkeypatch.setattr(llm, "supports_thinking", lambda model: True)
    return client


def test_settings_defaults(monkeypatch):
    for name in ("CHAT_MODEL", "EMBED_MODEL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)

    settings = config.get_settings()

    assert settings.chat_model == config.DEFAULT_CHAT_MODEL
    assert settings.embed_model == config.DEFAULT_EMBED_MODEL
    assert settings.ollama_host == config.DEFAULT_OLLAMA_HOST


def test_settings_read_environment(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "qwen3:1.7b")
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")

    settings = config.get_settings()

    assert settings == config.Settings(
        chat_model="qwen3:1.7b", embed_model="embeddinggemma", ollama_host="http://gpu-box:11434"
    )


def test_get_embeddings_uses_settings(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "embeddinggemma")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")

    embeddings = llm.get_embeddings()

    assert embeddings.model == "embeddinggemma"
    assert embeddings.base_url == "http://gpu-box:11434"


def test_ask_sends_system_and_user_messages(fake_client):
    answer = llm.ask(
        "Capital of France?",
        system="Be brief.",
        model="test-model",
        temperature=0.5,
        max_tokens=50,
        seed=1,
    )

    assert answer == "Paris"
    call = fake_client.calls[0]
    assert call["model"] == "test-model"
    assert call["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"},
    ]
    assert call["options"] == {"temperature": 0.5, "num_predict": 50, "seed": 1}
    assert call["think"] == "low"


def test_ask_accepts_a_message_list(fake_client):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What did I say first?"},
    ]

    llm.ask(messages, model="test-model")

    assert fake_client.calls[0]["messages"] == messages


def test_ask_skips_think_for_models_that_cannot_think(fake_client, monkeypatch):
    monkeypatch.setattr(llm, "supports_thinking", lambda model: False)

    llm.ask("Hi", model="test-model")

    assert fake_client.calls[0]["think"] is None


def test_get_chat_model_uses_settings(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "qwen3:1.7b")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    monkeypatch.setattr(llm, "supports_thinking", lambda model: False)

    model = llm.get_chat_model(temperature=0.2, max_tokens=64)

    assert model.model == "qwen3:1.7b"
    assert model.base_url == "http://gpu-box:11434"
    assert model.temperature == 0.2
    assert model.num_predict == 64
    assert model.reasoning is None


def test_explain_ollama_error_names_the_missing_model():
    error = ollama.ResponseError("model 'qwen3:4b' not found", 404)

    message = llm.explain_ollama_error(error)

    assert message == "⚠️ The model qwen3:4b is not downloaded. Try: `ollama pull qwen3:4b`"


def test_explain_ollama_error_for_other_errors():
    busy = ollama.ResponseError("busy")
    assert "Ollama returned an error: busy" in llm.explain_ollama_error(busy)
    assert "`ollama serve`" in llm.explain_ollama_error(ConnectionError("refused"))
    assert all(issubclass(kind, Exception) for kind in llm.OLLAMA_ERRORS)
