"""Tests for the Gradio handlers in app.py, with a fake ProfileBot: no Ollama server needed."""

import app
import pytest
from llama_index.core.llms import ChatMessage


class FakeBot:
    def __init__(self):
        self.profile = {
            "name": "Lena Okafor",
            "headline": "Platform Engineer at Brightleaf",
            "location": "Lisbon, Portugal",
            "skills": ["Kubernetes", "Go"],
        }
        self.seen_history = None

    def icebreakers(self):
        return "1. What was your favorite marathon?"

    def stream_chat(self, question, history):
        self.seen_history = history
        return ["summary", "career", "summary"], iter(["She works", " at Brightleaf."])


@pytest.fixture
def fake_bot(monkeypatch):
    bot = FakeBot()
    monkeypatch.setattr(app, "get_bot", lambda name: bot)
    monkeypatch.setattr(app, "_icebreakers", {})
    return bot


def test_select_profile_shows_placeholder_then_icebreakers(fake_bot):
    updates = list(app.select_profile("Lena Okafor"))

    assert len(updates) == 2
    card, placeholder, _ = updates[0]
    assert "Lena Okafor" in card and "Thinking" in placeholder
    assert updates[1][1] == "1. What was your favorite marathon?"


def test_select_profile_reuses_icebreakers(fake_bot):
    list(app.select_profile("Lena Okafor"))

    updates = list(app.select_profile("Lena Okafor"))

    assert len(updates) == 1  # no placeholder the second time


def test_to_llama_skips_source_boxes():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "Where does she work?"}]},
        {"role": "assistant", "content": "At Brightleaf."},
        {"role": "assistant", "content": "summary", "metadata": {"title": app.SOURCES_TITLE}},
    ]

    messages = app.to_llama(history)

    assert [(m.role.value, m.content) for m in messages] == [
        ("user", "Where does she work?"),
        ("assistant", "At Brightleaf."),
    ]
    assert all(isinstance(m, ChatMessage) for m in messages)


def test_respond_streams_answer_then_unique_sections(fake_bot):
    updates = list(app.respond("Where does she work?", [], "Lena Okafor"))

    chat, textbox = updates[-1]
    assert textbox == ""
    assert chat[1] == {"role": "assistant", "content": "She works at Brightleaf."}
    assert chat[2].content == "summary, career"
    assert chat[2].metadata["title"] == app.SOURCES_TITLE


def test_respond_asks_for_a_profile_first(fake_bot):
    [(chat, _)] = list(app.respond("Hello?", [], None))

    assert chat[-1]["content"] == "Pick a profile first."
