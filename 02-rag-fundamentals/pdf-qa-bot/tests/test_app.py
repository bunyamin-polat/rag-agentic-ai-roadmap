"""Tests for the Gradio handlers in app.py, with a fake PdfQA: no Ollama server needed."""

import app
import ollama
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from rag import IndexedPdf, PdfError


class FakeQA:
    def __init__(self):
        self.seen_history = None

    def index(self, path):
        if path.endswith("scan.pdf"):
            raise PdfError("This PDF has no text. Is it a scanned document?")
        return IndexedPdf("abc123", "handbook.pdf", 3, 13)

    def ask(self, file_id, question, history):
        self.seen_history = history
        docs = [Document(page_content="Eating is not allowed.", metadata={"page": 2})]
        return docs, iter(["No", ", only", " water."])


@pytest.fixture
def fake_qa(monkeypatch):
    fake = FakeQA()
    monkeypatch.setattr(app, "qa", fake)
    return fake


def test_text_of_handles_strings_and_parts():
    assert app.text_of("hello") == "hello"
    assert (
        app.text_of([{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}])
        == "hi there"
    )


def test_to_langchain_skips_source_boxes():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "Can I eat in the car?"}]},
        {"role": "assistant", "content": "No."},
        {"role": "assistant", "content": "- Page 2", "metadata": {"title": app.SOURCES_TITLE}},
    ]

    assert app.to_langchain(history) == [HumanMessage("Can I eat in the car?"), AIMessage("No.")]


def test_load_pdf_file_reports_status(fake_qa):
    file_id, status, chat = app.load_pdf_file("/tmp/handbook.pdf")

    assert file_id == "abc123"
    assert "handbook.pdf" in status and "13 chunks" in status
    assert chat == []


def test_load_pdf_file_reports_errors(fake_qa):
    file_id, status, _ = app.load_pdf_file("/tmp/scan.pdf")

    assert file_id is None
    assert "no text" in status


def test_respond_streams_the_answer_then_adds_sources(fake_qa):
    updates = list(app.respond("Can I eat in the car?", [], "abc123"))

    chat, textbox = updates[-1]
    assert textbox == ""
    assert chat[0] == {"role": "user", "content": "Can I eat in the car?"}
    assert chat[1] == {"role": "assistant", "content": "No, only water."}
    assert chat[2].metadata["title"] == app.SOURCES_TITLE
    assert "Page 2" in chat[2].content
    assert len(updates) == 4  # three answer pieces, then the sources


def test_respond_asks_for_a_pdf_first(fake_qa):
    [(chat, _)] = list(app.respond("Hello?", [], None))

    assert chat[-1]["content"] == app.START_MESSAGE


def test_missing_model_says_what_to_pull(monkeypatch):
    class MissingModelQA:
        def index(self, path):
            raise ollama.ResponseError("model 'nomic-embed-text' not found", 404)

    monkeypatch.setattr(app, "qa", MissingModelQA())

    _, status, _ = app.load_pdf_file("handbook.pdf")

    assert "`ollama pull nomic-embed-text`" in status
