"""Tests for rag.py with fake models: no Ollama server needed."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pypdf import PdfWriter
from rag import PdfError, PdfQA, format_docs, sources_markdown

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "data" / "handbook.pdf"


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Fake embeddings that count how many texts were embedded."""

    calls: int = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


def fake_llm(answer: str = "Up to 40 EUR per month.") -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=answer)]))


@pytest.fixture
def qa():
    return PdfQA(llm=fake_llm(), embeddings=CountingEmbeddings(size=32))


def test_index_reads_pages_and_chunks(qa):
    info = qa.index(SAMPLE_PDF)

    assert info.name == "handbook.pdf"
    assert info.pages == 3
    assert info.chunks > info.pages


def test_index_embeds_each_pdf_only_once(qa):
    first = qa.index(SAMPLE_PDF)
    embedded = qa.embeddings.calls

    second = qa.index(SAMPLE_PDF)

    assert second == first
    assert qa.embeddings.calls == embedded  # nothing was embedded again


def test_index_rejects_pdf_without_text(qa, tmp_path):
    blank = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.write(blank)

    with pytest.raises(PdfError, match="no text"):
        qa.index(blank)


def test_retrieve_needs_an_indexed_pdf(qa):
    with pytest.raises(PdfError, match="Upload a PDF first"):
        qa.retrieve("unknown-id", "anything")


def test_ask_returns_sources_and_streams_the_answer(qa):
    info = qa.index(SAMPLE_PDF)

    docs, stream = qa.ask(info.file_id, "How much of my phone costs are covered?")

    assert len(docs) == qa.k
    assert all("page" in doc.metadata for doc in docs)
    assert "".join(stream) == "Up to 40 EUR per month."


def test_first_question_is_searched_as_it_is(qa, monkeypatch):
    monkeypatch.setattr(qa, "rewrite", lambda question, history: pytest.fail("no rewrite"))

    assert qa.search_query("Can I eat in the car?", []) == "Can I eat in the car?"


def test_follow_up_question_is_rewritten(qa, monkeypatch):
    monkeypatch.setattr(qa, "rewrite", lambda question, history: f"rewritten: {question}")
    history = [HumanMessage("Tell me about the cars."), AIMessage("There are four.")]

    assert qa.search_query("Can I use one?", history) == "rewritten: Can I use one?"


def test_format_docs_and_sources_show_page_numbers():
    docs = [
        Document(page_content="Eating is not allowed in company vehicles.", metadata={"page": 2}),
        Document(page_content="word " * 40, metadata={"page": 3}),
    ]

    assert format_docs(docs).startswith("[page 2]\nEating")
    lines = sources_markdown(docs, preview=20).splitlines()
    assert lines[0] == "- **Page 2:** Eating is not ..."
    assert lines[1].startswith("- **Page 3:** word word") and lines[1].endswith(" ...")
