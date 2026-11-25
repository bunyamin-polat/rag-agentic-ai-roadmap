"""Tests for the Gradio handler in app.py, with a fake search: no Ollama server needed."""

import app
import ollama
import pytest
from search import Hit, Paragraph, SearchResult

STEAM = Paragraph("Steam engine", "A steam engine turns heat into motion.")


class FakeSearch:
    def __init__(self):
        self.seen = None

    def search(self, query, k, exact):
        self.seen = (query, k, exact)
        return SearchResult([Hit(STEAM, 0.81, 0)], 18.0, {"hnsw": 0.2, "flat": 2.5}, 4)


@pytest.fixture
def fake(monkeypatch):
    search = FakeSearch()
    monkeypatch.setattr(app, "get_search", lambda: search)
    return search


def test_results_show_timing_overlap_and_paragraphs(fake):
    markdown = app.run_search("How does a steam engine work?", 5, "Flat (exact)")

    assert "HNSW found **4 of the exact top 5**" in markdown
    assert "Flat search 2.50 ms" in markdown
    assert "#### 1. Steam engine · score 0.81" in markdown
    assert fake.seen == ("How does a steam engine work?", 5, True)


def test_missing_index_explains_how_to_build_it(monkeypatch):
    def no_index():
        raise FileNotFoundError("meta.json")

    monkeypatch.setattr(app, "get_search", no_index)

    assert app.run_search("anything", 5, "HNSW (fast, approximate)") == app.NO_INDEX


def test_empty_question_asks_for_one(fake):
    assert "Ask a question" in app.run_search("   ", 5, "HNSW (fast, approximate)")
    assert fake.seen is None


def test_missing_model_says_what_to_pull(monkeypatch):
    class MissingModel:
        def search(self, query, k, exact):
            raise ollama.ResponseError("model 'nomic-embed-text' not found", 404)

    monkeypatch.setattr(app, "get_search", MissingModel)

    assert "`ollama pull nomic-embed-text`" in app.run_search("anything", 5, "Flat (exact)")
