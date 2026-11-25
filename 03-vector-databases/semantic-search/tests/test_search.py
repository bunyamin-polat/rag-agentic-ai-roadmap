"""Tests for search.py with fake embeddings and a tiny SQuAD file: no Ollama, no download."""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from search import (
    IndexOutOfDate,
    Paragraph,
    SemanticSearch,
    embed_in_batches,
    load_paragraphs,
)


class FakeEmbeddings(DeterministicFakeEmbedding):
    """Fake embeddings with a model name, like OllamaEmbeddings."""

    model: str = "fake-model"


def squad(articles: dict[str, list[tuple[str, list[str]]]]) -> dict:
    """A tiny SQuAD-shaped file: {title: [(paragraph, [questions])]}."""
    return {
        "version": "1.1",
        "data": [
            {
                "title": title,
                "paragraphs": [
                    {"context": text, "qas": [{"question": q} for q in questions]}
                    for text, questions in paragraphs
                ],
            }
            for title, paragraphs in articles.items()
        ],
    }


@pytest.fixture
def data_dir(tmp_path):
    train = squad(
        {
            "Steam_engine": [("A steam engine turns heat into motion.", ["What is it?"])],
            "Roman_Empire": [
                ("Augustus was the first Roman emperor.", ["Who was first?"]),
                ("Rome had many roads.", []),
            ],
        }
    )
    dev = squad({"Autumn": [("Leaves change color in autumn.", ["Why?", "When?"])]})
    (tmp_path / "train-v1.1.json").write_text(json.dumps(train))
    (tmp_path / "dev-v1.1.json").write_text(json.dumps(dev))
    return tmp_path


@pytest.fixture
def paragraphs() -> list[Paragraph]:
    return [
        Paragraph(f"Title {i}", f"Paragraph number {i} about topic {i % 7}.") for i in range(40)
    ]


@pytest.fixture
def engine(paragraphs) -> SemanticSearch:
    embeddings = FakeEmbeddings(size=16)
    vectors = embed_in_batches([p.text for p in paragraphs], embeddings.embed_documents)
    return SemanticSearch(paragraphs, vectors, embeddings)


def test_load_paragraphs_reads_titles_and_test_questions(data_dir):
    paragraphs, dev_questions = load_paragraphs(data_dir)

    assert [p.title for p in paragraphs] == [
        "Steam engine",
        "Roman Empire",
        "Roman Empire",
        "Autumn",
    ]
    # Only the development questions are test questions, each with its paragraph's position.
    assert dev_questions == [("Why?", 3), ("When?", 3)]


def test_embed_in_batches_sends_small_batches():
    sizes = []

    def embed(texts):
        sizes.append(len(texts))
        return [[1.0, 0.0]] * len(texts)

    vectors = embed_in_batches([f"text {i}" for i in range(70)], embed, batch_size=32)

    assert sizes == [32, 32, 6]
    assert vectors.shape == (70, 2) and str(vectors.dtype) == "float32"


def test_search_finds_the_paragraph_itself_first(engine, paragraphs):
    result = engine.search(paragraphs[12].text, k=3)

    assert result.hits[0].position == 12
    assert set(result.search_ms) == {"hnsw", "flat"}
    assert 0 <= result.overlap <= 3


def test_exact_search_uses_the_flat_index(engine, paragraphs):
    exact = engine.search(paragraphs[5].text, k=5, exact=True)
    flat_scores = [hit.score for hit in exact.hits]

    assert flat_scores == sorted(flat_scores, reverse=True)
    assert exact.hits[0].position == 5


def test_evaluate_counts_hits(engine, paragraphs):
    questions = [(paragraphs[i].text, i) for i in range(10)]  # each "question" is its paragraph

    scores = engine.evaluate(questions, k=5)

    assert scores["hit@1"] == 1.0
    assert scores["hit@5"] == 1.0
    assert 0.0 <= scores["recall@5"] <= 1.0


def test_save_and_load_give_the_same_results(engine, paragraphs, tmp_path):
    engine.save(tmp_path)

    loaded = SemanticSearch.load(tmp_path, embeddings=FakeEmbeddings(size=16))

    assert loaded.paragraphs == paragraphs
    query = paragraphs[20].text
    assert loaded.search(query).hits == engine.search(query).hits


def test_load_refuses_an_index_from_another_model(engine, tmp_path):
    engine.save(tmp_path)

    with pytest.raises(IndexOutOfDate, match=r"build_index\.py"):
        SemanticSearch.load(tmp_path, embeddings=FakeEmbeddings(size=16, model="other-model"))


def test_load_without_an_index_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        SemanticSearch.load(tmp_path / "missing", embeddings=FakeEmbeddings(size=16))
