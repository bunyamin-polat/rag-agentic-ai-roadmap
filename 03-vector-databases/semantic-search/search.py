"""Semantic search over about 21,000 Wikipedia paragraphs with FAISS.

* The paragraphs come from SQuAD v1.1 (open data, CC BY-SA 4.0), the dataset of Notebook 3.
* `build_index.py` embeds them once with Ollama and saves a FAISS HNSW index.
* A search embeds the question and asks both the HNSW index and an exact Flat index, so you
  can see the speed and the accuracy of the approximate search side by side.

This module has no web UI code, so it can be tested without Gradio or Ollama (see tests/).
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from roadmap import get_embeddings

HERE = Path(__file__).parent
DATA_DIR = HERE.parent / ".cache"  # the SQuAD download, shared with Notebook 3
INDEX_DIR = HERE / ".cache"  # the saved index; listed in .gitignore
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
SQUAD_FILES = ("train-v1.1.json", "dev-v1.1.json")


@dataclass(frozen=True)
class Paragraph:
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    paragraph: Paragraph
    score: float  # cosine similarity, because the vectors have length 1
    position: int


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]  # from the index you asked for
    embed_ms: float  # time to embed the question with Ollama
    search_ms: dict[str, float]  # time per index: "hnsw" and "flat"
    overlap: int  # how many of the exact top k the HNSW index also found


def squad_file(name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download a SQuAD file once and return its path."""
    path = data_dir / name
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL + name, path)
    return path


def load_paragraphs(data_dir: Path = DATA_DIR) -> tuple[list[Paragraph], list[tuple[str, int]]]:
    """All SQuAD paragraphs, and every development question with the position of its paragraph.

    The development questions are the test set: SQuAD says which paragraph answers each one.
    """
    paragraphs: list[Paragraph] = []
    dev_questions: list[tuple[str, int]] = []
    for name in SQUAD_FILES:
        data = json.loads(squad_file(name, data_dir).read_text(encoding="utf-8"))["data"]
        for article in data:
            title = article["title"].replace("_", " ")
            for paragraph in article["paragraphs"]:
                paragraphs.append(Paragraph(title, paragraph["context"]))
                if name.startswith("dev"):
                    position = len(paragraphs) - 1
                    dev_questions += [(qa["question"], position) for qa in paragraph["qas"]]
    return paragraphs, dev_questions


def embed_in_batches(
    texts: Sequence[str],
    embed_documents: Callable[[list[str]], list[list[float]]],
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts a batch at a time; much bigger requests can make Ollama fail."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(embed_documents(list(texts[start : start + batch_size])))
    return np.array(vectors, dtype=np.float32)  # FAISS works with 32-bit floats


def build_hnsw(vectors: np.ndarray, links: int = 32, ef_search: int = 64) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vectors.shape[1], links, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(vectors)
    return index


def build_flat(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


class IndexOutOfDate(RuntimeError):
    """The saved index was built with another embedding model."""


class SemanticSearch:
    """Searches paragraphs with an HNSW index and, for comparison, an exact Flat index."""

    def __init__(self, paragraphs: list[Paragraph], vectors: np.ndarray, embeddings, hnsw=None):
        self.paragraphs = paragraphs
        self.vectors = vectors
        self.embeddings = embeddings
        self.indexes = {"hnsw": hnsw or build_hnsw(vectors), "flat": build_flat(vectors)}

    @property
    def model(self) -> str:
        return getattr(self.embeddings, "model", type(self.embeddings).__name__)

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        faiss.write_index(self.indexes["hnsw"], str(index_dir / "hnsw.index"))
        paragraphs = [asdict(paragraph) for paragraph in self.paragraphs]
        (index_dir / "paragraphs.json").write_text(json.dumps(paragraphs), encoding="utf-8")
        meta = {"model": self.model, "paragraphs": len(self.paragraphs)}
        (index_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR, embeddings=None) -> SemanticSearch:
        """Load a saved index. Raises FileNotFoundError if `build_index.py` has not run yet."""
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        embeddings = embeddings or get_embeddings()
        model = getattr(embeddings, "model", type(embeddings).__name__)
        if meta["model"] != model:
            raise IndexOutOfDate(
                f"The index was built with {meta['model']}, but the embedding model is {model}. "
                "Run build_index.py again."
            )
        paragraphs = json.loads((index_dir / "paragraphs.json").read_text(encoding="utf-8"))
        return cls(
            [Paragraph(**paragraph) for paragraph in paragraphs],
            np.load(index_dir / "vectors.npy"),
            embeddings,
            faiss.read_index(str(index_dir / "hnsw.index")),
        )

    def search(self, query: str, k: int = 5, exact: bool = False) -> SearchResult:
        """Embed the question, search both indexes, return the hits of the one you asked for."""
        start = time.perf_counter()
        vector = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        embed_ms = (time.perf_counter() - start) * 1000

        found, search_ms = {}, {}
        for name, index in self.indexes.items():
            start = time.perf_counter()
            scores, positions = index.search(vector, k)
            search_ms[name] = (time.perf_counter() - start) * 1000
            found[name] = [
                Hit(self.paragraphs[position], float(score), int(position))
                for score, position in zip(scores[0], positions[0], strict=True)
                if position >= 0  # FAISS pads with -1 when it finds fewer than k
            ]
        overlap = len({h.position for h in found["hnsw"]} & {h.position for h in found["flat"]})
        return SearchResult(found["flat" if exact else "hnsw"], embed_ms, search_ms, overlap)

    def evaluate(self, questions: Sequence[tuple[str, int]], k: int = 5) -> dict[str, float]:
        """Measure the search on questions whose right paragraph is known.

        hit@1 / hit@k: how often the right paragraph is first / in the top k (exact search).
        recall@k: how much of the exact top k the HNSW index finds.
        """
        vectors = embed_in_batches([q for q, _ in questions], self.embeddings.embed_documents, 128)
        answers = np.array([position for _, position in questions])
        positions = {name: index.search(vectors, k)[1] for name, index in self.indexes.items()}
        flat, hnsw = positions["flat"], positions["hnsw"]
        overlap = [len(set(f) & set(h)) / k for f, h in zip(flat, hnsw, strict=True)]
        return {
            "hit@1": float(np.mean(flat[:, 0] == answers)),
            f"hit@{k}": float(np.mean([a in row for a, row in zip(answers, flat, strict=True)])),
            f"recall@{k}": float(np.mean(overlap)),
        }
