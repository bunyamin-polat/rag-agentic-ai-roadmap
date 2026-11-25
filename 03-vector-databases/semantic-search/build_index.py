"""Download SQuAD, embed every paragraph once, save the index, and measure the search.

Run it once from the repo root (it takes a few minutes):

    uv run python 03-vector-databases/semantic-search/build_index.py

The paragraphs are embedded in parts that are saved as they finish, so if the run stops,
starting it again continues where it stopped.
"""

from __future__ import annotations

import re
import shutil
import time

import numpy as np
from search import INDEX_DIR, SemanticSearch, embed_in_batches, load_paragraphs

from roadmap import get_embeddings

PART_SIZE = 1024  # paragraphs per saved part
TEST_QUESTIONS = 500


def main() -> None:
    paragraphs, dev_questions = load_paragraphs()
    print(f"{len(paragraphs):,} paragraphs, {len(dev_questions):,} test questions")

    embeddings = get_embeddings()
    # One parts folder per embedding model, so a run never mixes vectors from two models.
    parts_dir = INDEX_DIR / f"parts-{re.sub(r'[^A-Za-z0-9._-]', '_', embeddings.model)}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    parts = []
    for first in range(0, len(paragraphs), PART_SIZE):
        path = parts_dir / f"{first:06d}.npy"
        if not path.exists():
            texts = [paragraph.text for paragraph in paragraphs[first : first + PART_SIZE]]
            np.save(path, embed_in_batches(texts, embeddings.embed_documents))
        parts.append(np.load(path))
        done = min(first + PART_SIZE, len(paragraphs))
        elapsed = time.perf_counter() - start
        print(f"\r  embedded {done:,} of {len(paragraphs):,} paragraphs ({elapsed:.0f} s)", end="")
    print()

    start = time.perf_counter()
    search = SemanticSearch(paragraphs, np.vstack(parts), embeddings)
    print(f"HNSW index built in {time.perf_counter() - start:.1f} s")
    search.save(INDEX_DIR)
    shutil.rmtree(parts_dir)  # the full vectors are saved now
    print(f"Saved to {INDEX_DIR.name}/ next to this script")

    rng = np.random.default_rng(0)  # the same test questions on every run
    picked = sorted(rng.choice(len(dev_questions), TEST_QUESTIONS, replace=False))
    scores = search.evaluate([dev_questions[i] for i in picked])
    print(f"On {TEST_QUESTIONS} test questions:")
    for name, value in scores.items():
        print(f"  {name}: {value:.1%}")


if __name__ == "__main__":
    main()
