"""Search about 21,000 Wikipedia paragraphs by meaning, with FAISS.

Build the index once (a few minutes), then start the app, from the repo root:

    uv run python 03-vector-databases/semantic-search/build_index.py
    uv run python 03-vector-databases/semantic-search/app.py
"""

from __future__ import annotations

import os

import gradio as gr
from search import IndexOutOfDate, SearchResult, SemanticSearch

from roadmap import OLLAMA_ERRORS, explain_ollama_error

MODES = {"HNSW (fast, approximate)": False, "Flat (exact)": True}
EXAMPLES = [
    "How does a steam engine work?",
    "When did the Black Death reach Europe?",
    "What do chloroplasts do?",
    "How many people live in New York City?",
]
NO_INDEX = (
    "⚠️ There is no index yet. Build it once with:\n\n"
    "`uv run python 03-vector-databases/semantic-search/build_index.py`"
)

_search: SemanticSearch | None = None


def get_search() -> SemanticSearch:
    """Load the saved index on first use."""
    global _search
    if _search is None:
        _search = SemanticSearch.load()
    return _search


def results_markdown(result: SearchResult, k: int) -> str:
    timing = (
        f"Question embedded in {result.embed_ms:.0f} ms · "
        f"HNSW search {result.search_ms['hnsw']:.2f} ms · "
        f"Flat search {result.search_ms['flat']:.2f} ms · "
        f"HNSW found **{result.overlap} of the exact top {k}**"
    )
    blocks = [timing]
    for number, hit in enumerate(result.hits, 1):
        blocks.append(
            f"#### {number}. {hit.paragraph.title} · score {hit.score:.2f}\n\n{hit.paragraph.text}"
        )
    return "\n\n".join(blocks)


def run_search(query: str, k: float, mode: str) -> str:
    query = (query or "").strip()
    if not query:
        return "*Ask a question about history, science, sports, music, places, ...*"
    try:
        result = get_search().search(query, int(k), exact=MODES[mode])
    except FileNotFoundError:
        return NO_INDEX
    except IndexOutOfDate as error:
        return f"⚠️ {error}"
    except OLLAMA_ERRORS as error:
        return explain_ollama_error(error)
    return results_markdown(result, int(k))


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Semantic Search") as app:
        gr.Markdown(
            "# 🔎 Semantic Search\n"
            "Ask a question and find the Wikipedia paragraphs that answer it, by meaning. "
            "About 21,000 paragraphs from SQuAD v1.1 (CC BY-SA 4.0)."
        )
        query = gr.Textbox(placeholder="Ask a question ...", show_label=False, submit_btn=True)
        with gr.Row():
            k = gr.Slider(1, 10, value=5, step=1, label="Results (k)")
            mode = gr.Radio(list(MODES), value=next(iter(MODES)), label="Index")
        gr.Examples(EXAMPLES, inputs=query)
        results = gr.Markdown()

        for event in (query.submit, k.change, mode.change):
            event(run_search, [query, k, mode], results)
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7863")))
