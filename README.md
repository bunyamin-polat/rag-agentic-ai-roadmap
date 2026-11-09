# RAG & Agentic AI Roadmap

A hands-on learning roadmap for building LLM, RAG, multimodal, and agentic AI applications,
with explanations, definitions, diagrams, notebooks, and small projects.

🚧 Work in progress.

## Setup

Everything runs on your own computer with [Ollama](https://ollama.com). No API keys, no costs.
You need [uv](https://docs.astral.sh/uv/) and Ollama installed.

```bash
git clone https://github.com/bunyamin-polat/rag-agentic-ai-roadmap.git
cd rag-agentic-ai-roadmap

ollama pull gpt-oss:20b  # default chat model (~13 GB, needs 16 GB RAM)
uv sync                  # creates .venv and installs everything
```

On a smaller machine, pull a smaller model such as `qwen3:4b-instruct` (~2.5 GB) and set
`CHAT_MODEL=qwen3:4b-instruct` in a `.env` file (see [`.env.example`](.env.example)).

To open a notebook, select the `.venv` Python interpreter as the kernel (VS Code), or run
`uv run --with jupyter jupyter lab`.

## Shared helpers

Every notebook and project uses the small `roadmap` package in [`src/roadmap`](src/roadmap):

```python
from roadmap import ask, get_chat_model

answer = ask("Explain RAG in one sentence.")  # plain Ollama client, returns a string
model = get_chat_model()  # LangChain chat model
```

## Development workflow

```text
main
 ├── feature/<topic>     one branch per topic, created from main
 │
 ▼
Pull request  →  CI checks  →  review  →  merge into main
```

Every pull request runs [CI](.github/workflows/ci.yml): ruff lint, ruff format check, and the
tests. Run the same checks locally before you push:

```bash
uv run ruff check .          # lint (notebooks included)
uv run ruff format --check . # formatting
uv run pytest                # tests for the shared helpers
uv run pytest 01-llm-foundations/message-analyzer   # tests for one project
```
