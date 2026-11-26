# RAG & Agentic AI Roadmap

A hands-on learning roadmap for building LLM, RAG, multimodal, and agentic AI applications,
with explanations, definitions, diagrams, notebooks, and small projects.

🚧 Work in progress. New stages are added step by step.

## Roadmap

Each stage has its own folder with a README, notebooks that explain the ideas, and at least one
small project.

| Stage | What you learn | Projects | Status |
| --- | --- | --- | --- |
| [1. LLM Foundations](01-llm-foundations/) | tokens, prompting, LangChain chains, structured output, memory, tools, agents | Message Analyzer (Flask) | ✅ Ready |
| [2. RAG Fundamentals](02-rag-fundamentals/) | loading and chunking documents, embeddings, retrieval, chat with your documents | PDF Q&A bot, profile chat | ✅ Ready |
| [3. Vector Databases](03-vector-databases/) | similarity measures, ChromaDB, FAISS, fast search indexes | food recommender, semantic search | ✅ Ready |
| [4. Advanced Retrieval](04-advanced-retrieval/) | better retrievers, hybrid search, reranking, measuring retrieval quality | YouTube Q&A | 🚧 In progress |
| 5. Multimodal | speech to text, text to speech, image generation, vision models | meeting assistant, image Q&A, style finder | 🔜 Planned |
| 6. Agentic AI | added as the roadmap grows | – | 🔜 Planned |

## Setup

The models run on your own computer with [Ollama](https://ollama.com). No API keys, no costs.
You only need the internet for one-time downloads: the models (`ollama pull`), the Python
packages (`uv sync`), and a few public files such as a tokenizer in Notebook 1.
You need [uv](https://docs.astral.sh/uv/) and Ollama installed.

```bash
git clone https://github.com/bunyamin-polat/rag-agentic-ai-roadmap.git
cd rag-agentic-ai-roadmap

ollama pull gpt-oss:20b  # default chat model (~13 GB, needs 16 GB RAM)
uv sync                  # creates .venv and installs everything
```

`uv sync` installs the packages of every stage. To install less, name a stage together with the
stages before it, because each stage builds on the earlier ones, for example
`uv sync --no-default-groups --group dev --group stage-01 --group stage-02`.

Each stage README lists any extra models it needs. On a smaller machine, pull a smaller model
such as `qwen3:4b-instruct` (~2.5 GB) and set `CHAT_MODEL=qwen3:4b-instruct` in a `.env` file
(see [`.env.example`](.env.example)).

To open a notebook, select the `.venv` Python interpreter as the kernel (VS Code), or run
`uv run --with jupyter jupyter lab`.

## Shared helpers

Every notebook and project uses the small `roadmap` package in [`src/roadmap`](src/roadmap):

```python
from roadmap import ask, get_chat_model, get_embeddings
from roadmap.loaders import load_directory

answer = ask("Explain RAG in one sentence.")  # plain Ollama client, returns a string
model = get_chat_model()  # LangChain chat model
embeddings = get_embeddings()  # LangChain embedding model (from Stage 2)
documents = load_directory("02-rag-fundamentals/data")  # .txt, .md, .pdf, .html files as Documents
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

Run each project's tests with its own command, like the last line: the projects share file
names such as `app.py`, so one pytest run cannot load two of them at once. CI does the same
for every folder that has a `conftest.py`.
