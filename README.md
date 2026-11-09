# RAG & Agentic AI Roadmap

A hands-on learning roadmap for building LLM, RAG, multimodal, and agentic AI applications,
with explanations, definitions, diagrams, notebooks, and small projects.

🚧 Work in progress.

## Setup

You need [uv](https://docs.astral.sh/uv/) and a free [Hugging Face](https://huggingface.co) account.
All models run through [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers).

```bash
git clone https://github.com/bunyamin-polat/rag-agentic-ai-roadmap.git
cd rag-agentic-ai-roadmap

uv sync                  # creates .venv and installs everything
cp .env.example .env     # then paste your token into .env
```

Create a token at <https://huggingface.co/settings/tokens>. A fine-grained token with the
**"Make calls to Inference Providers"** permission is enough.

To open a notebook, select the `.venv` Python interpreter as the kernel (VS Code), or run
`uv run --with jupyter jupyter lab`.

## Shared helpers

Every notebook and project uses the small `roadmap` package in [`src/roadmap`](src/roadmap):

```python
from roadmap import ask, get_chat_model

ask("Explain RAG in one sentence.")  # plain Hugging Face client, returns a string
model = get_chat_model()              # LangChain chat model
```

The default model is `meta-llama/Llama-3.3-70B-Instruct`. Change it with `HF_CHAT_MODEL` in `.env`.

## Code quality

```bash
uv run ruff check .      # lint (notebooks included)
uv run ruff format .     # format
```
