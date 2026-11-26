# Stage 4 — Advanced Retrieval

Stages 2 and 3 searched the same way every time: embed the question, keep the k closest
chunks. That works often, but not always. Copies fill the top k, unrelated chunks come back
when nothing fits, questions use other words than the documents, filters hide inside the
question, and exact terms get blurred by embeddings. This stage makes the search smarter, and
measures whether each change really helps.

![Stage 4 path](assets/stage_map.svg)

## What you will learn

- Five smarter retrievers, each built in a few lines with LCEL: **MMR**, a **score
  threshold**, **multi-query**, **self-query**, and **parent document**.
- Keyword search with **BM25**, **hybrid search**, and **reranking**.
- How to measure retrieval with **hit@k** and **MRR**, and weigh quality against time.

## Contents

| # | Type | Topic | Status |
| --- | --- | --- | --- |
| 1 | Notebook | [Better retrievers](01_better_retrievers.ipynb): MMR, score threshold, multi-query, self-query, parent document | ✅ Ready |
| 2 | Notebook | Keyword search, hybrid search, and reranking (LlamaIndex) | 🔜 Coming soon |
| 3 | Notebook | Measuring retrieval: hit@k, MRR, and time | 🔜 Coming soon |
| – | Project | YouTube Q&A: summarize a video and chat about it (Gradio) | 🔜 Coming soon |

Start with [Stage 2](../02-rag-fundamentals/) and [Stage 3](../03-vector-databases/) if
retrievers and Chroma are new to you.

## Before you start

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull gpt-oss:20b        # chat model
uv sync                        # run in the repo root
```

Open a notebook and select the `.venv` Python kernel.

## Sample data

The notebooks reuse Brightleaf's made-up knowledge base: the employee handbook and IT FAQ
from [Stage 2](../02-rag-fundamentals/data/), and the company announcements from
[Stage 3](../03-vector-databases/data/).

## Key terms

The notebook in brackets is where each term is explained in detail.

### Smarter retrievers

| Term | Meaning |
| --- | --- |
| **MMR** | Maximal marginal relevance: picks results one by one, each close to the question **and** different from the ones already picked. `lambda_mult` sets the balance (1 = only relevance), `fetch_k` the number of candidates. (NB 1) |
| **Near-duplicates** | Chunks with (almost) the same text, for example when the same file was indexed twice. They waste places in the top k. (NB 1) |
| **Relevance score** | A score where higher means closer. For a cosine store it is the cosine similarity. (NB 1) |
| **Score threshold** | Drops every result whose relevance score is below a limit, so the result can be empty. No limit separates good and bad matches perfectly; keep it low and measure. (NB 1) |
| **Multi-query** | The model writes the question in other words; every version is searched, and each chunk is kept once. Helps when questions and documents use different words. (NB 1) |
| **Self-query** | The model splits the question into a topic and metadata filters ("IT", "2026"), and the search uses both. (NB 1) |
| **Parent document** | Search small pieces for a precise match, then give the model the whole section (the parent) each piece came from. (NB 1) |

## Next

➡️ Stage 5 — Multimodal (coming soon): speech to text, text to speech, image generation, and
vision models.
