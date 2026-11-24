"""Question answering over one PDF: index it once, then answer questions with page sources.

This module has no web UI code, so it can be tested without Gradio or Ollama (see tests/).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from roadmap import get_chat_model, get_embeddings
from roadmap.loaders import load_pdf

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about a PDF document, using only the context below. The answer "
            "may be a yes or a no, or may need a small step of reasoning from the context. If the "
            "question is unclear, ask one short question to clarify instead of guessing. Only if "
            "the context does not cover the question, say: I don't know, the document does not "
            "say. Keep answers short.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("user", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question so that it can be understood without the chat "
            "history: wherever it says 'it', 'one', or 'them', name the thing itself. Keep it "
            "short. Do not answer the question.",
        ),
        MessagesPlaceholder("history"),
        ("user", "{question}"),
    ]
)


class StandaloneQuestion(BaseModel):
    question: str = Field(description="the rewritten question that can be understood on its own")


class PdfError(ValueError):
    """The PDF cannot be used, for example because it was never indexed or has no text."""


@dataclass(frozen=True)
class IndexedPdf:
    """What we know about an indexed PDF."""

    file_id: str
    name: str
    pages: int
    chunks: int


def format_docs(docs: Sequence[Document]) -> str:
    """Join chunks for the prompt, each with its page number."""
    return "\n\n---\n\n".join(f"[page {doc.metadata['page']}]\n{doc.page_content}" for doc in docs)


def sources_markdown(docs: Sequence[Document], preview: int = 90) -> str:
    """One line per retrieved chunk: its page number and the start of its text."""
    lines = []
    for doc in docs:
        text = " ".join(doc.page_content.split())
        if len(text) > preview:
            text = text[:preview].rsplit(" ", 1)[0] + " ..."
        lines.append(f"- **Page {doc.metadata['page']}:** {text}")
    return "\n".join(lines)


class PdfQA:
    """Indexes PDFs and answers questions about them.

    Every PDF is indexed once and kept in memory, keyed by a hash of its bytes, so asking many
    questions (or uploading the same file again) does not repeat the slow embedding step.
    The chat model and embedding model are created on first use; tests can pass fakes instead.
    """

    def __init__(
        self,
        llm=None,
        embeddings=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int = 3,
        max_history: int = 6,
    ):
        self._llm = llm
        self._embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.k = k
        self.max_history = max_history
        self._stores: dict[str, InMemoryVectorStore] = {}
        self._indexed: dict[str, IndexedPdf] = {}

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def index(self, path: str | Path) -> IndexedPdf:
        """Load, split, and embed a PDF. A file that was indexed before is not indexed again."""
        path = Path(path)
        file_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if file_id in self._indexed:
            return self._indexed[file_id]

        pages = load_pdf(path)
        if not pages:
            raise PdfError("This PDF has no text. Is it a scanned document?")
        chunks = self.splitter.split_documents(pages)
        self._stores[file_id] = InMemoryVectorStore.from_documents(chunks, self.embeddings)
        self._indexed[file_id] = IndexedPdf(file_id, path.name, len(pages), len(chunks))
        return self._indexed[file_id]

    def rewrite(self, question: str, history: Sequence[BaseMessage]) -> str:
        """Turn a follow-up question into a standalone question."""
        chain = REWRITE_PROMPT | self.llm.with_structured_output(StandaloneQuestion)
        result = chain.invoke({"history": list(history), "question": question})
        return StandaloneQuestion.model_validate(result).question

    def search_query(self, question: str, history: Sequence[BaseMessage]) -> str:
        """The text to search with: the question itself, or its rewrite if there is a history."""
        return self.rewrite(question, history) if history else question

    def retrieve(self, file_id: str, query: str) -> list[Document]:
        store = self._stores.get(file_id)
        if store is None:
            raise PdfError("Upload a PDF first.")
        return store.similarity_search(query, k=self.k)

    def ask(
        self, file_id: str, question: str, history: Sequence[BaseMessage] = ()
    ) -> tuple[list[Document], Iterator[str]]:
        """Find the chunks for a question and start streaming the answer.

        Returns the retrieved chunks (the sources) and an iterator of answer pieces.
        """
        history = list(history)[-self.max_history :]
        docs = self.retrieve(file_id, self.search_query(question, history))
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        stream = chain.stream(
            {"context": format_docs(docs), "history": history, "question": question}
        )
        return docs, stream
