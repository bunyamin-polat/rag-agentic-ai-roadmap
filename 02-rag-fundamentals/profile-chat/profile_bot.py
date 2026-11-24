"""Profile chat with LlamaIndex: index one person's profile, suggest icebreakers, answer questions.

The same RAG idea as the LangChain notebooks, built with a second framework:

* each part of the profile becomes a LlamaIndex `Document`,
* a `VectorStoreIndex` embeds and stores them,
* a query engine writes icebreakers from the most relevant parts,
* a chat engine (`condense_plus_context`) rewrites follow-ups, retrieves, and answers.

This module has no web UI code, so it can be tested without Gradio or Ollama.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from llama_index.core import Document, PromptTemplate, VectorStoreIndex
from llama_index.core.llms import ChatMessage

from roadmap import get_settings
from roadmap.llm import supports_thinking

PROFILES_DIR = Path(__file__).parent / "profiles"

ICEBREAKER_PROMPT = PromptTemplate(
    "Here is information from {name}'s professional profile:\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Write three short, friendly conversation starters for meeting {name} for the first time. "
    "Base each one on a specific fact above and invent nothing. Use a numbered list, one "
    "question per line, without quotation marks. {query_str}"
)

CHAT_SYSTEM_PROMPT = (
    "Today's date is {today}. "
    "You answer questions about {name}'s professional profile, using only the profile "
    "information you are given. The answer may be a yes or a no, or may need a small step of "
    "reasoning from the profile. If the question is unclear, ask one short question to clarify "
    "instead of guessing. Only if the profile does not cover the question, say: The profile "
    "does not say. Keep answers to two or three sentences."
)


def list_profiles(folder: Path = PROFILES_DIR) -> dict[str, Path]:
    """Map each person's name to their profile file."""
    profiles = {}
    for path in sorted(folder.glob("*.json")):
        profiles[json.loads(path.read_text(encoding="utf-8"))["name"]] = path
    return profiles


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile_to_documents(profile: dict) -> list[Document]:
    """Turn a profile into one Document per fact group, so each can be found on its own.

    Every text repeats the person's name, so a chunk still makes sense on its own.
    """
    name = profile["name"]
    jobs = profile.get("experience", [])
    current = next((job for job in jobs if job["end"] == "present"), None)

    summary = f"{name}: {profile['headline']}. Lives in {profile['location']}."
    if current:
        summary += (
            f" Current job: {current['title']} at {current['company']}, since {current['start']}."
        )
    texts = [("summary", summary), ("about", f"About {name}: {profile['about']}")]

    # One timeline with all jobs in order, so "what did they do before that?" can be answered.
    if jobs:
        steps = "; then before that, ".join(
            f"{job['title']} at {job['company']} ({job['start']} to {job['end']})" for job in jobs
        )
        texts.append(("career", f"Career timeline of {name}, newest job first: {steps}."))

    for job in jobs:
        when = (
            f"since {job['start']} (current job)"
            if job is current
            else (f"from {job['start']} to {job['end']}")
        )
        texts.append(
            (
                "experience",
                f"{name} worked as {job['title']} at {job['company']} {when}. {job['description']}",
            )
        )
    for school in profile.get("education", []):
        details = f" {school['details']}" if school.get("details") else ""
        texts.append(
            (
                "education",
                f"{name} earned a {school['degree']} at {school['school']} in "
                f"{school['year']}.{details}",
            )
        )
    for project in profile.get("projects", []):
        texts.append(
            ("projects", f"Project by {name}: {project['name']}. {project['description']}")
        )

    lists = {
        "skills": "Skills",
        "languages": "Languages",
        "interests": "Interests and hobbies",
    }
    for key, label in lists.items():
        if profile.get(key):
            texts.append((key, f"{label} of {name}: {'; '.join(profile[key])}."))
    if profile.get("volunteering"):
        texts.append(("volunteering", f"Volunteering by {name}: {profile['volunteering']}"))

    return [
        Document(text=text, metadata={"person": name, "section": section})
        for section, text in texts
    ]


def default_llm():
    """A LlamaIndex chat model for the configured Ollama server."""
    from llama_index.llms.ollama import Ollama

    settings = get_settings()
    return Ollama(
        model=settings.chat_model,
        base_url=settings.ollama_host,
        temperature=0,
        # Only reasoning models such as gpt-oss accept a thinking level; others return an error.
        thinking="low" if supports_thinking(settings.chat_model) else None,
        context_window=8192,
        request_timeout=180,
    )


def default_embed_model():
    """A LlamaIndex embedding model for the configured Ollama server."""
    from llama_index.embeddings.ollama import OllamaEmbedding

    settings = get_settings()
    return OllamaEmbedding(model_name=settings.embed_model, base_url=settings.ollama_host)


@dataclass(frozen=True)
class Reply:
    text: str
    sections: list[str]


class ProfileBot:
    """Answers questions about one person's profile."""

    def __init__(
        self, profile: dict, llm=None, embed_model=None, top_k: int = 3, today: date | None = None
    ):
        self.profile = profile
        # Models do not know today's date, so we tell them (needed for "how long" questions).
        self.today = today or date.today()
        self.name = profile["name"]
        self.llm = llm or default_llm()
        self.top_k = top_k
        self.documents = profile_to_documents(profile)
        self.index = VectorStoreIndex.from_documents(
            self.documents, embed_model=embed_model or default_embed_model()
        )

    def icebreakers(self) -> str:
        """Three conversation starters, each based on a fact from the profile."""
        engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=6,
            text_qa_template=ICEBREAKER_PROMPT.partial_format(name=self.name),
        )
        query = "Pick facts about projects, hobbies, volunteering, or unusual career steps."
        return str(engine.query(query)).strip()

    def _chat_engine(self):
        return self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=self.llm,
            similarity_top_k=self.top_k,
            system_prompt=CHAT_SYSTEM_PROMPT.format(name=self.name, today=f"{self.today:%d %B %Y}"),
        )

    def chat(self, question: str, history: Sequence[ChatMessage] = ()) -> Reply:
        """Answer a question. `history` holds the earlier turns of this conversation."""
        response = self._chat_engine().chat(question, chat_history=list(history))
        sections = [node.node.metadata["section"] for node in response.source_nodes]
        return Reply(response.response.strip(), sections)

    def stream_chat(
        self, question: str, history: Sequence[ChatMessage] = ()
    ) -> tuple[list[str], Iterator[str]]:
        """Like `chat`, but returns the source sections and an iterator of answer pieces."""
        response = self._chat_engine().stream_chat(question, chat_history=list(history))
        sections = [node.node.metadata["section"] for node in response.source_nodes]
        return sections, response.response_gen
