"""Brightleaf Kitchen: find dishes by meaning, with filters, similar dishes, and a RAG chat.

It puts the Stage 3 notebooks together:

* the dishes live in a Chroma collection that is saved to disk (Notebook 2),
* every search ranks by cosine similarity (Notebook 1), inside Chroma's HNSW index (Notebook 3),
* metadata filters narrow every search: cuisine, course, calories, diet, spice, allergens.

This module has no web UI code, so it can be tested without Gradio or Ollama (see tests/).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from roadmap import get_chat_model, get_embeddings

HERE = Path(__file__).parent
MENU_FILE = HERE / "data" / "dishes.json"
DB_PATH = HERE / "chroma_db"  # listed in .gitignore
COLLECTION = "brightleaf-kitchen"
SPICE_WORDS = ("not spicy", "mildly spicy", "spicy", "very spicy")

# Returned without asking the model when the filters leave no dish at all.
NO_MATCH = "No dish on the menu matches these filters. Try changing them."

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the assistant of Brightleaf Kitchen, an office canteen. Recommend up to "
            "three dishes, only from the menu below, and say in a few words why each one fits. "
            "Keep the user's wishes from earlier messages in mind. The answer may "
            "need a small step of reasoning from the menu. If the question is unclear, ask one "
            "short question to clarify instead of guessing. Take every number, diet label, and "
            "allergen from the menu below, never from memory. Never say that a dish is safe for "
            "an allergy; "
            "name its listed allergens instead. If no dish below fits the request, reply with "
            "exactly this sentence and nothing else: None of the matching dishes fits that "
            "request. Keep answers short.\n\nMenu:\n{context}",
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
            "history: wherever it says 'it', 'one', or 'them', name the thing itself, and keep "
            "the wishes from earlier messages that still apply, such as a diet or a course. For "
            "example, after 'I want a vegan curry', the question 'Is there a spicy one?' becomes "
            "'Is there a spicy vegan curry?'. Keep it short. Do not answer the question.",
        ),
        MessagesPlaceholder("history"),
        ("user", "{question}"),
    ]
)


class StandaloneQuestion(BaseModel):
    question: str = Field(description="the rewritten question that can be understood on its own")


def load_menu(path: str | Path = MENU_FILE) -> dict:
    """The menu file: a list of dishes and the list of allergens they can contain."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diet_labels(dish: dict) -> list[str]:
    labels = ["vegan"] if dish["vegan"] else ["vegetarian"] if dish["vegetarian"] else []
    if dish["gluten_free"]:
        labels.append("gluten-free")
    return labels


def dish_text(dish: dict) -> str:
    """The text that gets embedded: meaning only. Numbers are left to the filters."""
    labels = ", ".join(diet_labels(dish)).capitalize()
    return (
        f"{dish['name']}. {dish['course'].capitalize()} from {dish['cuisine']} cuisine. "
        f"{dish['description']} Ingredients: {', '.join(dish['ingredients'])}. "
        f"{SPICE_WORDS[dish['spice']].capitalize()}." + (f" {labels}." if labels else "")
    )


def dish_metadata(dish: dict, allergens: Sequence[str]) -> dict:
    """Everything a filter may need. Chroma metadata only takes simple values, so every
    allergen gets its own true/false field."""
    metadata = {key: dish[key] for key in ("name", "cuisine", "course", "calories", "spice")}
    metadata |= {key: dish[key] for key in ("vegetarian", "vegan", "gluten_free")}
    metadata |= {f"has_{allergen}": allergen in dish["allergens"] for allergen in allergens}
    return metadata


def dish_card(dish: dict) -> str:
    """One line per dish for the model's prompt, with the facts it may repeat."""
    labels = ", ".join(diet_labels(dish)) or "no diet label"
    allergens = ", ".join(dish["allergens"]) or "none listed"
    return (
        f"- {dish['name']} ({dish['cuisine']} {dish['course']}): {dish['description']} "
        f"{dish['calories']} kcal, {dish['protein_g']} g protein. {labels.capitalize()}. "
        f"{SPICE_WORDS[dish['spice']].capitalize()}. Allergens: {allergens}."
    )


@dataclass(frozen=True)
class Filters:
    """What the user picked in the filter panel. Empty values mean "no limit"."""

    cuisines: tuple[str, ...] = ()
    course: str | None = None
    max_calories: int | None = None
    vegetarian: bool = False
    vegan: bool = False
    gluten_free: bool = False
    max_spice: int | None = None
    avoid: tuple[str, ...] = ()  # allergens to leave out

    def where(self) -> dict | None:
        """The Chroma `where` filter, or None when nothing is selected."""
        conditions: list[dict] = []
        if self.cuisines:
            conditions.append({"cuisine": {"$in": list(self.cuisines)}})
        if self.course:
            conditions.append({"course": self.course})
        if self.max_calories is not None:  # 0 is a limit too
            conditions.append({"calories": {"$lte": self.max_calories}})
        for flag in ("vegetarian", "vegan", "gluten_free"):
            if getattr(self, flag):
                conditions.append({flag: True})
        if self.max_spice is not None:
            conditions.append({"spice": {"$lte": self.max_spice}})
        conditions += [{f"has_{allergen}": False} for allergen in self.avoid]
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass(frozen=True)
class Match:
    dish: dict
    similarity: float  # cosine similarity, 1 = same meaning


class FoodRecommender:
    """Searches the menu by meaning. The menu is embedded once and saved to `path`; it is only
    embedded again when the menu or the embedding model changes. Tests can pass fake models."""

    def __init__(
        self,
        menu: dict | None = None,
        embeddings=None,
        llm=None,
        path: str | Path = DB_PATH,
        k: int = 5,
        max_history: int = 6,
    ):
        menu = menu or load_menu()
        self.dishes: list[dict] = menu["dishes"]
        self.allergens: list[str] = menu["allergens"]
        self.by_id = {dish["id"]: dish for dish in self.dishes}
        self.embeddings = embeddings or get_embeddings()
        self._llm = llm
        self.k = k
        self.max_history = max_history
        self.collection = self._open(chromadb.PersistentClient(path=str(path)))

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

    def fingerprint(self) -> str:
        """Changes when the dishes, the allergen list, or the embedding model change."""
        model = getattr(self.embeddings, "model", type(self.embeddings).__name__)
        payload = json.dumps([model, self.allergens, self.dishes], sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _open(self, client):
        fingerprint = self.fingerprint()
        if COLLECTION in [collection.name for collection in client.list_collections()]:
            collection = client.get_collection(COLLECTION)
            if (collection.metadata or {}).get("fingerprint") == fingerprint:
                return collection  # saved and up to date: nothing to embed
            client.delete_collection(COLLECTION)

        collection = client.create_collection(
            COLLECTION,
            embedding_function=None,  # we pass the vectors ourselves
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"fingerprint": fingerprint},
        )
        texts = [dish_text(dish) for dish in self.dishes]
        collection.add(
            ids=[dish["id"] for dish in self.dishes],
            documents=texts,
            embeddings=self.embeddings.embed_documents(texts),
            metadatas=[dish_metadata(dish, self.allergens) for dish in self.dishes],
        )
        return collection

    def _query(self, vector, filters: Filters | None, k: int, skip: str | None = None):
        n_results = min(k + (skip is not None), len(self.dishes))
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            where=(filters or Filters()).where(),
            include=["distances"],
        )
        distances = (results["distances"] or [[]])[0]
        matches = [
            Match(self.by_id[dish_id], 1 - distance)  # cosine distance -> cosine similarity
            for dish_id, distance in zip(results["ids"][0], distances, strict=True)
            if dish_id != skip
        ]
        return matches[:k]

    def search(
        self, query: str, filters: Filters | None = None, k: int | None = None
    ) -> list[Match]:
        """The dishes closest in meaning to the query that pass the filters."""
        return self._query(self.embeddings.embed_query(query), filters, k or self.k)

    def similar(
        self, dish_id: str, filters: Filters | None = None, k: int | None = None
    ) -> list[Match]:
        """Dishes like this one: a search with the dish's own stored vector."""
        stored = self.collection.get(ids=[dish_id], include=["embeddings"])["embeddings"]
        if stored is None or len(stored) == 0:
            raise KeyError(f"No dish with id {dish_id!r}")
        return self._query(stored[0], filters, k or self.k, skip=dish_id)

    def rewrite(self, question: str, history: Sequence[BaseMessage]) -> str:
        """Turn a follow-up question into a standalone question."""
        chain = REWRITE_PROMPT | self.llm.with_structured_output(StandaloneQuestion)
        result = chain.invoke({"history": list(history), "question": question})
        return StandaloneQuestion.model_validate(result).question

    def ask(
        self, question: str, history: Sequence[BaseMessage] = (), filters: Filters | None = None
    ) -> tuple[list[Match], Iterator[str]]:
        """Find matching dishes and start streaming a recommendation.

        Returns the dishes that went into the prompt and an iterator of answer pieces.
        A follow-up question is rewritten, and the user's previous message is searched too,
        so the dishes from the last answer stay in the prompt with their real numbers.
        """
        history = list(history)[-self.max_history :]
        queries = [question]
        if history:
            earlier = [message.text for message in history if message.type == "human"]
            queries = [self.rewrite(question, history), *earlier[-1:]]
        best: dict[str, Match] = {}
        for query in queries:
            for match in self.search(query, filters):
                if (
                    match.dish["id"] not in best
                    or match.similarity > best[match.dish["id"]].similarity
                ):
                    best[match.dish["id"]] = match
        matches = sorted(best.values(), key=lambda match: match.similarity, reverse=True)
        if not matches:
            return [], iter([NO_MATCH])
        chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        context = "\n".join(dish_card(match.dish) for match in matches)
        return matches, chain.stream({"context": context, "history": history, "question": question})
