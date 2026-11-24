"""Tests for profile_bot.py with LlamaIndex's mock models: no Ollama server needed."""

from datetime import date

import profile_bot
import pytest
from llama_index.core import MockEmbedding
from llama_index.core.llms import MockLLM
from profile_bot import ProfileBot, list_profiles, load_profile, profile_to_documents


@pytest.fixture
def lena() -> dict:
    return load_profile(list_profiles()["Lena Okafor"])


@pytest.fixture
def bot(lena) -> ProfileBot:
    return ProfileBot(
        lena,
        llm=MockLLM(max_tokens=20),
        embed_model=MockEmbedding(embed_dim=16),
        today=date(2026, 11, 24),
    )


def test_list_profiles_maps_names_to_files():
    profiles = list_profiles()

    assert set(profiles) == {"Lena Okafor", "Tomás Herrera"}
    assert all(path.suffix == ".json" for path in profiles.values())


def test_documents_cover_every_part_of_the_profile(lena):
    documents = profile_to_documents(lena)
    sections = [doc.metadata["section"] for doc in documents]

    assert sections.count("experience") == 3
    assert sections.count("education") == 2
    assert sections.count("projects") == 2
    for section in (
        "summary",
        "about",
        "career",
        "skills",
        "languages",
        "interests",
        "volunteering",
    ):
        assert section in sections
    assert all("Lena Okafor" in doc.text for doc in documents)  # each text stands on its own


def test_summary_names_the_current_job(lena):
    summary = next(d for d in profile_to_documents(lena) if d.metadata["section"] == "summary")

    assert "Current job: Platform Engineer at Brightleaf, since 2023." in summary.text


def test_career_timeline_lists_jobs_newest_first(lena):
    career = next(d for d in profile_to_documents(lena) if d.metadata["section"] == "career")

    assert career.text.index("Brightleaf") < career.text.index("Parcelpath")
    assert career.text.index("Parcelpath") < career.text.index("City Express Couriers")


@pytest.mark.parametrize(("thinks", "expected"), [(True, "low"), (False, None)])
def test_default_llm_asks_for_thinking_only_when_supported(monkeypatch, thinks, expected):
    monkeypatch.setattr(profile_bot, "supports_thinking", lambda model: thinks)

    assert profile_bot.default_llm().thinking == expected


def test_chat_engine_knows_todays_date(bot):
    engine = bot._chat_engine()

    assert "Today's date is 24 November 2026." in engine._system_prompt
    assert "Lena Okafor" in engine._system_prompt


def test_chat_returns_text_and_source_sections(bot):
    reply = bot.chat("Where does Lena work now?")

    assert reply.text
    assert len(reply.sections) == bot.top_k
    assert all(section for section in reply.sections)


def test_stream_chat_streams_pieces_and_sections(bot):
    sections, stream = bot.stream_chat("Which languages does Lena speak?")

    assert len(sections) == bot.top_k
    assert "".join(stream)


def test_icebreakers_return_text(bot):
    assert bot.icebreakers()
