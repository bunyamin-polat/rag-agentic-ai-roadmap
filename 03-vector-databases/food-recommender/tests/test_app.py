"""Tests for the Gradio handlers in app.py, with a fake recommender: no Ollama server needed."""

import app
import ollama
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from recommender import Filters, Match

DAL = next(dish for dish in app.MENU["dishes"] if dish["name"] == "Red lentil dal")
NO_FILTERS = ([], "any", app.MAX_CALORIES, [], 3, [])


class FakeRecommender:
    def __init__(self):
        self.seen = {}

    def search(self, query, filters, k=5):
        self.seen["search"] = (query, filters, k)
        return [Match(DAL, 0.81)]

    def similar(self, dish_id, filters, k=5):
        self.seen["similar"] = (dish_id, filters, k)
        return [Match(DAL, 0.75)]

    def ask(self, question, history, filters):
        self.seen["ask"] = (question, history, filters)
        return [Match(DAL, 0.8)], iter(["Try the ", "red lentil dal."])


@pytest.fixture
def fake(monkeypatch):
    recommender = FakeRecommender()
    monkeypatch.setattr(app, "get_recommender", lambda: recommender)
    return recommender


def test_make_filters_reads_the_panel():
    filters = app.make_filters(["Indian"], "soup", 400, ["vegan", "gluten-free"], 1, ["nuts"])

    assert filters == Filters(
        cuisines=("Indian",),
        course="soup",
        max_calories=400,
        vegan=True,
        gluten_free=True,
        max_spice=1,
        avoid=("nuts",),
    )


def test_top_slider_values_mean_no_limit():
    assert app.make_filters(*NO_FILTERS) == Filters()


def test_search_shows_a_table(fake):
    table = app.search("lentil soup", 3, *NO_FILTERS)

    assert "**Red lentil dal**" in table and "81%" in table
    assert "left)" not in table  # no calorie limit, no budget
    assert fake.seen["search"] == ("lentil soup", Filters(), 3)


def test_a_calorie_limit_shows_the_calories_left(fake):
    table = app.search("lentil soup", 5, [], "any", 400, [], 3, [])

    assert "| 340 (60 left) |" in table  # the dal has 340 kcal: 60 left of 400


def test_similar_looks_up_the_dish_id(fake):
    app.similar("Red lentil dal", 5, *NO_FILTERS)

    assert fake.seen["similar"][0] == DAL["id"]


def test_respond_streams_then_adds_the_dishes_box(fake):
    history = [
        {"role": "user", "content": [{"type": "text", "text": "A warm soup"}]},
        {"role": "assistant", "content": "Try the dal."},
        {
            "role": "assistant",
            "content": "Red lentil dal (80%)",
            "metadata": {"title": app.SOURCES_TITLE},
        },
    ]

    chat, textbox = list(app.respond("Which one has the most protein?", history, *NO_FILTERS))[-1]

    assert textbox == ""
    assert chat[-2] == {"role": "assistant", "content": "Try the red lentil dal."}
    assert chat[-1].metadata["title"] == app.SOURCES_TITLE
    question, sent_history, _ = fake.seen["ask"]
    assert question == "Which one has the most protein?"
    assert sent_history == [HumanMessage("A warm soup"), AIMessage("Try the dal.")]  # no dish box


def test_missing_model_says_what_to_pull(monkeypatch):
    class MissingModel:
        def search(self, query, filters, k=5):
            raise ollama.ResponseError("model 'nomic-embed-text' not found", 404)

    monkeypatch.setattr(app, "get_recommender", MissingModel)

    assert "`ollama pull nomic-embed-text`" in app.search("lentil soup", 5, *NO_FILTERS)
