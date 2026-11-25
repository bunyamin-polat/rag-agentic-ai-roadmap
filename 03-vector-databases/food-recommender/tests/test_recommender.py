"""Tests for recommender.py with fake models: no Ollama server needed."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from recommender import NO_MATCH, Filters, FoodRecommender, dish_card, dish_text, load_menu


class CountingEmbeddings(DeterministicFakeEmbedding):
    """Fake embeddings that count how many texts were embedded."""

    calls: int = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


def fake_llm(answer: str = "Try the red lentil dal.") -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=answer)]))


@pytest.fixture
def menu() -> dict:
    return load_menu()


@pytest.fixture
def recommender(menu, tmp_path) -> FoodRecommender:
    return FoodRecommender(
        menu, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path
    )


def test_menu_is_consistent(menu):
    ids = [dish["id"] for dish in menu["dishes"]]
    assert len(ids) == len(set(ids))
    for dish in menu["dishes"]:
        assert set(dish["allergens"]) <= set(menu["allergens"]), dish["name"]
        assert dish["vegetarian"] or not dish["vegan"], dish["name"]
        assert dish["gluten_free"] == ("gluten" not in dish["allergens"]), dish["name"]


def test_dish_text_has_meaning_and_card_has_facts(menu):
    dal = next(dish for dish in menu["dishes"] if dish["name"] == "Red lentil dal")

    assert "Mildly spicy. Vegan, gluten-free." in dish_text(dal)
    assert "340 kcal, 18 g protein" in dish_card(dal)
    assert "Allergens: none listed." in dish_card(dal)


def test_filters_build_chroma_where():
    assert Filters().where() is None
    assert Filters(vegan=True).where() == {"vegan": True}
    assert Filters(max_calories=0).where() == {"calories": {"$lte": 0}}  # 0 is a limit too
    assert Filters(course="soup", avoid=("nuts",)).where() == {
        "$and": [{"course": "soup"}, {"has_nuts": False}]
    }


def test_search_keeps_only_dishes_that_pass_the_filters(recommender):
    matches = recommender.search("something warm", Filters(vegan=True, max_calories=400))

    assert matches
    assert all(m.dish["vegan"] and m.dish["calories"] <= 400 for m in matches)


def test_search_leaves_out_allergens(recommender):
    matches = recommender.search("a sweet treat", Filters(avoid=("gluten", "milk", "nuts")), k=10)

    assert matches
    assert not any({"gluten", "milk", "nuts"} & set(m.dish["allergens"]) for m in matches)


def test_similar_dishes_skip_the_dish_itself(recommender):
    matches = recommender.similar("dish-14", k=5)

    assert len(matches) == 5
    assert "dish-14" not in [m.dish["id"] for m in matches]


def test_menu_is_embedded_only_once(menu, tmp_path):
    first = FoodRecommender(
        menu, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path
    )
    assert first.embeddings.calls == len(menu["dishes"])

    again = FoodRecommender(
        menu, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path
    )

    assert again.embeddings.calls == 0  # loaded from disk


def test_a_changed_menu_is_embedded_again(menu, tmp_path):
    FoodRecommender(menu, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path)
    changed = {**menu, "dishes": menu["dishes"][:10]}

    updated = FoodRecommender(
        changed, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path
    )

    assert updated.embeddings.calls == 10
    assert updated.collection.count() == 10


def test_a_new_allergen_in_the_list_rebuilds_the_index(menu, tmp_path):
    FoodRecommender(menu, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path)
    more_allergens = {**menu, "allergens": [*menu["allergens"], "mustard"]}

    updated = FoodRecommender(
        more_allergens, embeddings=CountingEmbeddings(size=32), llm=fake_llm(), path=tmp_path
    )

    assert updated.embeddings.calls == len(menu["dishes"])
    stored = updated.collection.get(ids=["dish-01"], include=["metadatas"])["metadatas"][0]
    assert stored["has_mustard"] is False


def test_ask_without_matching_dishes_skips_the_model(recommender, monkeypatch):
    monkeypatch.setattr(recommender, "_llm", None)  # the model must not be needed

    matches, stream = recommender.ask(
        "A dessert", filters=Filters(course="dessert", max_calories=100)
    )

    assert matches == []
    assert "".join(stream) == NO_MATCH


def test_ask_streams_the_answer_with_its_dishes(recommender):
    matches, stream = recommender.ask(
        "A warm vegan soup", filters=Filters(vegan=True, course="soup")
    )

    assert matches and all(m.dish["vegan"] and m.dish["course"] == "soup" for m in matches)
    assert "".join(stream) == "Try the red lentil dal."


def test_first_question_is_searched_as_it_is(recommender, monkeypatch):
    monkeypatch.setattr(recommender, "rewrite", lambda question, history: pytest.fail("no rewrite"))

    recommender.ask("A warm vegan soup")


def test_follow_up_searches_the_rewrite_and_the_previous_message(recommender, monkeypatch):
    monkeypatch.setattr(recommender, "rewrite", lambda question, history: "rewritten question")
    searched = []
    original_search = recommender.search
    monkeypatch.setattr(
        recommender,
        "search",
        lambda query, filters=None: searched.append(query) or original_search(query, filters),
    )
    history = [HumanMessage("A warm vegan soup"), AIMessage("Try the red lentil dal.")]

    recommender.ask("Which one has the most protein?", history)

    assert searched == ["rewritten question", "A warm vegan soup"]
