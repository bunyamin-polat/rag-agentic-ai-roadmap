"""Brightleaf Kitchen: search the canteen menu by meaning, find similar dishes, ask for advice.

Run it from the repo root:

    uv run python 03-vector-databases/food-recommender/app.py
"""

from __future__ import annotations

import os

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from recommender import Filters, FoodRecommender, Match, diet_labels, load_menu

from roadmap import OLLAMA_ERRORS, explain_ollama_error

MENU = load_menu()
CUISINES = sorted({dish["cuisine"] for dish in MENU["dishes"]})
COURSES = ["any", "breakfast", "starter", "soup", "salad", "main", "dessert"]
DISH_IDS = {dish["name"]: dish["id"] for dish in sorted(MENU["dishes"], key=lambda d: d["name"])}
MAX_CALORIES = 1000  # the slider's top value means "no limit"
SOURCES_TITLE = "🍽️ Dishes used"

_recommender: FoodRecommender | None = None


def get_recommender() -> FoodRecommender:
    """Open the saved menu index on first use (it embeds the menu only if needed)."""
    global _recommender
    if _recommender is None:
        _recommender = FoodRecommender(MENU)
    return _recommender


def make_filters(cuisines, course, max_calories, diet, max_spice, avoid) -> Filters:
    """Turn the filter panel's values into a Filters object."""
    return Filters(
        cuisines=tuple(cuisines or ()),
        course=None if course in (None, "any") else course,
        max_calories=None if max_calories >= MAX_CALORIES else int(max_calories),
        vegetarian="vegetarian" in (diet or ()),
        vegan="vegan" in (diet or ()),
        gluten_free="gluten-free" in (diet or ()),
        max_spice=None if max_spice >= 3 else int(max_spice),
        avoid=tuple(avoid or ()),
    )


def results_table(matches: list[Match], max_calories: int | None = None) -> str:
    """A table of dishes. With a calorie limit, it also shows how much of the budget is left."""
    if not matches:
        return "*No dish matches. Try other words or fewer filters.*"
    rows = [
        "| Match | Dish | Cuisine | Course | kcal | Diet | Spice | Allergens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for match in matches:
        dish = match.dish
        kcal = str(dish["calories"])
        if max_calories is not None:
            kcal += f" ({max_calories - dish['calories']} left)"
        rows.append(
            f"| {match.similarity:.0%} | **{dish['name']}** | {dish['cuisine']} | {dish['course']} "
            f"| {kcal} | {', '.join(diet_labels(dish)) or '-'} "
            f"| {'🌶️' * dish['spice'] or '-'} | {', '.join(dish['allergens']) or '-'} |"
        )
    return "\n".join(rows)


def search(query: str, k: float, *filter_values) -> str:
    query = (query or "").strip()
    if not query:
        return "*Describe what you feel like eating.*"
    filters = make_filters(*filter_values)
    try:
        matches = get_recommender().search(query, filters, k=int(k))
    except OLLAMA_ERRORS as error:
        return explain_ollama_error(error)
    return results_table(matches, filters.max_calories)


def similar(dish_name: str | None, k: float, *filter_values) -> str:
    if not dish_name:
        return "*Pick a dish to see similar ones.*"
    filters = make_filters(*filter_values)
    try:
        matches = get_recommender().similar(DISH_IDS[dish_name], filters, k=int(k))
    except OLLAMA_ERRORS as error:
        return explain_ollama_error(error)
    return results_table(matches, filters.max_calories)


def text_of(content) -> str:
    """Gradio sends message content either as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


def to_langchain(history: list[dict]) -> list[BaseMessage]:
    """Turn the Gradio chat history into LangChain messages, without the dish boxes."""
    messages: list[BaseMessage] = []
    for message in history:
        if (message.get("metadata") or {}).get("title") == SOURCES_TITLE:
            continue
        text = text_of(message["content"])
        messages.append(HumanMessage(text) if message["role"] == "user" else AIMessage(text))
    return messages


def respond(question: str, history: list[dict], *filter_values):
    """Stream the recommendation into the chat, then add the dishes it was based on."""
    question = (question or "").strip()
    if not question:
        yield history, ""
        return
    history = [*history, {"role": "user", "content": question}]
    try:
        matches, stream = get_recommender().ask(
            question, to_langchain(history[:-1]), make_filters(*filter_values)
        )
        answer = ""
        for piece in stream:
            if piece:
                answer += piece
                yield [*history, {"role": "assistant", "content": answer}], ""
    except OLLAMA_ERRORS as error:
        yield [*history, {"role": "assistant", "content": explain_ollama_error(error)}], ""
        return

    reply = [*history, {"role": "assistant", "content": answer.strip()}]
    if matches:
        names = ", ".join(f"{match.dish['name']} ({match.similarity:.0%})" for match in matches)
        reply.append(
            gr.ChatMessage(
                role="assistant",
                content=names,
                metadata={"title": SOURCES_TITLE, "status": "done"},
            )
        )
    yield reply, ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Brightleaf Kitchen") as app:
        gr.Markdown(
            "# 🥗 Brightleaf Kitchen\n"
            "Find dishes on the canteen menu by meaning, not by keywords. The filters on the left "
            "apply to every tab. The menu is made up."
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Filters")
                cuisines = gr.CheckboxGroup(CUISINES, label="Cuisine (none = all)")
                course = gr.Dropdown(COURSES, value="any", label="Course")
                max_calories = gr.Slider(
                    200,
                    MAX_CALORIES,
                    value=MAX_CALORIES,
                    step=50,
                    label="Max calories (1000 = any)",
                )
                diet = gr.CheckboxGroup(["vegetarian", "vegan", "gluten-free"], label="Diet")
                max_spice = gr.Slider(0, 3, value=3, step=1, label="Max spice (0 = not spicy)")
                avoid = gr.CheckboxGroup(MENU["allergens"], label="Leave out allergens")
                k = gr.Slider(1, 10, value=5, step=1, label="Results in Search and Similar (k)")
            with gr.Column(scale=3):
                with gr.Tab("Search"):
                    query = gr.Textbox(
                        placeholder="What do you feel like eating?",
                        show_label=False,
                        submit_btn=True,
                    )
                    gr.Examples(
                        ["a warm lentil soup", "something fresh and crunchy", "a sweet treat"],
                        inputs=query,
                    )
                    results = gr.Markdown()
                with gr.Tab("Similar dishes"):
                    dish = gr.Dropdown(list(DISH_IDS), value=None, label="Dish")
                    similar_results = gr.Markdown("*Pick a dish to see similar ones.*")
                with gr.Tab("Ask"):
                    chatbot = gr.Chatbot(label="Ask the canteen", height=540)
                    question = gr.Textbox(
                        placeholder="Ask for a recommendation ...",
                        show_label=False,
                        submit_btn=True,
                    )
                    gr.Examples(
                        [
                            "I'd like a warm vegan soup for lunch.",
                            "Which one has the most protein?",
                            "Is the pad thai OK with a nut allergy?",
                        ],
                        inputs=question,
                    )

        filters = [cuisines, course, max_calories, diet, max_spice, avoid]
        query.submit(search, [query, k, *filters], results)
        dish.change(similar, [dish, k, *filters], similar_results)
        for component in [k, *filters]:  # update the search and the similar dishes at once
            component.change(search, [query, k, *filters], results)
            component.change(similar, [dish, k, *filters], similar_results)
        question.submit(respond, [question, chatbot, *filters], [chatbot, question])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7862")))
