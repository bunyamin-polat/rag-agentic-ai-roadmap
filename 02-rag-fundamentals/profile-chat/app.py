"""Profile chat: pick a person, get icebreakers, then ask questions about their profile.

Run it from the repo root:

    uv run python 02-rag-fundamentals/profile-chat/app.py
"""

from __future__ import annotations

import os

import gradio as gr
from llama_index.core.llms import ChatMessage
from profile_bot import ProfileBot, list_profiles, load_profile

from roadmap import OLLAMA_ERRORS, explain_ollama_error

PROFILES = list_profiles()
SOURCES_TITLE = "📄 Profile sections used"

_bots: dict[str, ProfileBot] = {}
_icebreakers: dict[str, str] = {}


def get_bot(name: str) -> ProfileBot:
    """Build the index for a profile once and reuse it."""
    if name not in _bots:
        _bots[name] = ProfileBot(load_profile(PROFILES[name]))
    return _bots[name]


def profile_card(profile: dict) -> str:
    return (
        f"### {profile['name']}\n"
        f"{profile['headline']} · {profile['location']}\n\n"
        f"*Skills:* {', '.join(profile['skills'][:5])}"
    )


def select_profile(name: str):
    """Show the profile and its icebreakers. Yields twice: first a placeholder, then the result."""
    if not name:
        yield "", "", []
        return
    try:
        bot = get_bot(name)
        card = profile_card(bot.profile)
        if name not in _icebreakers:
            yield card, "*Thinking of icebreakers ...*", []
            _icebreakers[name] = bot.icebreakers()
    except OLLAMA_ERRORS as error:
        yield explain_ollama_error(error), "", []
        return
    yield card, _icebreakers[name], []


def text_of(content) -> str:
    """Gradio sends message content either as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


def to_llama(history: list[dict]) -> list[ChatMessage]:
    """Turn the Gradio chat history into LlamaIndex messages, without the source boxes."""
    return [
        ChatMessage(role=message["role"], content=text_of(message["content"]))
        for message in history
        if (message.get("metadata") or {}).get("title") != SOURCES_TITLE
    ]


def respond(question: str, history: list[dict], name: str):
    """Stream the answer into the chat, then add the profile sections it used."""
    question = question.strip()
    if not question:
        yield history, ""
        return
    history = [*history, {"role": "user", "content": question}]
    if not name:
        yield [*history, {"role": "assistant", "content": "Pick a profile first."}], ""
        return

    try:
        sections, stream = get_bot(name).stream_chat(question, to_llama(history[:-1]))
        answer = ""
        for piece in stream:
            if piece:
                answer += piece
                yield [*history, {"role": "assistant", "content": answer}], ""
    except OLLAMA_ERRORS as error:
        yield [*history, {"role": "assistant", "content": explain_ollama_error(error)}], ""
        return

    sources = gr.ChatMessage(
        role="assistant",
        content=", ".join(dict.fromkeys(sections)),  # unique, in order
        metadata={"title": SOURCES_TITLE, "status": "done"},
    )
    yield [*history, {"role": "assistant", "content": answer.strip()}, sources], ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Profile Chat") as app:
        gr.Markdown(
            "# Profile Chat\n"
            "Pick a person to get three conversation starters, then ask anything about their "
            "profile. All people here are made up."
        )
        with gr.Row():
            with gr.Column(scale=1):
                name = gr.Dropdown(choices=list(PROFILES), value=None, label="Profile")
                card = gr.Markdown()
                gr.Markdown("#### Icebreakers")
                icebreakers = gr.Markdown("Pick a profile.")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=480)
                question = gr.Textbox(
                    placeholder="Ask about this person ...", show_label=False, submit_btn=True
                )
                gr.Examples(
                    examples=[
                        "Where do they work now?",
                        "What did they do before that?",
                        "Which languages do they speak?",
                    ],
                    inputs=question,
                )

        name.change(select_profile, inputs=name, outputs=[card, icebreakers, chatbot])
        question.submit(respond, inputs=[question, chatbot, name], outputs=[chatbot, question])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7861")))
