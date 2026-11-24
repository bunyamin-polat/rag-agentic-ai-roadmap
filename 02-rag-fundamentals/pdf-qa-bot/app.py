"""PDF Q&A bot: upload a PDF and chat about it. Answers come only from the PDF, with sources.

Run it from the repo root:

    uv run python 02-rag-fundamentals/pdf-qa-bot/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rag import PdfError, PdfQA, sources_markdown

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "data" / "handbook.pdf"
SOURCES_TITLE = "📄 Sources"
START_MESSAGE = "Upload a PDF, or use the sample handbook, to start."

qa = PdfQA()


def text_of(content) -> str:
    """Gradio sends message content either as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


def to_langchain(history: list[dict]) -> list[BaseMessage]:
    """Turn the Gradio chat history into LangChain messages, without the source boxes."""
    messages: list[BaseMessage] = []
    for message in history:
        if (message.get("metadata") or {}).get("title") == SOURCES_TITLE:
            continue
        text = text_of(message["content"])
        messages.append(HumanMessage(text) if message["role"] == "user" else AIMessage(text))
    return messages


def load_pdf_file(path: str | None):
    """Index a new PDF. Returns the file id, a status line, and an empty chat."""
    if not path:
        return None, START_MESSAGE, []
    try:
        info = qa.index(path)
    except PdfError as error:
        return None, f"⚠️ {error}", []
    except ConnectionError:
        return None, "⚠️ Cannot reach Ollama. Is it running? Try: `ollama serve`", []
    status = f"✅ **{info.name}**: {info.pages} pages, {info.chunks} chunks. Ask a question."
    return info.file_id, status, []


def respond(question: str, history: list[dict], file_id: str | None):
    """Answer a question and stream the answer into the chat, then add the sources."""
    question = question.strip()
    if not question:
        yield history, ""
        return
    history = [*history, {"role": "user", "content": question}]
    if not file_id:
        yield [*history, {"role": "assistant", "content": START_MESSAGE}], ""
        return

    try:
        docs, stream = qa.ask(file_id, question, to_langchain(history[:-1]))
        answer = ""
        for piece in stream:
            if piece:  # a reasoning model sends empty pieces while it thinks
                answer += piece
                yield [*history, {"role": "assistant", "content": answer}], ""
    except (PdfError, ConnectionError) as error:
        yield [*history, {"role": "assistant", "content": f"⚠️ {error}"}], ""
        return

    sources = gr.ChatMessage(
        role="assistant",
        content=sources_markdown(docs),
        metadata={"title": SOURCES_TITLE, "status": "done"},  # "done" shows it closed
    )
    yield [*history, {"role": "assistant", "content": answer}, sources], ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PDF Q&A Bot") as app:
        gr.Markdown(
            "# PDF Q&A Bot\n"
            "Upload a PDF and ask questions about it. Answers come only from the PDF, "
            "and every answer shows the pages it used."
        )
        file_id = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                pdf = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                sample = gr.Button("Use the sample handbook")
                status = gr.Markdown(START_MESSAGE)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat", height=520)
                question = gr.Textbox(
                    placeholder="Ask a question about the PDF ...",
                    show_label=False,
                    submit_btn=True,
                )
                gr.Examples(
                    examples=[
                        "Can I eat in company vehicles?",
                        "Who pays if I get a parking ticket with one?",
                        "How many vacation days do I get?",
                    ],
                    inputs=question,
                    label="Example questions for the sample handbook",
                )

        pdf.change(load_pdf_file, inputs=pdf, outputs=[file_id, status, chatbot])
        sample.click(lambda: str(SAMPLE_PDF), outputs=pdf)
        question.submit(respond, inputs=[question, chatbot, file_id], outputs=[chatbot, question])
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7860")))
