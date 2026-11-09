"""Message Analyzer: a small Flask web app.

Paste a customer message, pick one or more local models, and compare how each model
summarises it, scores the sentiment, and drafts a reply.

Run it from the repo root:

    uv run python 01-llm-foundations/message-analyzer/app.py
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import ollama
from analyzer import MAX_MESSAGE_CHARS, AnalysisError, analyze, list_chat_models
from flask import Flask, jsonify, render_template, request

from roadmap import get_settings

MAX_MODELS = 3

log = logging.getLogger(__name__)


def create_app(
    analyze_fn: Callable[[str, str], dict[str, Any]] = analyze,
    models_fn: Callable[[], list[str]] = list_chat_models,
) -> Flask:
    """Build the app. Tests pass fake functions so no Ollama server is needed."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html", max_chars=MAX_MESSAGE_CHARS, max_models=MAX_MODELS)

    @app.get("/api/models")
    def models():
        try:
            return jsonify(models=models_fn(), default=get_settings().chat_model)
        except ConnectionError:
            return jsonify(error="Cannot reach Ollama. Is it running? Try: ollama serve"), 503

    @app.post("/api/analyze")
    def analyze_message():
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        models = data.get("models") or [get_settings().chat_model]

        if not message:
            return jsonify(error="Please enter a message."), 400
        if len(message) > MAX_MESSAGE_CHARS:
            return jsonify(error=f"Messages can be at most {MAX_MESSAGE_CHARS} characters."), 400
        if (
            not isinstance(models, list)
            or not 1 <= len(models) <= MAX_MODELS
            or not all(isinstance(model, str) for model in models)
        ):
            return jsonify(error=f"Choose between 1 and {MAX_MODELS} models."), 400

        results = []
        for model in models:
            try:
                results.append(analyze_fn(message, model))
            except (AnalysisError, ollama.ResponseError, ConnectionError) as error:
                log.warning("Analysis with %s failed: %s", model, error)
                results.append({"model": model, "error": str(error)})
        return jsonify(results=results)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=int(os.getenv("PORT", "5050")))
