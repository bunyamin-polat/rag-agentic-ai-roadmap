"""Tests for the Flask routes with fake model functions (no Ollama server needed)."""

import pytest
from analyzer import AnalysisError
from app import MAX_MODELS, create_app


def fake_analyze(message, model):
    if model == "broken-model":
        raise AnalysisError("broken-model did not return a valid analysis.")
    return {"model": model, "summary": f"Summary of: {message}", "sentiment": "neutral"}


def fake_models():
    return ["gpt-oss:20b", "qwen3:1.7b"]


@pytest.fixture
def client():
    app = create_app(analyze_fn=fake_analyze, models_fn=fake_models)
    app.config["TESTING"] = True
    return app.test_client()


def test_index_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Message Analyzer" in response.data


def test_models_lists_installed_chat_models(client):
    data = client.get("/api/models").get_json()

    assert data["models"] == ["gpt-oss:20b", "qwen3:1.7b"]
    assert data["default"]


def test_models_reports_when_ollama_is_down():
    def ollama_down():
        raise ConnectionError("connection refused")

    client = create_app(analyze_fn=fake_analyze, models_fn=ollama_down).test_client()

    response = client.get("/api/models")

    assert response.status_code == 503
    assert "Ollama" in response.get_json()["error"]


def test_analyze_runs_every_selected_model(client):
    response = client.post(
        "/api/analyze", json={"message": "Where is my order?", "models": ["a:1b", "b:2b"]}
    )

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert [result["model"] for result in results] == ["a:1b", "b:2b"]
    assert results[0]["summary"] == "Summary of: Where is my order?"


def test_analyze_reports_errors_per_model(client):
    response = client.post(
        "/api/analyze", json={"message": "Hi", "models": ["broken-model", "good-model"]}
    )

    results = response.get_json()["results"]
    assert "error" in results[0]
    assert results[1]["model"] == "good-model"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": "   "},
        {"message": "x" * 5000},
        {"message": "Hi", "models": "gpt-oss:20b"},
        {"message": "Hi", "models": [f"model-{i}" for i in range(MAX_MODELS + 1)]},
    ],
)
def test_analyze_rejects_bad_requests(client, payload):
    response = client.post("/api/analyze", json=payload)

    assert response.status_code == 400
    assert response.get_json()["error"]
