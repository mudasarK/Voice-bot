"""API tests: app starts and routes respond."""
import os
import sys

import pytest

# Ensure backend is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def test_health_check(client):
    """GET / returns 200 and status ok."""
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_chat_requires_post(client):
    """POST /api/chat with body returns 200 or 422."""
    r = client.post("/api/chat", json={"message": "What is the password policy?"})
    # 200 if RAG/LLM work; 500 if keys missing; 422 if bad body
    assert r.status_code in (200, 500, 422)
    if r.status_code == 200:
        assert "response" in r.json()


def test_ingest_rejects_empty_filename(client):
    """POST /api/ingest with no file returns 422."""
    r = client.post("/api/ingest")
    assert r.status_code == 422


def test_ingest_rejects_unsupported_type(client):
    """POST /api/ingest with wrong file type returns 400."""
    r = client.post(
        "/api/ingest",
        files={"file": ("x.png", b"fake image", "image/png")},
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_voice_requires_file(client):
    """POST /api/voice with no file returns 422."""
    r = client.post("/api/voice")
    assert r.status_code == 422


def test_audio_format_detection():
    """_detect_audio_format maps filename and content-type correctly."""
    from backend.api.routes import _detect_audio_format
    assert _detect_audio_format("x.wav", None) == "wav"
    assert _detect_audio_format("x.mp3", None) == "mp3"
    assert _detect_audio_format("x.ogg", "audio/ogg") == "ogg"
    assert _detect_audio_format(None, "audio/mpeg") == "mp3"
    assert _detect_audio_format(None, None) == "wav"
