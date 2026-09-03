"""
Calls the FastAPI backend's /chat endpoint over HTTP, matching the
project's architecture: Streamlit UI -> FastAPI backend (two processes,
not a direct in-process import of the orchestrator).

Kept as a small standalone function so it's testable by mocking
`requests.post` — see tests/test_chat_client.py.
"""
from __future__ import annotations

import requests

from app.config import FASTAPI_BASE_URL


def send_chat_message(message: str, history: list[dict], timeout: int = 60) -> dict:
    """
    Sends a chat message + conversation history to the backend.
    Returns {"answer": str, "from_cache": bool}. Raises
    requests.RequestException on network/HTTP failure — the caller
    (the Streamlit chat tab) is responsible for catching this and
    showing the user a friendly error instead of a stack trace.
    """
    response = requests.post(
        f"{FASTAPI_BASE_URL}/chat",
        json={"message": message, "history": history},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return {"answer": data["answer"], "from_cache": data.get("from_cache", False)}