"""
Semantic caching for the narration tool. Unlike query_data/generate_chart
(exact-match — same metric+groupby always means the same answer),
narration is free-text: "why the spike?" and "explain the jump" should
both hit the same cached explanation even though the wording differs.

Uses a small pretrained sentence-transformer (built on PyTorch) to
embed the question, then does a similarity search against previously
cached narrations for the same metric+groupby slice. Entries are
stored as a small JSON list under one Redis key per slice — this
avoids needing a Redis vector-search module, which is fine at the
scale a project's cache will ever hold (a handful of question variants
per metric/groupby pair, not millions of vectors).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.cache import CACHE_TTL_SECONDS, get_redis_client, mark_cache_hit

SIMILARITY_THRESHOLD = 0.85
MAX_CACHED_ENTRIES_PER_SLICE = 20  # caps how many question variants we keep per metric/groupby

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> list[float]:
    """Encodes text to a normalized embedding vector (PyTorch under the hood)."""
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    # both vectors are pre-normalized (unit length), so dot product == cosine similarity
    return float(np.dot(np.array(a), np.array(b)))


def _slice_key(metric: str, groupby: str) -> str:
    return f"narration_cache:{metric}:{groupby}"


def find_similar_narration(metric: str, groupby: str, question: str) -> str | None:
    """
    Returns a cached narration if a sufficiently similar question was
    asked before for this metric/groupby slice, else None. Marks the
    request as cache-served on a hit.
    """
    client = get_redis_client()
    raw = client.get(_slice_key(metric, groupby))
    if raw is None:
        return None

    entries: list[dict[str, Any]] = json.loads(raw)
    if not entries:
        return None

    question_embedding = embed(question)
    best_score, best_entry = -1.0, None
    for entry in entries:
        score = _cosine_similarity(question_embedding, entry["embedding"])
        if score > best_score:
            best_score, best_entry = score, entry

    if best_entry is not None and best_score >= SIMILARITY_THRESHOLD:
        mark_cache_hit()
        return best_entry["narration"]
    return None


def store_narration(metric: str, groupby: str, question: str, narration: str) -> None:
    """Adds a new question/narration pair to the cache for this slice."""
    client = get_redis_client()
    key = _slice_key(metric, groupby)

    raw = client.get(key)
    entries: list[dict[str, Any]] = json.loads(raw) if raw else []

    entries.append({"question": question, "embedding": embed(question), "narration": narration})
    entries = entries[-MAX_CACHED_ENTRIES_PER_SLICE:]  # cap growth per slice

    client.set(key, json.dumps(entries), ex=CACHE_TTL_SECONDS)