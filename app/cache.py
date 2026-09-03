"""
Exact-match Redis caching, used to wrap query_data and generate_chart
(deterministic: same metric+groupby always produces the same answer).
narrate_insight uses semantic caching instead — see app/semantic_cache.py.

Cache invalidation is intentionally minimal, per the project's design
decision: entries just expire via TTL rather than being explicitly
flushed when the underlying data changes (e.g. after an ETL re-run).

Also holds a request-scoped flag (via contextvars, so it's safe under
FastAPI's concurrent requests) that any cache layer can set on a hit —
app/main.py reads it after the orchestrator loop finishes and reports
it back to the UI, which is what powers the "served from cache" note.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
from typing import Any, Callable

import redis

from app.config import REDIS_URL

CACHE_TTL_SECONDS = 600  # 10 minutes — simple TTL, no explicit invalidation

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def make_cache_key(tool_name: str, **kwargs: Any) -> str:
    """Deterministic key from the tool name + its sorted arguments."""
    payload = json.dumps(kwargs, sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"cache:{tool_name}:{digest}"


# --- request-scoped cache-hit flag ---
_cache_hit_var: contextvars.ContextVar[bool] = contextvars.ContextVar("cache_hit", default=False)


def reset_cache_hit_flag() -> None:
    """Call at the start of each request, before the orchestrator runs."""
    _cache_hit_var.set(False)


def mark_cache_hit() -> None:
    _cache_hit_var.set(True)


def was_served_from_cache() -> bool:
    return _cache_hit_var.get()


def with_exact_cache(tool_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wraps a tool function with exact-match Redis caching. On a hit,
    marks the request as cache-served and returns the cached result
    without calling fn. On a miss, calls fn, caches the result, and
    returns it.
    """

    def wrapped(**kwargs: Any) -> Any:
        client = get_redis_client()
        key = make_cache_key(tool_name, **kwargs)

        cached_raw = client.get(key)
        if cached_raw is not None:
            mark_cache_hit()
            return json.loads(cached_raw)

        result = fn(**kwargs)
        client.set(key, json.dumps(result), ex=CACHE_TTL_SECONDS)
        return result

    return wrapped