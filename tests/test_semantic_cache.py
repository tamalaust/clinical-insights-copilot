from unittest.mock import patch

import app.semantic_cache as semantic_cache
from app.cache import reset_cache_hit_flag, was_served_from_cache
from app.semantic_cache import find_similar_narration, store_narration


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


# A tiny fake embedding scheme so similarity is fully predictable in
# tests, without downloading or running a real sentence-transformer
# model: each "question" maps to a fixed 2D vector by keyword.
def fake_embed(text: str) -> list[float]:
    text = text.lower()
    if "spike" in text or "jump" in text or "higher" in text:
        return [1.0, 0.0]  # cluster A: "why did it go up" questions
    return [0.0, 1.0]  # cluster B: everything else


@patch("app.semantic_cache.get_redis_client")
@patch("app.semantic_cache.embed", side_effect=fake_embed)
def test_cache_miss_when_no_entries_exist(mock_embed, mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    result = find_similar_narration("readmission_rate", "age", "why the spike?")
    assert result is None
    assert was_served_from_cache() is False


@patch("app.semantic_cache.get_redis_client")
@patch("app.semantic_cache.embed", side_effect=fake_embed)
def test_similar_question_hits_cache(mock_embed, mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    store_narration("readmission_rate", "age", "why the spike?", "Older patients readmit more.")

    # differently worded, but same semantic cluster in our fake embedding
    result = find_similar_narration("readmission_rate", "age", "why the jump?")
    assert result == "Older patients readmit more."
    assert was_served_from_cache() is True


@patch("app.semantic_cache.get_redis_client")
@patch("app.semantic_cache.embed", side_effect=fake_embed)
def test_dissimilar_question_does_not_hit_cache(mock_embed, mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    store_narration("readmission_rate", "age", "why the spike?", "Older patients readmit more.")

    # unrelated question -> different embedding cluster -> should NOT match
    result = find_similar_narration("readmission_rate", "age", "what is the average age?")
    assert result is None
    assert was_served_from_cache() is False


@patch("app.semantic_cache.get_redis_client")
@patch("app.semantic_cache.embed", side_effect=fake_embed)
def test_different_metric_groupby_slices_are_isolated(mock_embed, mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    store_narration("readmission_rate", "age", "why the spike?", "Age-related answer.")

    # same question wording, but a DIFFERENT metric/groupby slice —
    # should not accidentally pull in the other slice's cached answer
    result = find_similar_narration("avg_num_medications", "gender", "why the spike?")
    assert result is None


@patch("app.semantic_cache.get_redis_client")
@patch("app.semantic_cache.embed", side_effect=fake_embed)
def test_max_entries_per_slice_caps_growth(mock_embed, mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    for i in range(semantic_cache.MAX_CACHED_ENTRIES_PER_SLICE + 5):
        store_narration("readmission_rate", "age", f"question variant {i} spike", f"answer {i}")

    import json

    raw = fake_redis.get(semantic_cache._slice_key("readmission_rate", "age"))
    entries = json.loads(raw)
    assert len(entries) == semantic_cache.MAX_CACHED_ENTRIES_PER_SLICE