from unittest.mock import patch

from app.cache import (
    make_cache_key,
    reset_cache_hit_flag,
    was_served_from_cache,
    with_exact_cache,
)


class FakeRedis:
    """Minimal in-memory stand-in for redis.Redis, supporting only get/set."""

    def __init__(self):
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value


def test_make_cache_key_is_deterministic_and_order_independent():
    key1 = make_cache_key("query_data", metric="readmission_rate", groupby="gender")
    key2 = make_cache_key("query_data", groupby="gender", metric="readmission_rate")
    assert key1 == key2  # same kwargs, different order -> same key


def test_make_cache_key_differs_for_different_tools_or_args():
    key1 = make_cache_key("query_data", metric="readmission_rate", groupby="gender")
    key2 = make_cache_key("query_data", metric="readmission_rate", groupby="race")
    key3 = make_cache_key("generate_chart", metric="readmission_rate", groupby="gender")
    assert key1 != key2
    assert key1 != key3


@patch("app.cache.get_redis_client")
def test_cache_miss_calls_underlying_function_and_stores_result(mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    call_count = {"n": 0}

    def real_fn(**kwargs):
        call_count["n"] += 1
        return {"result": "computed"}

    cached_fn = with_exact_cache("query_data", real_fn)
    result = cached_fn(metric="readmission_rate", groupby="gender")

    assert result == {"result": "computed"}
    assert call_count["n"] == 1
    assert was_served_from_cache() is False
    # something should now be stored under the deterministic key
    key = make_cache_key("query_data", metric="readmission_rate", groupby="gender")
    assert fake_redis.get(key) is not None


@patch("app.cache.get_redis_client")
def test_cache_hit_skips_underlying_function_and_marks_cache_hit(mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    call_count = {"n": 0}

    def real_fn(**kwargs):
        call_count["n"] += 1
        return {"result": "computed"}

    cached_fn = with_exact_cache("query_data", real_fn)

    first = cached_fn(metric="readmission_rate", groupby="gender")
    second = cached_fn(metric="readmission_rate", groupby="gender")

    assert first == second
    assert call_count["n"] == 1  # underlying function only ran once
    assert was_served_from_cache() is True


@patch("app.cache.get_redis_client")
def test_different_arguments_are_not_confused_in_cache(mock_get_client):
    fake_redis = FakeRedis()
    mock_get_client.return_value = fake_redis
    reset_cache_hit_flag()

    def real_fn(**kwargs):
        return kwargs

    cached_fn = with_exact_cache("query_data", real_fn)

    result_gender = cached_fn(metric="readmission_rate", groupby="gender")
    result_race = cached_fn(metric="readmission_rate", groupby="race")

    assert result_gender != result_race