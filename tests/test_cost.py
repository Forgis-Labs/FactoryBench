import json

import pytest

from factorybench.cost import (
    BUNDLED_PRICES_PATH, PRICES_METADATA, PRICES_PER_M_TOKENS, _dollars,
    _price_of, compute_cost_from_usage, estimate_tokens, set_price,
)


def test_set_price_then_lookup():
    set_price("test-only-model", input_per_m=0.10, output_per_m=0.20)
    assert _price_of("test-only-model") == (0.10, 0.20)


def test_unknown_model_uses_default():
    # Default fallback rates (in cost.py: $5/$15).
    inp, out = _price_of("nonexistent-x9z")
    assert inp == 5.00 and out == 15.00


def test_normalized_dotted_alias():
    # claude-sonnet-4.6 -> claude-sonnet-4-6 (dot -> dash).
    assert _price_of("claude-sonnet-4.6") == PRICES_PER_M_TOKENS["claude-sonnet-4-6"]


def test_estimate_tokens_contract():
    # When tiktoken is installed estimate_tokens() routes through it; otherwise
    # it uses the len/4 heuristic. In both cases: empty/None -> 0, non-empty -> >= 1.
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("a" * 40) >= 1


def test_dollars_arithmetic():
    # 1M input tokens at $3/M + 1M output at $15/M = $18 total.
    assert _dollars(1_000_000, 1_000_000, (3.0, 15.0)) == pytest.approx(18.0)


def test_compute_cost_from_usage_no_judges():
    usage = {
        "candidate": {"model": "gpt-4o", "input_tokens": 10000, "output_tokens": 500, "calls": 10},
        "judges": {},
    }
    # gpt-4o is $2.50/$10.00 per M.
    expected = 10000 / 1_000_000 * 2.50 + 500 / 1_000_000 * 10.00
    assert compute_cost_from_usage(usage) == pytest.approx(expected)


def test_compute_cost_from_usage_with_judges():
    usage = {
        "candidate": {"model": "deepseek-v3.2", "input_tokens": 2000, "output_tokens": 150, "calls": 3},
        "judges": {
            "j1": {"model": "claude-sonnet-4-6", "input_tokens": 800, "output_tokens": 3, "calls": 3},
            "j2": {"model": "gpt-5.1", "input_tokens": 800, "output_tokens": 3, "calls": 3},
        },
    }
    expected = (
        2000 / 1_000_000 * 0.27 + 150 / 1_000_000 * 1.10
        + 800 / 1_000_000 * 3.00 + 3 / 1_000_000 * 15.00
        + 800 / 1_000_000 * 5.00 + 3 / 1_000_000 * 15.00
    )
    assert compute_cost_from_usage(usage) == pytest.approx(expected)


def test_compute_cost_empty_usage():
    assert compute_cost_from_usage({}) == 0.0
    assert compute_cost_from_usage(None) == 0.0


def test_bundled_prices_loaded_at_import():
    # Sanity: the shipped JSON populates PRICES_PER_M_TOKENS at import time.
    assert "claude-sonnet-4-6" in PRICES_PER_M_TOKENS
    assert PRICES_METADATA.get("version")
    assert PRICES_METADATA.get("source")


def test_bundled_json_parses(tmp_path):
    # Shipped file is valid JSON with the expected shape.
    payload = json.loads(BUNDLED_PRICES_PATH.read_text(encoding="utf-8"))
    assert "models" in payload
    assert "default_rate" in payload
    for model, rates in payload["models"].items():
        assert "input_per_m" in rates
        assert "output_per_m" in rates


def test_user_override_takes_precedence(tmp_path, monkeypatch):
    """Drop a fake user-config prices file; it should layer on top of the bundle."""
    # Point USER_PRICES_PATH at a temporary file before re-loading.
    user_path = tmp_path / "prices.json"
    user_path.write_text(json.dumps({
        "version": "user-test-2026",
        "updated_at": "2026-12-01T00:00:00Z",
        "models": {
            "claude-sonnet-4-6": {"input_per_m": 99.99, "output_per_m": 99.99},
            "my-secret-model": {"input_per_m": 0.01, "output_per_m": 0.02},
        },
    }))
    import factorybench.cost as cost_mod
    monkeypatch.setattr(cost_mod, "USER_PRICES_PATH", user_path)
    cost_mod._load_bundled_prices()
    try:
        assert cost_mod.PRICES_PER_M_TOKENS["claude-sonnet-4-6"] == (99.99, 99.99)
        assert cost_mod.PRICES_PER_M_TOKENS["my-secret-model"] == (0.01, 0.02)
        # Bundled entries not in the override are still present.
        assert "gpt-5.1" in cost_mod.PRICES_PER_M_TOKENS
        # ``source`` is the file path(s); ``source_description`` is the human label.
        assert "user override" in cost_mod.PRICES_METADATA["source_description"]
        assert str(user_path) in cost_mod.PRICES_METADATA["source"]
    finally:
        # Restore the bundled-only state for other tests.
        cost_mod._load_bundled_prices()
