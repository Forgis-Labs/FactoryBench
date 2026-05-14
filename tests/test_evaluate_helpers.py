"""Tests for evaluate.py internal helpers that don't hit the HF Hub."""
import json
import math
from datetime import timedelta

from factorybench.evaluate import (
    _adapter_model_id, _is_resumable, _load_resume, _summarize_usage,
)
from factorybench.result import ItemResult, Result


def _ok_item(item_id="x"):
    return ItemResult(
        id=item_id, level=1, template_id=1, template_type="predictive",
        answer_format="scalar_range", dataset="aursad", fault_id=0,
        raw_output="0", parsed=0, score=1.0, chance=0.0, parse_error=None,
    )


def test_is_resumable_clean_item():
    assert _is_resumable(_ok_item()) is True


def test_is_resumable_rejects_parse_error():
    it = _ok_item()
    it.parse_error = "boom"
    assert _is_resumable(it) is False


def test_is_resumable_rejects_nan_score():
    it = _ok_item()
    it.score = float("nan")
    assert _is_resumable(it) is False


def test_load_resume_round_trip(tmp_path):
    r = Result(model_name="m", items=[_ok_item("a"), _ok_item("b")])
    p = tmp_path / "r.json"
    r.save(p)
    loaded = _load_resume(p)
    assert set(loaded.keys()) == {"a", "b"}
    assert loaded["a"].score == 1.0


def test_load_resume_preserves_parse_error(tmp_path):
    bad = _ok_item("c")
    bad.parse_error = "yikes"
    bad.score = float("nan")
    r = Result(model_name="m", items=[bad])
    p = tmp_path / "r.json"
    r.save(p)
    loaded = _load_resume(p)
    assert loaded["c"].parse_error == "yikes"
    assert math.isnan(loaded["c"].score)


def test_adapter_model_id_uses_property():
    class A:
        model_name = "claude-opus-4-7"
    assert _adapter_model_id(A(), fallback="x") == "claude-opus-4-7"


def test_adapter_model_id_falls_back():
    class A: pass
    assert _adapter_model_id(A(), fallback="fallback") == "fallback"


def test_summarize_usage_aggregates_candidate():
    usages = [
        {"input_tokens": 100, "output_tokens": 10, "model": "m"},
        {"input_tokens": 200, "output_tokens": 20, "model": "m"},
        None,  # adapter without usage
    ]
    summary = _summarize_usage(
        candidate_model_id="m", candidate_usages=usages, panel=None,
    )
    assert summary["candidate"]["input_tokens"] == 300
    assert summary["candidate"]["output_tokens"] == 30
    assert summary["candidate"]["calls"] == 2
    assert summary["judges"] == {}
