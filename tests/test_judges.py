import math

import pytest

import factorybench as fb
from factorybench.judges import (
    JudgePanel, PAPER_DEFAULT_JUDGES, fleiss_kappa, parse_judges_flag,
    precheck_credentials,
)


# ---------------------------------------------------------------- mock judges

class _Recorder:
    """Track per-class call counts via a module-level dict."""
    counts: dict = {}

    @classmethod
    def reset(cls):
        cls.counts = {}


def _mk_judge(spec: str, vote: str, model_id: str | None = None):
    class _J:
        model_name = model_id
        def predict(self, prompt):
            _Recorder.counts[spec] = _Recorder.counts.get(spec, 0) + 1
            return vote
        def predict_with_usage(self, prompt):
            _Recorder.counts[spec] = _Recorder.counts.get(spec, 0) + 1
            return vote, {"input_tokens": 50, "output_tokens": 1, "model": model_id or spec}
    _J.__name__ = f"_Judge_{spec}"
    return _J


# Register once per test module.
@pytest.fixture(autouse=True)
def _register_mock_judges():
    _Recorder.reset()
    if "test-j-1" not in fb.list_models():
        fb.register_model("test-j-1")(_mk_judge("test-j-1", "1", "claude-sonnet-4-6"))
        fb.register_model("test-j-2")(_mk_judge("test-j-2", "0.5", "gpt-5.1"))
        fb.register_model("test-j-3")(_mk_judge("test-j-3", "0", "deepseek-v3.2"))


# ----------------------------------------------------------------- panel core

def test_panel_requires_at_least_one_judge():
    with pytest.raises(ValueError):
        JudgePanel([], cache_dir=None)


def test_paper_default_recognized():
    panel = parse_judges_flag("paper-default")
    assert panel.is_paper_default is True
    assert tuple(panel.judge_specs) == PAPER_DEFAULT_JUDGES


def test_parse_csv():
    panel = parse_judges_flag("gpt-5.1, claude-sonnet-4.6")
    assert panel.judge_specs == ["gpt-5.1", "claude-sonnet-4.6"]
    assert panel.is_single_judge is False


def test_panel_score_median_aggregation(item_l4, tmp_cache_dir):
    panel = JudgePanel(["test-j-1", "test-j-2", "test-j-3"], cache_dir=tmp_cache_dir, concurrency=1)
    res = panel.score(item_l4, "halt and restart")
    # Votes: 1, 0.5, 0 -> median = 0.5
    assert res.score == 0.5
    assert [v.judge for v in res.votes] == ["test-j-1", "test-j-2", "test-j-3"]


def test_panel_cache_reuse(item_l4, tmp_cache_dir):
    panel = JudgePanel(["test-j-1", "test-j-2", "test-j-3"], cache_dir=tmp_cache_dir)
    panel.score(item_l4, "x")
    first = dict(_Recorder.counts)
    panel.score(item_l4, "x")
    # Second call: identical (item, prediction, judges) -> every vote cached.
    assert _Recorder.counts == first


def test_panel_cache_only_no_calls(item_l4, tmp_cache_dir):
    panel = JudgePanel(["test-j-1"], cache_dir=tmp_cache_dir, cache_only=True)
    res = panel.score(item_l4, "never-seen-prediction")
    assert math.isnan(res.score)
    assert res.votes[0].parse_error == "judge_cache_miss"
    # No actual call was made.
    assert _Recorder.counts.get("test-j-1", 0) == 0


def test_panel_usage_tracking(item_l4, tmp_cache_dir):
    panel = JudgePanel(["test-j-1"], cache_dir=tmp_cache_dir)
    panel.score(item_l4, "p")
    usage = panel.usage_by_judge()
    assert usage["test-j-1"]["calls"] == 1
    assert usage["test-j-1"]["model"] == "claude-sonnet-4-6"  # learned from adapter, not spec


def test_panel_parallel_default_concurrency():
    panel = JudgePanel(["a", "b", "c"], cache_dir=None)
    assert panel.concurrency == 3
    panel_eight = JudgePanel(["a"] * 12, cache_dir=None)
    assert panel_eight.concurrency == 8  # capped


def test_precheck_credentials_paper_default_when_keys_missing(monkeypatch):
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    panel = JudgePanel(list(PAPER_DEFAULT_JUDGES), cache_dir=None)
    missing = precheck_credentials(panel)
    assert len(missing) == 3


def test_precheck_credentials_skips_registered_judges():
    panel = JudgePanel(["test-j-1"], cache_dir=None)
    # Registered name doesn't match any provider prefix -> no env requirement.
    assert precheck_credentials(panel) == []


# ----------------------------------------------------------------- fleiss kappa

def test_fleiss_kappa_perfect_agreement():
    # 3 items, all 3 judges agree.
    rows = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
    assert fleiss_kappa(rows) == 1.0


def test_fleiss_kappa_identical_row_undefined():
    # All judges identical AND same value across items -> Pe = 1 -> nan.
    rows = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert math.isnan(fleiss_kappa(rows))


def test_fleiss_kappa_skips_under_two_valid_votes():
    # Items with fewer than 2 valid votes are skipped.
    rows = [[1.0, float("nan"), float("nan")], [0.0, float("nan"), float("nan")]]
    assert math.isnan(fleiss_kappa(rows))
