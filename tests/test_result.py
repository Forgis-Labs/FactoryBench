import json
import math
from datetime import timedelta

from factorybench.result import ItemResult, Result


def _item(level, template_id, score, chance, parse_error=None, judge_votes=None, dataset="aursad"):
    return ItemResult(
        id=f"L{level}_{template_id}",
        level=level,
        template_id=template_id,
        template_type="predictive",
        answer_format="scalar_range",
        dataset=dataset,
        fault_id=0,
        raw_output="raw",
        parsed=None,
        score=score,
        chance=chance,
        parse_error=parse_error,
        judge_votes=judge_votes,
    )


def test_score_chance_corrected_aggregate():
    r = Result(
        model_name="m",
        items=[_item(1, 1, 1.0, 0.5), _item(1, 1, 0.5, 0.5)],
    )
    # mean_acc=0.75, mean_chance=0.5 -> (0.75-0.5)/0.5 = 0.5
    assert r.score == 0.5
    assert r.raw_accuracy == 0.75


def test_empty_score_is_nan():
    r = Result(model_name="m", items=[])
    assert math.isnan(r.score)
    assert math.isnan(r.raw_accuracy)


def test_parse_failures_excluded_from_score():
    r = Result(
        model_name="m",
        items=[_item(1, 1, 1.0, 0.0), _item(1, 1, float("nan"), 0.0, parse_error="boom")],
    )
    # Only the first (clean) item contributes.
    assert r.raw_accuracy == 1.0
    assert len(r.parse_failures()) == 1


def test_by_level_template_dataset():
    r = Result(model_name="m", items=[
        _item(1, 1, 1.0, 0.0),
        _item(1, 1, 0.0, 0.0),
        _item(2, 7, 0.5, 0.5),
    ])
    assert r.by_level() == {"L1": 0.5, "L2": 0.0}
    assert r.by_template() == {"L1.1": 0.5, "L2.7": 0.0}
    assert r.by_dataset() == {"aursad": 0.4}  # mean_acc=0.5, chance=mean(0,0,0.5)/3 = 0.167


def test_save_includes_judges_and_tokens(tmp_path):
    r = Result(
        model_name="m",
        items=[_item(1, 1, 1.0, 0.0)],
        wall_time=timedelta(seconds=12.5),
        cost=0.123,
        judges=["gpt-5.1", "claude-sonnet-4-6"],
        tokens_used={"candidate": {"model": "m", "input_tokens": 10, "output_tokens": 5, "calls": 1}, "judges": {}},
    )
    out = tmp_path / "r.json"
    r.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["cost"] == 0.123
    assert loaded["tokens_used"]["candidate"]["calls"] == 1
    assert loaded["judges"] == ["gpt-5.1", "claude-sonnet-4-6"]


def test_judge_mode_paper_default_vs_custom_vs_single():
    from factorybench.judges import PAPER_DEFAULT_JUDGES
    r_paper = Result(model_name="m", items=[], judges=list(PAPER_DEFAULT_JUDGES))
    r_single = Result(model_name="m", items=[], judges=["gpt-5.1"])
    r_custom = Result(model_name="m", items=[], judges=["gpt-5.1", "claude-sonnet-4-6"])
    r_none = Result(model_name="m", items=[])
    assert r_paper.judge_mode() == "paper-default"
    assert r_single.judge_mode() == "single-judge"
    assert r_custom.judge_mode() == "custom"
    assert r_none.judge_mode() is None


def test_l4_items_and_fleiss_kappa():
    r = Result(
        model_name="m",
        items=[
            _item(4, 1, 1.0, 0.0, judge_votes=[{"judge": "j1", "score": 1.0}, {"judge": "j2", "score": 1.0}, {"judge": "j3", "score": 1.0}]),
            _item(4, 1, 1.0, 0.0, judge_votes=[{"judge": "j1", "score": 1.0}, {"judge": "j2", "score": 1.0}, {"judge": "j3", "score": 1.0}]),
        ],
        judges=["j1", "j2", "j3"],
    )
    assert len(r.l4_items()) == 2
    # All identical -> kappa undefined (Pe=1); library returns nan in that case.
    kappa = r.fleiss_kappa()
    assert math.isnan(kappa)


def test_result_load_round_trip(tmp_path):
    """Result.load is the documented inverse of Result.save."""
    original = Result(
        model_name="m",
        items=[_item(1, 7, 0.5, 0.5, parse_error=None)],
        wall_time=timedelta(seconds=12.5),
        cost=0.456,
        judges=["gpt-5.1", "claude-sonnet-4-6"],
        tokens_used={"candidate": {"model": "m", "input_tokens": 100, "output_tokens": 10, "calls": 1}, "judges": {}},
    )
    p = tmp_path / "r.json"
    original.save(p)

    loaded = Result.load(p)

    assert loaded.model_name == "m"
    assert loaded.cost == 0.456
    assert loaded.judges == ["gpt-5.1", "claude-sonnet-4-6"]
    assert loaded.tokens_used["candidate"]["calls"] == 1
    assert len(loaded.items) == 1
    assert loaded.items[0].id == original.items[0].id
    assert loaded.items[0].score == original.items[0].score
    assert loaded.wall_time.total_seconds() == 12.5


def test_result_load_preserves_parse_errors(tmp_path):
    bad = _item(2, 1, float("nan"), 0.0, parse_error="could not extract a number")
    p = tmp_path / "r.json"
    Result(model_name="m", items=[bad]).save(p)
    loaded = Result.load(p)
    assert loaded.items[0].parse_error == "could not extract a number"
    assert math.isnan(loaded.items[0].score)
    assert len(loaded.parse_failures()) == 1


def test_fleiss_kappa_perfect_agreement_three_categories():
    r = Result(
        model_name="m",
        items=[
            _item(4, 1, 1.0, 0.0, judge_votes=[{"judge": "j1", "score": 1.0}, {"judge": "j2", "score": 1.0}, {"judge": "j3", "score": 1.0}]),
            _item(4, 1, 0.5, 0.0, judge_votes=[{"judge": "j1", "score": 0.5}, {"judge": "j2", "score": 0.5}, {"judge": "j3", "score": 0.5}]),
            _item(4, 1, 0.0, 0.0, judge_votes=[{"judge": "j1", "score": 0.0}, {"judge": "j2", "score": 0.0}, {"judge": "j3", "score": 0.0}]),
        ],
        judges=["j1", "j2", "j3"],
    )
    # P_bar = 1 across all items, Pe < 1, so kappa = 1.
    assert r.fleiss_kappa() == 1.0
