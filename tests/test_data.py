"""Tests for the data-shape utilities that don't hit the HF Hub."""
from factorybench.data import _infer_answer_format, _resolve_levels
from factorybench.types import AnswerFormat


def test_infer_scalar_range():
    fmt = _infer_answer_format({
        "options": {}, "answer": 0,
        "acceptance_bounds": {"min": 0, "max": 297},
    })
    assert fmt is AnswerFormat.SCALAR_RANGE


def test_infer_scalar_margin():
    fmt = _infer_answer_format({
        "options": {}, "answer": -90.5,
        "acceptance_bounds": {"signal": "x", "std": 0.003, "margin": 0.005},
    })
    assert fmt is AnswerFormat.SCALAR_MARGIN


def test_infer_scalar_exact():
    fmt = _infer_answer_format({
        "options": {}, "answer": 0.7,
        "acceptance_bounds": {"signal": "x", "steps_ahead": 5, "actual_value": 0.7},
    })
    assert fmt is AnswerFormat.SCALAR_EXACT


def test_infer_tensor_margin():
    fmt = _infer_answer_format({
        "options": {}, "answer": "[1, 2, 3]",
        "acceptance_bounds": {"signal": "x", "std": [0.1, 0.1, 0.1], "margin": [0.05, 0.05, 0.05]},
    })
    assert fmt is AnswerFormat.TENSOR_MARGIN


def test_infer_single_letter_mcq():
    fmt = _infer_answer_format({
        "options": {"A": "a", "B": "b", "C": "c"}, "answer": "C", "acceptance_bounds": None,
    })
    assert fmt is AnswerFormat.SINGLE_LETTER_MCQ


def test_infer_four_letter_tf():
    fmt = _infer_answer_format({
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer": "TFTF", "acceptance_bounds": None,
    })
    assert fmt is AnswerFormat.FOUR_LETTER_TF


def test_infer_four_letter_ranking():
    fmt = _infer_answer_format({
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "answer": "BDCA", "acceptance_bounds": None,
    })
    assert fmt is AnswerFormat.FOUR_LETTER_RANKING


def test_infer_free_form_l4():
    fmt = _infer_answer_format({
        "options": {}, "answer": "Halt the cycle.", "acceptance_bounds": None,
    })
    assert fmt is AnswerFormat.FREE_FORM


def test_resolve_levels_string_all():
    assert _resolve_levels("all") == [1, 2, 3, 4]


def test_resolve_levels_short_form():
    assert _resolve_levels("L2") == [2]
    assert _resolve_levels("l3") == [3]
    assert _resolve_levels("4") == [4]


def test_resolve_levels_int_and_iterable():
    assert _resolve_levels(1) == [1]
    assert _resolve_levels([1, 2, 3]) == [1, 2, 3]


def test_resolve_levels_dedups_preserving_order():
    assert _resolve_levels([2, 1, 2, 3, 1]) == [2, 1, 3]
