"""Regression guard for the chance correction.

Two defects once made the main results figure disagree with the body text by
2-3 points on every L1-L3 cell, while the L4 column (uncorrected) matched
exactly:

  1. the correction was floored at zero, so below-chance scores rendered as 0
     and the figure contained no negative values while the prose did;
  2. scripts/generate_figures.py hardcoded E = 0.25 for every single-select
     item, while the benchmark uses E = 1/k with k in {3, 4}. On a 3-option
     item that inflates the corrected score by 10 points.

Both are fixed. These tests exist so neither can come back silently.

Run:  python -m pytest tests/test_chance_correction_is_signed.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.chance_correct import chance_correct, expected_chance_score

FOUR = {"options": dict.fromkeys("ABCD")}
THREE = {"options": dict.fromkeys("ABC")}
SS = "multiple_choice_single_select"


def test_chance_level_comes_from_the_item_not_a_hardcode():
    assert expected_chance_score(SS, FOUR) == 0.25
    assert abs(expected_chance_score(SS, THREE) - 1.0 / 3.0) < 1e-12


def test_below_chance_stays_negative():
    """The property the whole correction exists to provide."""
    assert chance_correct(0.10, SS, FOUR) < 0.0
    assert chance_correct(0.20, SS, THREE) < 0.0


def test_floor_is_minus_e_over_one_minus_e_not_zero():
    assert abs(chance_correct(0.0, SS, FOUR) - (-1.0 / 3.0)) < 1e-12
    assert abs(chance_correct(0.0, SS, THREE) - (-0.5)) < 1e-12


def test_chance_maps_to_exactly_zero():
    assert abs(chance_correct(0.25, SS, FOUR)) < 1e-12
    assert abs(chance_correct(1.0 / 3.0, SS, THREE)) < 1e-12


def test_perfect_maps_to_one():
    assert abs(chance_correct(1.0, SS, FOUR) - 1.0) < 1e-12
    assert abs(chance_correct(1.0, SS, THREE) - 1.0) < 1e-12


def test_hardcoding_e_would_inflate_a_three_option_item():
    """Documents the size of the defect, so its return is obvious."""
    raw = 0.40
    correct = chance_correct(raw, SS, THREE)
    hardcoded = (raw - 0.25) / (1 - 0.25)
    assert hardcoded - correct > 0.09, "expected ~10pp inflation from E=0.25 on k=3"


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except Exception:
                failed += 1
                print("FAIL %s" % name)
                traceback.print_exc()
    print("\nfailures: %d" % failed)
    sys.exit(1 if failed else 0)
