import math

import pytest

from factorybench.score import chance_corrected, chance_of, score_item


def test_mcq_chance_one_over_n(item_mcq):
    assert chance_of(item_mcq) == pytest.approx(1 / 3)


def test_tf_chance_half(item_tf):
    assert chance_of(item_tf) == 0.5


def test_ranking_chance_one_over_24(item_ranking):
    assert chance_of(item_ranking) == pytest.approx(1 / 24)


def test_scalar_chance_zero(item_scalar_range, item_scalar_margin, item_tensor_margin):
    assert chance_of(item_scalar_range) == 0.0
    assert chance_of(item_scalar_margin) == 0.0
    assert chance_of(item_tensor_margin) == 0.0


def test_mcq_score(item_mcq):
    assert score_item(item_mcq, "C") == 1.0
    assert score_item(item_mcq, "A") == 0.0


def test_tf_partial_credit(item_tf):
    # Truth is "TFTF"; 3 of 4 correct = 0.75.
    assert score_item(item_tf, "TFTT") == pytest.approx(0.75)
    assert score_item(item_tf, "TFTF") == 1.0
    assert score_item(item_tf, "FTFT") == 0.0


def test_ranking_all_or_nothing(item_ranking):
    # Truth is "BDCA"; off-by-one swap = 0, exact = 1.
    assert score_item(item_ranking, "BDCA") == 1.0
    assert score_item(item_ranking, "BDAC") == 0.0


def test_scalar_range(item_scalar_range):
    # Range is [0, 297].
    assert score_item(item_scalar_range, 0.0) == 1.0
    assert score_item(item_scalar_range, 150.0) == 1.0
    assert score_item(item_scalar_range, 297.0) == 1.0
    assert score_item(item_scalar_range, 298.0) == 0.0
    assert score_item(item_scalar_range, -1.0) == 0.0


def test_scalar_margin(item_scalar_margin):
    # Truth -90.5, margin 0.005.
    assert score_item(item_scalar_margin, -90.5) == 1.0
    assert score_item(item_scalar_margin, -90.504) == 1.0
    assert score_item(item_scalar_margin, -90.6) == 0.0


def test_scalar_exact_isclose(item_scalar_exact):
    assert score_item(item_scalar_exact, 0.7276) == 1.0
    # rel_tol is 1e-2 (1%); 0.73 is within 1% of 0.7276.
    assert score_item(item_scalar_exact, 0.73) == 1.0
    assert score_item(item_scalar_exact, 0.5) == 0.0


def test_tensor_margin_element_wise(item_tensor_margin):
    # Truth [1, 2, 3], margin [0.05]*3 -> exact match = 1.0, 1-off = 0/3.
    assert score_item(item_tensor_margin, [1.0, 2.0, 3.0]) == 1.0
    assert score_item(item_tensor_margin, [1.04, 2.04, 3.04]) == 1.0
    # One element outside margin -> 2/3.
    assert score_item(item_tensor_margin, [1.0, 2.0, 5.0]) == pytest.approx(2 / 3)


def test_tensor_wrong_length_zero(item_tensor_margin):
    assert score_item(item_tensor_margin, [1.0, 2.0]) == 0.0
    assert score_item(item_tensor_margin, "not a list") == 0.0


def test_free_form_is_nan(item_l4):
    assert math.isnan(score_item(item_l4, "anything"))


def test_chance_corrected_kappa_style():
    # Cohen's-kappa-style: (acc - chance) / (1 - chance).
    assert chance_corrected(0.75, 0.5) == pytest.approx(0.5)
    assert chance_corrected(0.5, 0.5) == 0.0
    assert chance_corrected(1.0, 0.5) == 1.0
    # Lower bound: when acc < chance, value goes negative.
    assert chance_corrected(0.25, 0.5) == pytest.approx(-0.5)
    # chance = 1 -> denom is 0; library returns 0.
    assert chance_corrected(0.5, 1.0) == 0.0
