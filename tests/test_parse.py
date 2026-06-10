import pytest

from factorybench.parse import ParseError, parse_output


def test_single_letter_exact(item_mcq):
    assert parse_output("C", item_mcq) == "C"
    assert parse_output("c", item_mcq) == "C"


def test_single_letter_within_prose(item_mcq):
    # Real-world preambles: "Sure! The answer is C."
    assert parse_output("Sure! The answer is C.", item_mcq) == "C"


def test_single_letter_rejects_outside_options(item_mcq):
    # Option set is {A, B, C}; 'D' is not valid.
    with pytest.raises(ParseError):
        parse_output("D", item_mcq)


def test_tf_string_exact(item_tf):
    assert parse_output("TFTF", item_tf) == "TFTF"


def test_tf_string_in_prose(item_tf):
    assert parse_output("My answer is TFFT.", item_tf) == "TFFT"


def test_tf_no_match_raises(item_tf):
    with pytest.raises(ParseError):
        parse_output("yes", item_tf)


def test_ranking_must_be_permutation(item_ranking):
    assert parse_output("BDCA", item_ranking) == "BDCA"
    with pytest.raises(ParseError):
        parse_output("AABB", item_ranking)  # not a permutation


def test_ranking_in_prose(item_ranking):
    assert parse_output("Order: BDCA", item_ranking) == "BDCA"


def test_scalar_pure_number(item_scalar_range):
    assert parse_output("42", item_scalar_range) == 42.0
    assert parse_output("3.14", item_scalar_range) == 3.14


def test_scalar_in_prose(item_scalar_range):
    assert parse_output("The answer is 42.", item_scalar_range) == 42.0


def test_scalar_negative(item_scalar_margin):
    assert parse_output("-90.5", item_scalar_margin) == -90.5


def test_scalar_unparseable(item_scalar_range):
    with pytest.raises(ParseError):
        parse_output("not a number", item_scalar_range)


def test_tensor_json(item_tensor_margin):
    assert parse_output("[1.0, 2.0, 3.0]", item_tensor_margin) == [1.0, 2.0, 3.0]


def test_tensor_loose_numbers(item_tensor_margin):
    # Falls back to number extraction when JSON parse fails.
    assert parse_output("around 1.0, 2.0, 3.0 maybe", item_tensor_margin) == [1.0, 2.0, 3.0]


def test_free_form_passthrough(item_l4):
    out = parse_output("Halt the cycle.", item_l4)
    assert out == "Halt the cycle."


def test_none_raises(item_scalar_range):
    with pytest.raises(ParseError):
        parse_output(None, item_scalar_range)
