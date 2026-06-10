"""count_tokens works with or without tiktoken installed."""
from factorybench.tokens import count_tokens, has_tiktoken, is_precise


def test_empty_and_none_return_zero():
    assert count_tokens("") == 0
    assert count_tokens(None) == 0


def test_heuristic_minimum_one():
    # One-char input always counts as at least 1 token.
    assert count_tokens("a") >= 1


def test_count_tokens_positive_for_nontrivial_input():
    # Without tiktoken: len/4 heuristic ~= 10. With tiktoken: BPE compresses
    # repeated chars, can be much smaller. Both should be >= 1.
    s = "x" * 40
    assert count_tokens(s) >= 1


def test_precise_encoding_when_available():
    if not has_tiktoken():
        # No tiktoken installed; heuristic is exact 4-chars-per-token.
        assert count_tokens("hello world world world") == 23 // 4
        return
    # With tiktoken (cl100k_base), "hello world" is exactly 2 tokens.
    assert count_tokens("hello world", model="gpt-4o") == 2
    # And cl100k_base falls back gracefully for unknown models.
    assert count_tokens("hello world", model="unknown-x") == 2


def test_count_with_model_routes_to_encoder_when_available():
    # With or without tiktoken, this should return a positive integer.
    n = count_tokens("hello world", model="gpt-4o")
    assert n > 0


def test_is_precise_consistent_with_has_tiktoken():
    # If tiktoken is importable, the universal encoder loads, so is_precise(None)
    # should be True. If not, it's False.
    assert is_precise(None) == has_tiktoken()


def test_unknown_model_still_counts():
    # Unknown models fall back to cl100k_base; non-tiktoken installs fall back to heuristic.
    assert count_tokens("foo bar", model="totally-unknown-x9z") > 0
