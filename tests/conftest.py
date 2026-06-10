"""Shared pytest fixtures.

All fixtures build :class:`Item` instances by hand to keep the suite offline --
no calls to ``load_split`` (which would hit Hugging Face).
"""
from __future__ import annotations

import pytest

from factorybench.types import AnswerFormat, Item


def _make_item(
    *,
    level: int,
    template_id: int,
    template_type: str,
    answer,
    options=None,
    acceptance_bounds=None,
    answer_format: AnswerFormat,
    id_suffix: str = "abc",
    root_cause: str | None = None,
    dataset: str = "aursad",
    fault_id: int = 0,
) -> Item:
    return Item(
        id=f"L{level}_{template_id}_{id_suffix}",
        level=level,
        template_id=template_id,
        template_type=template_type,
        question=f"Test {template_type} prompt for L{level}.{template_id}. Answer:",
        options=options or {},
        answer=answer,
        acceptance_bounds=acceptance_bounds,
        provenance={"dataset": dataset, "relevance": {"fault_id": fault_id}},
        context={
            "time_series_format": {
                "description": "Each row is a timestep.",
                "acronym_mapping": {"ett0": "effort_target_torque_0"},
            },
            "time_series": ["t=0: ett0=0.0", "t=99: ett0=0.1"],
        },
        hides=[],
        root_cause=root_cause,
        answer_format=answer_format,
    )


@pytest.fixture
def item_scalar_range() -> Item:
    return _make_item(
        level=1, template_id=1, template_type="predictive",
        answer=0, acceptance_bounds={"min": 0, "max": 297},
        answer_format=AnswerFormat.SCALAR_RANGE,
    )


@pytest.fixture
def item_scalar_margin() -> Item:
    return _make_item(
        level=2, template_id=4, template_type="predictive",
        answer=-90.5, acceptance_bounds={"signal": "feedback_pos_4", "std": 0.003, "margin": 0.005},
        answer_format=AnswerFormat.SCALAR_MARGIN,
    )


@pytest.fixture
def item_scalar_exact() -> Item:
    return _make_item(
        level=1, template_id=7, template_type="predictive",
        answer=0.7276,
        acceptance_bounds={"signal": "est_contact_force_5", "steps_ahead": 5, "actual_value": 0.7276},
        answer_format=AnswerFormat.SCALAR_EXACT,
    )


@pytest.fixture
def item_tensor_margin() -> Item:
    return _make_item(
        level=2, template_id=5, template_type="predictive",
        answer="[1.0, 2.0, 3.0]",
        acceptance_bounds={"signal": "x", "std": [0.1, 0.1, 0.1], "margin": [0.05, 0.05, 0.05]},
        answer_format=AnswerFormat.TENSOR_MARGIN,
    )


@pytest.fixture
def item_mcq() -> Item:
    return _make_item(
        level=1, template_id=6, template_type="identification",
        answer="C", options={"A": "alpha", "B": "beta", "C": "gamma"},
        answer_format=AnswerFormat.SINGLE_LETTER_MCQ,
    )


@pytest.fixture
def item_tf() -> Item:
    return _make_item(
        level=1, template_id=3, template_type="comparative",
        answer="TFTF", options={"A": "1", "B": "2", "C": "3", "D": "4"},
        answer_format=AnswerFormat.FOUR_LETTER_TF,
    )


@pytest.fixture
def item_ranking() -> Item:
    return _make_item(
        level=2, template_id=1, template_type="predictive",
        answer="BDCA", options={"A": "1", "B": "2", "C": "3", "D": "4"},
        answer_format=AnswerFormat.FOUR_LETTER_RANKING,
    )


@pytest.fixture
def item_l4() -> Item:
    return _make_item(
        level=4, template_id=1, template_type="troubleshooting",
        answer="Halt the cycle and check program logic.",
        acceptance_bounds=None,
        answer_format=AnswerFormat.FREE_FORM,
        root_cause="loosening_phase",
    )


@pytest.fixture
def all_items(
    item_scalar_range, item_scalar_margin, item_scalar_exact, item_tensor_margin,
    item_mcq, item_tf, item_ranking, item_l4,
):
    """One item per supported answer format (handy for round-trip tests)."""
    return [
        item_scalar_range, item_scalar_margin, item_scalar_exact, item_tensor_margin,
        item_mcq, item_tf, item_ranking, item_l4,
    ]


@pytest.fixture
def tmp_cache_dir(tmp_path):
    cache = tmp_path / "judge_cache"
    cache.mkdir()
    return cache
