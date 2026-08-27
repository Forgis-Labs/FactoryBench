"""Unit test for Q&A parsers -- validates locally without HF data.

Creates synthetic Q&A items matching the exact structure produced by the
FactoryBench generation pipeline, then checks the parser extracts the
correct signals, horizons, and can round-trip the scoring.

Run: python experiments/v1/test_parsers.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from parse_qa import (
    ChronosTask,
    load_applicable_items,
    parse_encoded_rows,
    parse_qa_item,
)


def _make_encoded_rows(n_steps: int = 40, dt_ms: float = 8.0) -> tuple:
    """Create synthetic encoded time-series rows + acronym mapping."""
    # Simulate 6 feedback_pos channels + 6 effort_target_torque channels
    channels = {
        f"feedback_pos_{i}": f"fbp{i}" for i in range(6)
    }
    channels.update({
        f"effort_target_torque_{i}": f"ett{i}" for i in range(6)
    })
    # Reverse: acronym -> full name
    acronym_mapping = {v: k for k, v in channels.items()}

    # Generate data
    np.random.seed(42)
    rows = []
    for step in range(n_steps):
        t_ms = step * dt_ms
        parts = [f"t={t_ms:.1f}"]
        for full_name, acro in sorted(channels.items()):
            val = np.sin(step * 0.1 + hash(full_name) % 10) * 2 + 5
            parts.append(f"{acro}={val:.2f}")
        rows.append(": ".join([parts[0]]) + ": " + ", ".join(parts[1:]))

    # Fix the encoding: first part is "t=X", rest are "acro=val, ..."
    rows_fixed = []
    for step in range(n_steps):
        t_ms = step * dt_ms
        kv_parts = []
        for full_name, acro in sorted(channels.items()):
            val = np.sin(step * 0.1 + hash(full_name) % 10) * 2 + 5
            kv_parts.append(f"{acro}={val:.2f}")
        rows_fixed.append(f"t={t_ms:.1f}: " + ", ".join(kv_parts))

    return rows_fixed, acronym_mapping


def make_l1_7_item() -> dict:
    """Synthetic L1.7 item: predict signal value N steps ahead."""
    rows, mapping = _make_encoded_rows(40, dt_ms=8.0)
    return {
        "id": "test-l1-7-001",
        "level": 1,
        "template_id": 7,
        "template_type": "predictive",
        "question": "Given the sensor stream below, what is the expected value of joint positions (axis 2) at T+5ms? Answer only with an integer or decimal number, nothing else.",
        "answer": 5.1234,
        "acceptance_bounds": {
            "signal": "feedback_pos_2",
            "steps_ahead": 5,
        },
        "context": {
            "time_series_format": {
                "acronym_mapping": mapping,
            },
            "time_series": rows,
        },
        "options": {},
    }


def make_l2_4_item() -> dict:
    """Synthetic L2.4 item: predict signal value at T+Nms under anomaly."""
    rows, mapping = _make_encoded_rows(40, dt_ms=8.0)
    return {
        "id": "test-l2-4-001",
        "level": 2,
        "template_id": 4,
        "template_type": "predictive",
        "question": "The sensor stream below is from a robot exhibiting unexpected_payload. What is the expected value of joint positions (axis 2) at T+40ms? Answer only with an integer or decimal number, nothing else.",
        "answer": 5.5678,
        "acceptance_bounds": {
            "signal": "feedback_pos_2",
            "std": 0.15,
            "margin": 0.1125,
        },
        "context": {
            "time_series_format": {
                "acronym_mapping": mapping,
            },
            "time_series": rows,
        },
        "options": {},
    }


def make_l2_5_item() -> dict:
    """Synthetic L2.5 item: predict 6-joint tensor at T+Nms under anomaly."""
    rows, mapping = _make_encoded_rows(40, dt_ms=8.0)
    return {
        "id": "test-l2-5-001",
        "level": 2,
        "template_id": 5,
        "template_type": "predictive",
        "question": "The sensor stream below is from a robot exhibiting unexpected_payload. What are the expected values of joint positions at T+40ms? Answer only with a list of 6 numbers (integer or decimal) formatted as a JSON array, ie. [10.2,9,2.1,1,7,0.21]. Do not return anything else.",
        "answer": "[5.1,5.2,5.3,5.4,5.5,5.6]",
        "acceptance_bounds": {
            "signal": "feedback_pos",
            "std": [0.15, 0.12, 0.18, 0.10, 0.14, 0.16],
            "margin": [0.11, 0.09, 0.14, 0.08, 0.11, 0.12],
        },
        "context": {
            "time_series_format": {
                "acronym_mapping": mapping,
            },
            "time_series": rows,
        },
        "options": {},
    }


def test_parse_encoded_rows():
    """Test that encoded rows can be parsed back to numeric arrays."""
    rows, mapping = _make_encoded_rows(10, dt_ms=8.0)
    timestamps, channel_names, data = parse_encoded_rows(rows, mapping)

    assert len(timestamps) == 10, f"Expected 10 timestamps, got {len(timestamps)}"
    assert timestamps[0] == 0.0
    assert abs(timestamps[1] - 8.0) < 0.01
    assert len(channel_names) == 12  # 6 fbp + 6 ett
    assert data.shape == (10, 12)
    assert "feedback_pos_0" in channel_names
    assert "effort_target_torque_5" in channel_names
    print("  parse_encoded_rows: PASSED")


def test_parse_l1_7():
    """Test L1.7 parser extracts signal and steps_ahead."""
    item = make_l1_7_item()
    task = parse_qa_item(item)

    assert task is not None, "L1.7 parse returned None"
    assert task.level == 1
    assert task.template_id == 7
    assert task.answer_format == "numerical"
    assert task.target_channels == ["feedback_pos_2"]
    assert task.horizon_steps == 5
    assert task.ground_truth == 5.1234
    assert len(task.timestamps_ms) == 40
    assert len(task.channels) == 12
    assert "feedback_pos_2" in task.channels
    print(f"  L1.7: PASSED (target={task.target_channels}, horizon={task.horizon_steps} steps)")


def test_parse_l2_4():
    """Test L2.4 parser extracts signal and converts ms to steps."""
    item = make_l2_4_item()
    task = parse_qa_item(item)

    assert task is not None, "L2.4 parse returned None"
    assert task.level == 2
    assert task.template_id == 4
    assert task.answer_format == "numerical"
    assert task.target_channels == ["feedback_pos_2"]
    # 40ms / 8ms per step = 5 steps
    assert task.horizon_steps == 5, f"Expected 5 steps, got {task.horizon_steps}"
    assert task.ground_truth == 5.5678
    print(f"  L2.4: PASSED (target={task.target_channels}, horizon={task.horizon_steps} steps, from 40ms)")


def test_parse_l2_5():
    """Test L2.5 parser extracts 6-joint channels and converts ms to steps."""
    item = make_l2_5_item()
    task = parse_qa_item(item)

    assert task is not None, "L2.5 parse returned None"
    assert task.level == 2
    assert task.template_id == 5
    assert task.answer_format == "tensor"
    assert len(task.target_channels) == 6
    assert task.target_channels[0] == "feedback_pos_0"
    assert task.target_channels[5] == "feedback_pos_5"
    assert task.horizon_steps == 5
    print(f"  L2.5: PASSED (target={task.target_channels}, horizon={task.horizon_steps} steps, from 40ms)")


def test_load_from_jsonl():
    """Test loading from a JSONL file with mixed templates."""
    items = [
        make_l1_7_item(),
        make_l2_4_item(),
        make_l2_5_item(),
        # Add a non-applicable item (L1.2 MCQ) that should be skipped
        {
            "id": "skip-me",
            "level": 1,
            "template_id": 2,
            "question": "What anomaly?",
            "answer": "A",
            "context": {},
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
        tmp_path = Path(f.name)

    try:
        tasks = load_applicable_items(tmp_path, max_per_template=10)
        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
        levels = {(t.level, t.template_id) for t in tasks}
        assert levels == {(1, 7), (2, 4), (2, 5)}
        print(f"  load_from_jsonl: PASSED (loaded {len(tasks)}, skipped 1 non-applicable)")
    finally:
        tmp_path.unlink()


def test_scoring():
    """Test that scoring logic matches expected behavior."""
    from scoring import score_numerical, score_tensor

    # Numerical: within margin -> 1.0
    assert score_numerical(5.0, 5.05, {"margin": 0.1}) == 1.0
    # Numerical: within 2x margin -> 0.5
    assert score_numerical(5.0, 5.15, {"margin": 0.1}) == 0.5
    # Numerical: outside -> 0.0
    assert score_numerical(5.0, 5.25, {"margin": 0.1}) == 0.0

    # Tensor: mixed scores
    pred = [5.0, 5.0, 5.0]
    gt = [5.05, 5.15, 5.25]
    margins = [0.1, 0.1, 0.1]
    score = score_tensor(pred, gt, {"margin": margins})
    assert abs(score - 0.5) < 0.01, f"Expected ~0.5, got {score}"  # (1.0 + 0.5 + 0.0) / 3
    print(f"  scoring: PASSED (numerical + tensor)")


if __name__ == "__main__":
    print("Running parser tests...\n")

    test_parse_encoded_rows()
    test_parse_l1_7()
    test_parse_l2_4()
    test_parse_l2_5()
    test_load_from_jsonl()
    test_scoring()

    print("\n All tests passed!")
