"""Scoring functions for Chronos-2 predictions.

Mirrors the scoring logic from src/evaluation/run_foundry_eval.py
but without any torch dependency, so it can be tested locally.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def score_numerical(pred: float, gt: float, bounds: Optional[Dict[str, Any]]) -> float:
    """Three-level piecewise: 1.0 within margin, 0.5 within 2x, 0.0 outside."""
    if bounds and "margin" in bounds:
        margin = float(bounds["margin"])
        d = abs(pred - gt)
        if d <= margin:
            return 1.0
        elif d <= 2 * margin:
            return 0.5
        else:
            return 0.0
    elif bounds and "min" in bounds and "max" in bounds:
        return float(float(bounds["min"]) <= pred <= float(bounds["max"]))
    else:
        return float(abs(pred - gt) < 1e-4)


def score_tensor(pred: List[float], gt: List[float], bounds: Optional[Dict[str, Any]]) -> float:
    """Per-channel margin scoring, averaged."""
    if bounds and "margin" in bounds:
        margins = bounds["margin"]
        if len(gt) == len(pred) == len(margins):
            scores = []
            for p, g, m in zip(pred, gt, margins):
                d = abs(p - g)
                if d <= m:
                    scores.append(1.0)
                elif d <= 2 * m:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            return sum(scores) / len(scores) if scores else 0.0
    return float(pred == gt)


def score_task_result(
    answer_format: str,
    prediction: Any,
    ground_truth: Any,
    acceptance_bounds: Optional[Dict[str, Any]],
) -> float:
    """Score a prediction against ground truth."""
    if answer_format == "numerical":
        gt = float(ground_truth)
        pred = float(prediction)
        return score_numerical(pred, gt, acceptance_bounds)
    elif answer_format == "tensor":
        if isinstance(ground_truth, str):
            gt = json.loads(ground_truth)
        else:
            gt = list(ground_truth)
        pred = list(prediction)
        return score_tensor(pred, gt, acceptance_bounds)
    return 0.0
