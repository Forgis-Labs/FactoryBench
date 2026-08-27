"""Run Chronos-2 zero-shot forecasting on FactoryBench Q&A pairs.

Usage:
    # Download test data first, then:
    python experiments/v1/run_chronos2.py \
        --input experiments/v1/data/level_1_test.jsonl experiments/v1/data/level_2_test.jsonl \
        --max-per-template 999999 \
        --output experiments/v1/results_full.json \
        --device cuda
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from parse_qa import ChronosTask, load_applicable_items, APPLICABLE_TEMPLATES
from scoring import score_task_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chronos-2 forecaster
# ---------------------------------------------------------------------------


class Chronos2Forecaster:
    """Wraps Chronos-Bolt for zero-shot univariate forecasting."""

    def __init__(self, device: str = "cpu", model_id: str = "amazon/chronos-bolt-base"):
        from chronos import BaseChronosPipeline

        logger.info(f"Loading Chronos model {model_id} on {device}...")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float32,
        )
        self.device = device
        logger.info("Chronos model loaded.")

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int,
    ) -> np.ndarray:
        """Forecast future values for a single univariate series.

        Args:
            context: (T,) array of past values
            prediction_length: number of future steps to predict

        Returns:
            (prediction_length,) array of median forecasts
        """
        ctx = torch.tensor(context, dtype=torch.float32).unsqueeze(0)  # (1, T)
        # Chronos-Bolt returns (1, num_quantiles, prediction_length)
        # Quantiles are 9 evenly-spaced; index 4 is the median.
        forecast_tensor = self.pipeline.predict(ctx, prediction_length=prediction_length)
        median = forecast_tensor[:, 4, :]  # (1, prediction_length) -- median quantile
        return median.squeeze(0).cpu().numpy()


# Scoring is in scoring.py (no torch dependency, testable locally)


def _ensure_margin(task: ChronosTask) -> dict:
    """Ensure acceptance_bounds has a margin for scoring.

    L1.7 items lack a margin field. We compute one from the context signal's
    range, using R/12 (same calibration as the tensor scorer in the paper:
    E[uniform random] = 3m/R = 1/4 = MCQ chance).
    """
    bounds = dict(task.acceptance_bounds or {})
    if "margin" in bounds:
        return bounds
    # Compute from context channel range
    if task.target_channels and task.answer_format == "numerical":
        ch = task.target_channels[0]
        arr = task.channels.get(ch)
        if arr is not None and len(arr) > 1:
            R = float(np.ptp(arr))  # range = max - min
            if R > 0:
                bounds["margin"] = R / 12.0
            else:
                # Constant signal -- use small absolute margin
                bounds["margin"] = max(abs(float(np.mean(arr))) * 0.01, 1e-3)
    return bounds


def score_task(task: ChronosTask, prediction: Any) -> float:
    """Score a Chronos prediction against ground truth."""
    bounds = _ensure_margin(task)
    return score_task_result(
        task.answer_format, prediction, task.ground_truth, bounds
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_inference(
    tasks: List[ChronosTask],
    forecaster: Chronos2Forecaster,
) -> List[Dict[str, Any]]:
    """Run Chronos-2 on all tasks and return results."""
    results = []

    for i, task in enumerate(tasks):
        t0 = time.time()
        logger.info(
            f"[{i + 1}/{len(tasks)}] L{task.level}.{task.template_id} "
            f"({task.answer_format}) horizon={task.horizon_steps} "
            f"target={task.target_channels}"
        )

        try:
            if task.answer_format == "numerical":
                # Single channel forecast
                ch_name = task.target_channels[0]
                context = task.channels[ch_name]
                forecast = forecaster.forecast(context, task.horizon_steps)
                prediction = round(float(forecast[-1]), 4)  # last step = target

            elif task.answer_format == "tensor":
                # 6-channel independent forecast
                prediction = []
                for ch_name in task.target_channels:
                    context = task.channels[ch_name]
                    forecast = forecaster.forecast(context, task.horizon_steps)
                    prediction.append(round(float(forecast[-1]), 6))

            else:
                prediction = None

            score = score_task(task, prediction)
            dt = time.time() - t0

            result = {
                "qa_id": task.qa_id,
                "level": task.level,
                "template_id": task.template_id,
                "answer_format": task.answer_format,
                "target_channels": task.target_channels,
                "horizon_steps": task.horizon_steps,
                "context_length": len(task.timestamps_ms),
                "prediction": prediction,
                "ground_truth": task.ground_truth,
                "acceptance_bounds": task.acceptance_bounds,
                "score": score,
                "inference_time_s": round(dt, 3),
                "question": task.question,
            }
            results.append(result)

            logger.info(
                f"  pred={prediction}, gt={task.ground_truth}, "
                f"score={score:.2f}, time={dt:.2f}s"
            )

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results.append({
                "qa_id": task.qa_id,
                "level": task.level,
                "template_id": task.template_id,
                "error": str(e),
                "score": 0.0,
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 baseline for FactoryBench")
    parser.add_argument(
        "--input", nargs="+", type=Path, required=True,
        help="Path(s) to test JSONL files (one per level)",
    )
    parser.add_argument("--max-per-template", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("experiments/v1/results_full.json"))
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--model-id", type=str, default="amazon/chronos-bolt-base",
        help="HuggingFace model ID for Chronos",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load tasks
    all_tasks: List[ChronosTask] = []
    for p in args.input:
        if not p.exists():
            logger.warning(f"File not found: {p}")
            continue
        tasks = load_applicable_items(p, max_per_template=args.max_per_template)
        logger.info(f"Loaded {len(tasks)} applicable tasks from {p.name}")
        all_tasks.extend(tasks)

    if not all_tasks:
        logger.error("No applicable tasks found. Exiting.")
        return

    # Print summary
    for key in sorted(APPLICABLE_TEMPLATES):
        n = sum(1 for t in all_tasks if (t.level, t.template_id) == key)
        logger.info(f"  L{key[0]}.{key[1]}: {n} items")

    # Load model and run
    forecaster = Chronos2Forecaster(device=args.device, model_id=args.model_id)
    results = run_inference(all_tasks, forecaster)

    # Aggregate
    agg: Dict[str, Any] = {"model": args.model_id, "device": args.device}
    for key in sorted(APPLICABLE_TEMPLATES):
        level, tid = key
        template_results = [r for r in results if r.get("level") == level and r.get("template_id") == tid]
        scores = [r["score"] for r in template_results if "score" in r]
        tag = f"L{level}.{tid}"
        agg[tag] = {
            "n": len(template_results),
            "mean_score": round(np.mean(scores), 4) if scores else None,
            "std_score": round(np.std(scores), 4) if scores else None,
            "scores": scores,
        }
    all_scores = [r["score"] for r in results if "score" in r]
    agg["overall"] = {
        "n": len(results),
        "mean_score": round(np.mean(all_scores), 4) if all_scores else None,
    }
    agg["total_inference_time_s"] = round(
        sum(r.get("inference_time_s", 0) for r in results), 2
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"aggregate": agg, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to {args.output}")
    logger.info(f"Aggregate: {json.dumps(agg, indent=2)}")


if __name__ == "__main__":
    main()
