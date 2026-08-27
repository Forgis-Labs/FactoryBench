"""Run TimesFM-2.5 (200M, PyTorch) zero-shot forecasting on FactoryBench Q&A.

Mirrors run_chronos2.py end-to-end so the two TSFM baselines are
evaluated under identical parsing, scoring, and chance-correction.

Usage:
    python experiments/v1/run_timesfm.py \
        --input experiments/v1/data/level_1_test.jsonl experiments/v1/data/level_2_test.jsonl \
        --max-per-template 999999 \
        --output experiments/v1/results_timesfm.json \
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
from typing import Any, Dict, List

import numpy as np
import torch

from parse_qa import ChronosTask, load_applicable_items, APPLICABLE_TEMPLATES
from scoring import score_task_result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TimesFM-2.5 forecaster
# ---------------------------------------------------------------------------


class TimesFM25Forecaster:
    """Wraps TimesFM-2.5 (200M, PyTorch) for zero-shot univariate forecasting.

    Bypasses PyTorchModelHubMixin.from_pretrained because the version of
    huggingface_hub installed here forwards 'proxies' into the constructor,
    which the TimesFM wrapper does not accept. We download the safetensors
    manually and call load_checkpoint directly.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        max_context: int = 512,
        max_horizon: int = 64,
        torch_compile: bool = False,
    ):
        import timesfm  # imported lazily so the script can be linted without it
        from huggingface_hub import hf_hub_download

        logger.info(f"Loading TimesFM-2.5 from {model_id} on {device}...")
        ckpt = hf_hub_download(repo_id=model_id, filename="model.safetensors")

        self.pipeline = timesfm.TimesFM_2p5_200M_torch(torch_compile=torch_compile)
        self.pipeline.model.load_checkpoint(ckpt, torch_compile=torch_compile)

        # Place model on the requested device. The torch wrapper exposes the
        # underlying nn.Module via .model — move it.
        try:
            self.pipeline.model.to(device)
        except Exception:  # pragma: no cover - rare path
            logger.warning("Could not move TimesFM model to %s; staying on default", device)

        self.pipeline.compile(timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
        ))
        self.device = device
        self.model_id = model_id
        logger.info("TimesFM-2.5 model loaded.")

    def forecast(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        """Forecast future values for one univariate series.

        Returns the point forecast at every step; caller takes the last one
        (matching how Chronos uses its median quantile at the last step).
        """
        ctx = np.asarray(context, dtype=np.float32).reshape(-1)
        # TimesFM 2.5 returns (point_forecast, quantile_forecast).
        # point_forecast: shape (1, horizon)
        point, _quantiles = self.pipeline.forecast(
            horizon=prediction_length,
            inputs=[ctx],
        )
        return np.asarray(point[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Scoring (shared with the Chronos pipeline)
# ---------------------------------------------------------------------------


def _ensure_margin(task: ChronosTask) -> dict:
    """Ensure acceptance_bounds has a margin for scoring.

    L1.7 items lack a margin field. We compute one from the context signal's
    range using R/12 (calibrates to chance level 1/4 — same calibration as
    the tensor scorer in the paper). This is identical to the Chronos
    pipeline so the two baselines are directly comparable.
    """
    bounds = dict(task.acceptance_bounds or {})
    if "margin" in bounds:
        return bounds
    if task.target_channels and task.answer_format == "numerical":
        ch = task.target_channels[0]
        arr = task.channels.get(ch)
        if arr is not None and len(arr) > 1:
            R = float(np.ptp(arr))
            if R > 0:
                bounds["margin"] = R / 12.0
            else:
                bounds["margin"] = max(abs(float(np.mean(arr))) * 0.01, 1e-3)
    return bounds


def score_task(task: ChronosTask, prediction: Any) -> float:
    bounds = _ensure_margin(task)
    return score_task_result(task.answer_format, prediction, task.ground_truth, bounds)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_inference(
    tasks: List[ChronosTask],
    forecaster: TimesFM25Forecaster,
) -> List[Dict[str, Any]]:
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
                ch_name = task.target_channels[0]
                context = task.channels[ch_name]
                forecast = forecaster.forecast(context, task.horizon_steps)
                prediction = round(float(forecast[-1]), 4)

            elif task.answer_format == "tensor":
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


def main() -> None:
    parser = argparse.ArgumentParser(description="TimesFM-2.5 baseline for FactoryBench")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--max-per-template", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("experiments/v1/results_timesfm.json"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-id", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--max-horizon", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

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

    for key in sorted(APPLICABLE_TEMPLATES):
        n = sum(1 for t in all_tasks if (t.level, t.template_id) == key)
        logger.info(f"  L{key[0]}.{key[1]}: {n} items")

    forecaster = TimesFM25Forecaster(
        device=args.device,
        model_id=args.model_id,
        max_context=args.max_context,
        max_horizon=args.max_horizon,
    )
    results = run_inference(all_tasks, forecaster)

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"aggregate": agg, "results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to {args.output}")
    logger.info(f"Aggregate: {json.dumps(agg, indent=2)}")


if __name__ == "__main__":
    main()
