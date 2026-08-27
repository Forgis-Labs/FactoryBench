"""Simple non-LLM baselines for FactoryBench QA pairs.

Two strategies, picked per-item from the answer format:

    - linear regression  : L1 template 7 (predict signal value at n steps ahead).
                           Fits y = a*x + b on the target signal across the context
                           rows, extrapolates to (N + steps_ahead - 1).
    - uniform random     : all MCQ (A/B/C/D), TFFT multi-select, A-B-C-D ranking
                           permutations, and the L1 t1 phase-window numeric.

Free-form templates (L4 t1, t2) are skipped, no traditional baseline maps cleanly
to "produce a remediation protocol".

Outputs reply JSONs in the same shape as model replies so they slot into
`rescore_results.py` with no special handling:

    output/replies/level{N}/<baseline_name>/level{N}_{NNNN}_0_answer.json

Usage:
    python -m src.evaluation.simple_baselines \
        --input output/questions \
        --output output/replies \
        --name baseline_simple
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_KV_RE = re.compile(r"([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?)")
_ROW_PREFIX_RE = re.compile(r"^t=-?\d+(?:\.\d+)?\s*:\s*")
_T_PREFIX_RE = re.compile(r"^t=(-?\d+(?:\.\d+)?)\s*:\s*")
_TIME_DELTA_RE = re.compile(r"T\+(\d+)\s*ms", re.IGNORECASE)


def _coerce_list_answer(ans: Any) -> Optional[List[float]]:
    """Return a python list if `ans` is a list (or a JSON string of one), else None."""
    if isinstance(ans, list):
        return ans
    if isinstance(ans, str) and ans.strip().startswith("[") and ans.strip().endswith("]"):
        try:
            parsed = json.loads(ans)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, TypeError):
            return None
    return None


def _looks_like_list_answer(ans: Any) -> bool:
    return _coerce_list_answer(ans) is not None


def _seeded_rng(item_id: str) -> random.Random:
    try:
        seed = uuid.UUID(item_id).int & 0xFFFFFFFF
    except (ValueError, AttributeError, TypeError):
        seed = abs(hash(item_id)) & 0xFFFFFFFF
    return random.Random(seed)


def _parse_row_values(row: str, signal: str) -> Optional[float]:
    body = _ROW_PREFIX_RE.sub("", row)
    for m in _KV_RE.finditer(body):
        if m.group(1) == signal:
            try:
                return float(m.group(2))
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Per-strategy answer producers
# ---------------------------------------------------------------------------


def _acronym_for(signal: str, context: Dict[str, Any]) -> str:
    """Resolve the long-form signal name to its row-level acronym, if a mapping exists."""
    mapping = (
        (context.get("time_series_format") or {}).get("acronym_mapping")
        or context.get("acronym_mapping")
        or {}
    )
    # mapping is acronym → long, build reverse lookup
    for acronym, long_name in mapping.items():
        if long_name == signal:
            return acronym
    return signal  # fallback: use as-is (already an acronym, or no mapping)


def _row_time_ms(row: str) -> Optional[float]:
    m = _T_PREFIX_RE.match(row)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _fit_predict_at(times: List[float], values: List[float], target_t: float) -> Optional[float]:
    """Linear regression on (t, v); predict at target_t. Returns None if insufficient data."""
    if len(values) < 2:
        return values[-1] if values else None
    a, b = np.polyfit(np.asarray(times, dtype=float), np.asarray(values, dtype=float), 1)
    return float(a * target_t + b)


def regression_answer(item: Dict[str, Any]) -> Any:
    """Linear regression on the target signal.

    Supports two shapes:
      - scalar (L1 t7): predict one value `steps_ahead` rows ahead.
      - vector (L2/L3 t5): predict 6 per-axis values at last_t + 'T+<delta>ms'.
        Per-axis linear fit on (t_ms, value), evaluated at the target time.
    """
    bounds = item.get("acceptance_bounds") or {}
    signal = bounds.get("signal")
    if not signal:
        return None
    context = item.get("context") or {}
    rows = context.get("time_series") or []
    gold = item.get("answer")

    # Vector answer (predictive list of joint values), handles both real lists
    # and JSON-string-encoded lists (which is how L2/L3 t5 actually store them).
    gold_list = _coerce_list_answer(gold)
    if gold_list is not None:
        m = _TIME_DELTA_RE.search(item.get("question", ""))
        if not m:
            return None
        delta_ms = float(m.group(1))
        # Build (t_ms, row_dict) for rows that carry a parsable timestamp
        timed_rows: List[Tuple[float, str]] = []
        for r in rows:
            t = _row_time_ms(r)
            if t is not None:
                timed_rows.append((t, r))
        if len(timed_rows) < 2:
            return None
        target_t = timed_rows[-1][0] + delta_ms
        preds: List[float] = []
        for axis in range(len(gold_list)):
            full_name = f"{signal}_{axis}"
            feat_key = _acronym_for(full_name, context)
            ts: List[float] = []
            vs: List[float] = []
            for t, r in timed_rows:
                v = _parse_row_values(r, feat_key)
                if v is not None:
                    ts.append(t)
                    vs.append(v)
            pred = _fit_predict_at(ts, vs, target_t)
            preds.append(round(pred, 6) if pred is not None else 0.0)
        return preds

    # Scalar answer (L1 t7): row-index regression, extrapolate `steps_ahead`
    steps_ahead = int(bounds.get("steps_ahead", 1))
    feat_key = _acronym_for(signal, context)
    ys: List[float] = []
    for r in rows:
        v = _parse_row_values(r, feat_key)
        if v is not None:
            ys.append(v)
    if len(ys) < 2:
        return round(ys[-1], 4) if ys else None
    xs = np.arange(len(ys), dtype=float)
    a, b = np.polyfit(xs, ys, 1)
    pred = a * (len(ys) - 1 + steps_ahead) + b
    return round(float(pred), 4)


def random_mcq_answer(item: Dict[str, Any], rng: random.Random) -> str:
    """Pick one option key at random."""
    keys = list((item.get("options") or {}).keys())
    return rng.choice(keys) if keys else ""


def random_tfft_answer(item: Dict[str, Any], rng: random.Random) -> str:
    """Multi-select T/F string of the same length as the gold answer."""
    n = len(str(item.get("answer") or ""))
    if n == 0:
        n = len(item.get("options") or {}) or 4
    return "".join(rng.choice("TF") for _ in range(n))


def random_ranking_answer(item: Dict[str, Any], rng: random.Random) -> str:
    """Random permutation of the option keys."""
    keys = list((item.get("options") or {}).keys())
    rng.shuffle(keys)
    return "".join(keys)


def random_numeric_answer(item: Dict[str, Any], rng: random.Random) -> Optional[float]:
    """Integer in a sensible range derived from provenance (episode_length)."""
    prov = item.get("provenance") or {}
    upper = prov.get("episode_length")
    if not isinstance(upper, int) or upper <= 0:
        # Fallback: guess from context length
        ctx = (item.get("context") or {}).get("time_series") or []
        upper = max(len(ctx), 1)
    return rng.randint(0, max(upper - 1, 0))


# ---------------------------------------------------------------------------
# Strategy classification + dispatch
# ---------------------------------------------------------------------------


def classify(item: Dict[str, Any]) -> str:
    """Return one of: 'regression', 'mcq', 'tfft', 'ranking', 'numeric', 'skip'."""
    level = item.get("level")
    tid = item.get("template_id")
    ans = item.get("answer")
    opts = item.get("options") or {}

    # L1 t7 (scalar) and L2/L3 t5 (vector of 6 joint values) → linear regression
    if level == 1 and tid == 7:
        return "regression"
    if level in (2, 3) and tid == 5 and _looks_like_list_answer(ans):
        return "regression"

    # MC-style: gold answer is a string with options provided
    if opts and isinstance(ans, str):
        keys = set(opts.keys())
        # Ranking: gold answer length == #options and is a permutation of keys
        if len(ans) == len(opts) and set(ans) == keys:
            return "ranking"
        # TFFT-style multi-select
        if len(ans) >= 2 and set(ans) <= {"T", "F"}:
            return "tfft"
        # Single-letter MCQ
        if len(ans) == 1 and ans in keys:
            return "mcq"
        # Catch-all: pick a key at random
        return "mcq"

    # Numeric (no options), phase-window question and friends
    if isinstance(ans, (int, float)):
        return "numeric"

    # Free-form text, out of scope for these baselines
    return "skip"


def baseline_answer(item: Dict[str, Any]) -> Tuple[str, Any]:
    """Return (strategy, answer) for an item. answer is None when 'skip'."""
    strategy = classify(item)
    if strategy == "skip":
        return strategy, None
    rng = _seeded_rng(str(item.get("id", "")))
    if strategy == "regression":
        return strategy, regression_answer(item)
    if strategy == "mcq":
        return strategy, random_mcq_answer(item, rng)
    if strategy == "tfft":
        return strategy, random_tfft_answer(item, rng)
    if strategy == "ranking":
        return strategy, random_ranking_answer(item, rng)
    if strategy == "numeric":
        return strategy, random_numeric_answer(item, rng)
    return strategy, None


# ---------------------------------------------------------------------------
# Reply file emission (matches the shape rescore_results.py expects)
# ---------------------------------------------------------------------------


def make_reply(item: Dict[str, Any], strategy: str, answer: Any, model_name: str) -> Dict[str, Any]:
    item_stem_id = item.get("id", "")
    custom_id = f"level{item.get('level')}_{Path(item.get('_source_filename', '')).stem.split('_')[-1]}_0"
    if answer is None:
        ans_str = ""
    elif isinstance(answer, list):
        ans_str = json.dumps(answer, separators=(",", ","))
    else:
        ans_str = str(answer)
    return {
        "custom_id": custom_id,
        "prompt_index": 0,
        "prompt_file": "",
        "prompt": "",
        "answer": ans_str,
        "ground_truth": item.get("answer"),
        "score": None,
        "llm_judge_score": None,
        "llm_judge_reason": None,
        "answer_format": "unknown",
        "question_type": "unknown",
        "model": model_name,
        "baseline_strategy": strategy,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
        "estimated_cost": 0.0,
        "raw_api_response": None,
    }


def process_level(input_dir: Path, output_dir: Path, model_name: str) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for src in sorted(input_dir.glob("*.json")):
        item = json.loads(src.read_text(encoding="utf-8"))
        item["_source_filename"] = src.name
        strategy, answer = baseline_answer(item)
        counts[strategy] = counts.get(strategy, 0) + 1
        if strategy == "skip":
            continue
        reply = make_reply(item, strategy, answer, model_name)
        out_name = f"{src.stem}_0_answer.json"
        (output_dir / out_name).write_text(json.dumps(reply, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=Path("output/questions"))
    parser.add_argument("--output", type=Path, default=Path("output/replies"))
    parser.add_argument("--name", default="baseline_simple", help="Subdir under output/replies/level{N}/")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    grand: Dict[str, int] = {}
    for level in args.levels:
        in_dir = args.input / f"level{level}"
        out_dir = args.output / f"level{level}" / args.name
        if not in_dir.exists():
            logger.warning(f"Skipping level {level}: {in_dir} does not exist")
            continue
        counts = process_level(in_dir, out_dir, args.name)
        emitted = sum(v for k, v in counts.items() if k != "skip")
        skipped = counts.get("skip", 0)
        logger.info(f"Level {level}: emitted {emitted}, skipped {skipped} → {out_dir}")
        for k, v in counts.items():
            grand[k] = grand.get(k, 0) + v
    logger.info(f"Done. Strategy totals across all levels: {grand}")


if __name__ == "__main__":
    main()
