"""Noise the time-series numbers in generated QA pairs.

Same questions, options, answers, provenance, only the numerical time-series
values change. For each float-valued feature, the first value is preserved and
subsequent values follow a random walk:

    x_{t+1} = x_t + N(0, sigma^2)

with a single global `sigma` shared across every feature. Integer-valued
features (task_phase, fault_label, joint_mode_*, etc.) are preserved as-is.

The same noising is applied to time-series chunks embedded in option strings
(L1 template 5 severity ranking, L2/L3 template 1 chunk ranking).

Per-item RNG is seeded from the item's UUID, so re-runs produce identical
outputs.

Usage:
    python -m src.question_generation.noise_qa_pairs \
        --input output/questions \
        --output output/questions_noised \
        --sigma 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Row formats:
#   Context rows:    "t=339: fp0=58.55, fp1=-101.28..."
#   Option chunks:   "fp0=58.55, fp1=-101.28..." (no t= prefix)
#   Multi-row chunks are joined by " | " (encode_chunk).
_ROW_PREFIX_RE = re.compile(r"^(t=-?\d+(?:\.\d+)?\s*:\s*)(.*)$")
_KV_RE = re.compile(r"\s*([^=,\s]+)\s*=\s*(-?[^,]+?)\s*(?:,|$)")
_CHUNK_SEP = " | "


def _split_prefix(row: str) -> Tuple[str, str]:
    m = _ROW_PREFIX_RE.match(row)
    if m:
        return m.group(1), m.group(2)
    return "", row


def _parse_kv(body: str) -> Dict[str, str]:
    return {km.group(1): km.group(2) for km in _KV_RE.finditer(body)}


def _serialize(prefix: str, feats: Dict[str, str]) -> str:
    return f"{prefix}{', '.join(f'{k}={v}' for k, v in feats.items())}"


def _decimal_places(s: str) -> int:
    if "." not in s:
        return 0
    return len(s.split(".", 1)[1])


def _is_integer_string(s: str) -> bool:
    s = s.strip()
    if "." in s:
        return False
    try:
        int(s)
        return True
    except ValueError:
        return False


def _noise_rows(rows: List[Tuple[str, Dict[str, str]]], sigma: float, rng: np.random.Generator) -> List[Tuple[str, Dict[str, str]]]:
    """Apply random-walk noising to all numeric float features in a row stream."""
    if not rows:
        return rows
    feats = list(rows[0][1].keys())
    out_feats: List[Dict[str, str]] = [dict(f) for _, f in rows]

    for feat in feats:
        values: List[float] = []
        precisions: List[int] = []
        skip = False
        for _, fmap in rows:
            s = fmap.get(feat)
            if s is None:
                skip = True
                break
            try:
                values.append(float(s))
                precisions.append(_decimal_places(s))
            except ValueError:
                skip = True
                break
        if skip:
            continue
        # Preserve any feature with non-finite values (nan/inf) as-is.
        if any(not np.isfinite(v) for v in values):
            continue
        # Preserve integer-formatted features as-is (phase, fault label, modes...)
        if all(_is_integer_string(rows[i][1][feat]) for i in range(len(rows))):
            continue
        # Random walk from the original first value
        new_vals = [values[0]]
        for _ in range(1, len(values)):
            new_vals.append(new_vals[-1] + rng.normal(0.0, sigma))
        precision = max(precisions) if precisions else 0
        for i, v in enumerate(new_vals):
            if precision > 0:
                out_feats[i][feat] = f"{v:.{precision}f}"
            else:
                out_feats[i][feat] = str(int(round(v)))
    return [(prefix, out_feats[i]) for i, (prefix, _) in enumerate(rows)]


def _noise_string_rows(row_strings: List[str], sigma: float, rng: np.random.Generator) -> List[str]:
    parsed = [(_split_prefix(r)[0], _parse_kv(_split_prefix(r)[1])) for r in row_strings]
    if not parsed or not parsed[0][1]:
        return row_strings
    noised = _noise_rows(parsed, sigma, rng)
    return [_serialize(p, f) for p, f in noised]


def _noise_chunk_option(value: str, sigma: float, rng: np.random.Generator) -> str:
    """Noise a serialized chunk option string of the form 'kv | kv | ...'."""
    rows = value.split(_CHUNK_SEP)
    parsed = [(_split_prefix(r)[0], _parse_kv(_split_prefix(r)[1])) for r in rows]
    if not parsed[0][1]:
        return value  # not actually a chunk
    noised = _noise_rows(parsed, sigma, rng)
    return _CHUNK_SEP.join(_serialize(p, f) for p, f in noised)


def _looks_like_encoded_chunk(value: str) -> bool:
    if not isinstance(value, str) or "=" not in value:
        return False
    head = value.split(_CHUNK_SEP, 1)[0]
    parsed = _parse_kv(_split_prefix(head)[1])
    # Encoded chunks have many features per row; multi-choice text statements don't
    return len(parsed) >= 3


def _noise_context(ctx: Any, sigma: float, rng: np.random.Generator) -> Any:
    if not isinstance(ctx, dict):
        return ctx
    if "time_series" in ctx and isinstance(ctx["time_series"], list):
        ctx["time_series"] = _noise_string_rows(ctx["time_series"], sigma, rng)
    for key in ("series_a", "series_b"):
        sub = ctx.get(key)
        if isinstance(sub, dict) and isinstance(sub.get("time_series"), list):
            sub["time_series"] = _noise_string_rows(sub["time_series"], sigma, rng)
    return ctx


def _noise_options(opts: Any, sigma: float, rng: np.random.Generator) -> Any:
    if not isinstance(opts, dict):
        return opts
    out = {}
    for k, v in opts.items():
        if _looks_like_encoded_chunk(v):
            out[k] = _noise_chunk_option(v, sigma, rng)
        else:
            out[k] = v
    return out


def noise_item(item: Dict[str, Any], sigma: float) -> Dict[str, Any]:
    """In-place noising. Returns the same dict for chaining."""
    item_id = str(item.get("id", ""))
    try:
        seed = uuid.UUID(item_id).int & 0xFFFFFFFF
    except (ValueError, AttributeError):
        seed = abs(hash(item_id)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    item["context"] = _noise_context(item.get("context", {}), sigma, rng)
    item["options"] = _noise_options(item.get("options", {}), sigma, rng)
    item.setdefault("noise", {})
    item["noise"] = {"applied": True, "sigma": sigma, "rng_seed": seed}
    return item


def process_level(input_dir: Path, output_dir: Path, sigma: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.json"))
    for src in files:
        item = json.loads(src.read_text(encoding="utf-8"))
        noised = noise_item(item, sigma)
        (output_dir / src.name).write_text(
            json.dumps(noised, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/questions"),
        help="Root containing level{1,2,3,4} subdirs (default: output/questions)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/questions_noised"),
        help="Output root, mirrored to level{1,2,3,4} subdirs (default: output/questions_noised)")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Single global gaussian noise stddev added per timestep (default: 0.5)")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Levels to process (default: 1 2 3 4)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    total = 0
    for level in args.levels:
        in_dir = args.input / f"level{level}"
        out_dir = args.output / f"level{level}"
        if not in_dir.exists():
            logger.warning(f"Skipping level {level}: {in_dir} does not exist")
            continue
        n = process_level(in_dir, out_dir, args.sigma)
        logger.info(f"Level {level}: noised {n} items → {out_dir}")
        total += n
    logger.info(f"Done. {total} items written. sigma={args.sigma}")


if __name__ == "__main__":
    main()
