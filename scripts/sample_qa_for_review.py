"""Stratified sample of QA pairs for human quality review.

Walks ``--questions-root/level{N}/*.json``, groups by ``(level, template_id)``
and takes ``--per-stratum`` items per group with a deterministic seed. Output
JSONL has one row per sampled item, ready for the Streamlit review app
(``scripts.qa_review_app``).

Usage::

    python -m scripts.sample_qa_for_review \\
        --questions-root output/test_eval/questions \\
        --per-stratum 5 \\
        --output output/qa_review_sample.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load(p: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--questions-root",
        type=Path,
        default=Path("output/test_eval/questions"),
        help="Root with level<N>/*.json question files.",
    )
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument(
        "--per-stratum", type=int, default=5,
        help="Items per (level, template_id) group.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output", type=Path, default=Path("output/qa_review_sample.jsonl"),
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_stratum: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for lvl in args.levels:
        d = args.questions_root / f"level{lvl}"
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            q = _load(p)
            if not q:
                continue
            tid = q.get("template_id")
            if tid is None:
                continue
            by_stratum[(int(q.get("level", lvl)), int(tid))].append(p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_strata = 0
    n_items = 0
    with args.output.open("w", encoding="utf-8") as out:
        for key in sorted(by_stratum):
            paths = by_stratum[key]
            rng.shuffle(paths)
            picked = paths[: args.per_stratum]
            n_strata += 1
            for p in picked:
                q = _load(p) or {}
                # Trim time_series to keep the JSONL viewer-friendly; the app
                # also slices for plotting, but the file stays usable in
                # plain editors.
                row = {
                    "_source_path": str(p),
                    "stratum": f"L{key[0]}_t{key[1]}",
                    **q,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_items += 1

    print(f"Wrote {n_items} items across {n_strata} strata to {args.output}")
    print(f"Strata (level, template_id) -> count after sampling:")
    for key in sorted(by_stratum):
        n_avail = len(by_stratum[key])
        n_taken = min(args.per_stratum, n_avail)
        print(f"  L{key[0]} t{key[1]}: {n_taken}/{n_avail}")


if __name__ == "__main__":
    main()
