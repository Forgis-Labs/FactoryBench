"""Quick-look at FactoryBench prediction JSONLs from a SageMaker eval run.

Usage:
    python finetuning/peek_predictions.py                        # all level_*_predictions.jsonl in cwd
    python finetuning/peek_predictions.py level_1_predictions.jsonl
    python finetuning/peek_predictions.py *.jsonl --n 5          # show 5 per file
    python finetuning/peek_predictions.py *.jsonl --wrong        # only mismatches
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*",
                   help="JSONL files (glob expanded). Default: level_*_predictions.jsonl")
    p.add_argument("--n", type=int, default=10,
                   help="How many samples to print per file (default 10).")
    p.add_argument("--wrong", action="store_true",
                   help="Only show records where answer != ground_truth.")
    p.add_argument("--show-question", action="store_true",
                   help="Also print the question (truncated to 160 chars).")
    args = p.parse_args()

    patterns = args.files or ["level_*_predictions.jsonl"]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path(x) for x in sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit(f"No JSONL files matched: {patterns}")

    for path in paths:
        print(f"\n=== {path} ===")
        total = 0
        correct = 0
        shown = 0
        token_counts: list[int] = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                total += 1
                gt = r.get("ground_truth")
                ans = (r.get("answer") or "").strip()
                ok = (gt is not None and ans == str(gt).strip())
                if ok:
                    correct += 1
                token_counts.append(r.get("n_pred_tokens", 0))
                if shown >= args.n:
                    continue
                if args.wrong and ok:
                    continue
                mark = "OK" if ok else "  "
                line_out = (
                    f"  [{mark}] L{r.get('level')} idx={r.get('idx')}  "
                    f"GT={str(gt)!r:<14}  PRED={ans!r}  "
                    f"({r.get('n_pred_tokens')} tok)"
                )
                print(line_out)
                if args.show_question:
                    q = (r.get("question") or "").replace("\n", " ")[:160]
                    print(f"        Q: {q}")
                shown += 1
        acc = correct / total if total else 0.0
        avg_tok = sum(token_counts) / len(token_counts) if token_counts else 0
        max_tok = max(token_counts) if token_counts else 0
        print(f"  -> {correct}/{total} exact-match ({acc:.3f}), "
              f"avg pred tokens = {avg_tok:.1f}, max = {max_tok}")


if __name__ == "__main__":
    main()
