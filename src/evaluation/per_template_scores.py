"""Aggregate reply scores into a per-template CSV.

Walks ``replies_root/level{N}/<model>/*_answer.json``, joins each reply with its
source question from ``questions_root/level{N}/<stem>.json`` to recover
``template_id`` / ``template_type`` / ``answer_format``, then pivots model
columns and adds ``mean_across_models``.

Pass ``--chance-correct`` to apply the FactoryBench chance correction
(``src.evaluation.chance_correct``) before aggregation. Per-format chance
levels: 1/k for single-select MCQ (k from option count), 1/2 for multi-select,
1/n for ranking, 1/4 for tensor / numerical (calibrated). Free-form is
returned unchanged.

Usage:
    python -m src.evaluation.per_template_scores \\
        --levels 1,2,3 \\
        --chance-correct \\
        --output output/per_template_scores_2026-04-28.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.evaluation.chance_correct import chance_correct

LEVEL_RE = re.compile(r"level(\d+)")


def _q_stem(reply_stem: str) -> str:
    s = reply_stem[:-len("_answer")] if reply_stem.endswith("_answer") else reply_stem
    parts = s.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return s


def _load_question_index(level_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for qf in level_dir.glob("*.json"):
        try:
            index[qf.stem] = json.loads(qf.read_text(encoding="utf-8"))
        except Exception:
            continue
    return index


def collect_one(
    replies_root: Path,
    questions_root: Path,
    levels: list[int],
    condition: str,
    apply_chance_correct: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for level in levels:
        q_index = _load_question_index(questions_root / f"level{level}")
        rdir = replies_root / f"level{level}"
        if not rdir.exists():
            continue
        for model_dir in sorted(p for p in rdir.iterdir() if p.is_dir()):
            for reply_path in model_dir.glob("*_answer.json"):
                try:
                    r = json.loads(reply_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(r, dict) or r.get("score") is None:
                    continue
                q = q_index.get(_q_stem(reply_path.stem)) or {}
                q_af = q.get("answer_format")
                if isinstance(q_af, dict):
                    q_af = q_af.get("type")
                resolved_af = q_af or r.get("answer_format") or "unknown"
                raw_score = float(r["score"])
                score = (
                    chance_correct(raw_score, resolved_af, q)
                    if apply_chance_correct
                    else raw_score
                )
                if score is None:
                    continue
                rows.append({
                    "level": level,
                    "template_id": q.get("template_id"),
                    "condition": condition,
                    "answer_format": resolved_af,
                    "template_type": q.get("template_type") or r.get("question_type") or "unknown",
                    "model": model_dir.name,
                    "score": float(score),
                })
    return rows


def collect(
    replies_root: Path,
    questions_root: Path,
    levels: list[int],
    noised_replies_root: Path | None = None,
    noised_questions_root: Path | None = None,
    apply_chance_correct: bool = False,
) -> pd.DataFrame:
    rows = collect_one(replies_root, questions_root, levels, "normal", apply_chance_correct)
    if noised_replies_root and noised_questions_root:
        rows.extend(collect_one(noised_replies_root, noised_questions_root, levels, "noised", apply_chance_correct))
    return pd.DataFrame(rows)


def pivot(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["level", "template_id", "condition", "answer_format", "template_type"]
    agg = df.groupby(group_cols + ["model"], dropna=False)["score"].mean().reset_index()
    n = df.groupby(group_cols, dropna=False).size().rename("n").reset_index()
    wide = agg.pivot_table(index=group_cols, columns="model", values="score").reset_index()
    out = n.merge(wide, on=group_cols, how="left")
    model_cols = [c for c in out.columns if c not in group_cols + ["n"]]
    out["mean_across_models"] = out[model_cols].mean(axis=1).round(3)
    out[model_cols] = out[model_cols].round(3)
    return out.sort_values(["level", "template_id", "condition", "answer_format"]).reset_index(drop=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replies-root", type=Path, default=repo_root / "output" / "replies")
    p.add_argument("--questions-root", type=Path, default=repo_root / "output" / "questions")
    p.add_argument("--noised-replies-root", type=Path, default=repo_root / "output" / "replies_noised",
                   help="When --include-noised is set, look here for noised replies.")
    p.add_argument("--noised-questions-root", type=Path, default=repo_root / "output" / "questions_noised",
                   help="When --include-noised is set, look here for noised questions.")
    p.add_argument("--include-noised", action="store_true",
                   help="Also collect rows with condition=noised from --noised-* roots.")
    p.add_argument("--levels", type=str, default="1,2,3")
    p.add_argument("--output", type=Path, default=repo_root / "output" / "per_template_scores.csv")
    p.add_argument("--chance-correct", action="store_true",
                   help="Apply per-format chance correction (max(0, (s-E)/(1-E))) before aggregating.")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    df = collect(
        args.replies_root,
        args.questions_root,
        levels,
        noised_replies_root=args.noised_replies_root if args.include_noised else None,
        noised_questions_root=args.noised_questions_root if args.include_noised else None,
        apply_chance_correct=args.chance_correct,
    )
    if df.empty:
        raise SystemExit(f"No scored replies found under {args.replies_root} for levels {levels}")
    out = pivot(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
