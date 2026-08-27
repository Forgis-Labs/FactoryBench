"""Re-tag (and re-score) reply files whose ``answer_format`` was poisoned.

Background: ``run_aws_eval.py`` previously built its ground-truth index via
``test_gpt_5mini.build_ground_truth_index``, which stored only the bare answer
string. ``infer_answer_format`` then saw a payload with no ``options`` /
``template_type`` and fell through to ``"free_form"`` for every reply, sending
single-letter / TFTF / numeric answers through the LLM judge instead of the
correct exact-match scorer.

This script walks ``replies_root/level{N}/<model>/*_answer.json``, resolves
each reply against ``questions_root/level{N}/<stem>.json``, recomputes
``answer_format`` and ``question_type`` from the full question payload, and, if the format changes, re-scores the existing model answer with the proper
branch of ``score_prediction``. The judge is never invoked: we only ever
transition *away* from ``"free_form"``, never towards it.

Usage:
    # preview
    python -m src.evaluation.retag_replies --levels 1,2,3
    # apply changes in-place
    python -m src.evaluation.retag_replies --levels 1,2,3 --write
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.evaluation.run_foundry_eval import (
    build_question_index,
    infer_answer_format,
    score_prediction)


def _q_stem(reply_stem: str) -> str:
    s = reply_stem[:-len("_answer")] if reply_stem.endswith("_answer") else reply_stem
    parts = s.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return s


def retag_one(
    reply_path: Path,
    qa_payload: Dict[str, Any],
    write: bool) -> Optional[Dict[str, Any]]:
    """Return a summary dict if anything changed, else None."""
    try:
        reply = json.loads(reply_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(reply, dict):
        return None

    new_af = infer_answer_format(qa_payload)
    new_qt = str(qa_payload.get("template_type") or qa_payload.get("type") or "unknown")

    old_af = reply.get("answer_format")
    old_qt = reply.get("question_type")
    if old_af == new_af and old_qt == new_qt:
        return None

    # Re-score under the corrected format. We never re-enter free_form here,
    # so the judge is never called (judge_model="").
    new_score = reply.get("score")
    new_judge_score = reply.get("llm_judge_score")
    new_judge_reason = reply.get("llm_judge_reason")
    new_provenance = reply.get("parse_provenance")
    if old_af != new_af:
        score, judge_result, provenance = score_prediction(
            answer_format=new_af,
            prediction=reply.get("answer"),
            ground_truth=reply.get("ground_truth", qa_payload.get("answer")),
            acceptance_bounds=qa_payload.get("acceptance_bounds"),
            question_text=qa_payload.get("question", ""),
            judge_model="")
        new_score = score
        new_provenance = provenance
        if judge_result is None:
            new_judge_score = None
            new_judge_reason = None
        else:
            new_judge_score, new_judge_reason = judge_result

    summary = {
        "path": str(reply_path),
        "old_answer_format": old_af,
        "new_answer_format": new_af,
        "old_question_type": old_qt,
        "new_question_type": new_qt,
        "old_score": reply.get("score"),
        "new_score": new_score,
    }

    if write:
        reply["answer_format"] = new_af
        reply["question_type"] = new_qt
        reply["score"] = new_score
        reply["parse_provenance"] = new_provenance
        reply["llm_judge_score"] = new_judge_score
        reply["llm_judge_reason"] = new_judge_reason
        reply_path.write_text(json.dumps(reply, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replies-root", type=Path, default=repo_root / "output" / "replies")
    p.add_argument("--questions-root", type=Path, default=repo_root / "output" / "questions")
    p.add_argument("--levels", type=str, default="1,2,3")
    p.add_argument("--write", action="store_true", help="Apply changes in-place (default: dry-run)")
    p.add_argument("--max-print", type=int, default=20, help="Max change-rows to print")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    total_changed = 0
    total_seen = 0
    by_model: Dict[str, int] = {}
    sample_rows: list[Dict[str, Any]] = []

    for level in levels:
        q_index = build_question_index(args.questions_root / f"level{level}")
        rdir = args.replies_root / f"level{level}"
        if not rdir.exists():
            continue
        for model_dir in sorted(p for p in rdir.iterdir() if p.is_dir()):
            for reply_path in model_dir.glob("*_answer.json"):
                total_seen += 1
                stem = _q_stem(reply_path.stem)
                qa = q_index.get(stem)
                if not qa:
                    continue
                changed = retag_one(reply_path, qa, write=args.write)
                if changed:
                    total_changed += 1
                    by_model[model_dir.name] = by_model.get(model_dir.name, 0) + 1
                    if len(sample_rows) < args.max_print:
                        sample_rows.append(changed)

    print(f"Scanned {total_seen} reply files; {total_changed} need re-tagging.")
    if by_model:
        print("By model:")
        for m, n in sorted(by_model.items(), key=lambda kv: -kv[1]):
            print(f"  {m}: {n}")
    if sample_rows:
        print(f"\nFirst {len(sample_rows)} changes:")
        for s in sample_rows:
            rel = re.sub(r".*[\\/]replies[\\/]", "", s["path"])
            print(
                f"  {rel}: af={s['old_answer_format']!r}->{s['new_answer_format']!r} "
                f"qt={s['old_question_type']!r}->{s['new_question_type']!r} "
                f"score={s['old_score']}->{s['new_score']}"
            )
    if not args.write and total_changed:
        print("\nDry run, re-run with --write to apply changes in-place.")


if __name__ == "__main__":
    main()
