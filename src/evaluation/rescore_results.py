"""Re-score already-generated `*_answer.json` files.

Earlier runs saved replies with `answer_format: "unknown"` because the code
read it from the prompt JSON (which doesn't carry that field) instead of the
question JSON. This script walks a replies tree, finds each reply's
corresponding question JSON, re-infers the answer format, recomputes the
score with the fixed logic, and writes the updated reply back in place.

Usage:
    python -m src.evaluation.rescore_results \
        --replies-root output/replies \
        --questions-root output/questions

    # single model:
    python -m src.evaluation.rescore_results \
        --replies-root output/replies/level1/gpt-5.1 \
        --questions-root output/questions
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.evaluation.run_foundry_eval import (
    infer_answer_format,
    score_prediction,
    build_question_index)
from src.evaluation.test_gpt_5mini import load_json, save_json

logger = logging.getLogger(__name__)

LEVEL_RE = re.compile(r"level(\d+)")


def _level_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        m = LEVEL_RE.fullmatch(part) or LEVEL_RE.match(part)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def _question_stem_from_reply_stem(reply_stem: str) -> str:
    """`level1_0000_0_answer` -> `level1_0000`."""
    s = reply_stem
    if s.endswith("_answer"):
        s = s[: -len("_answer")]
    # Drop the trailing prompt_index suffix: `level1_0000_0` -> `level1_0000`.
    parts = s.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return s


def rescore_reply(
    reply_path: Path,
    question_indices: Dict[int, Dict[str, Dict[str, Any]]],
    judge_model: str,
    dry_run: bool = False,
    rerun_judge: bool = False) -> Optional[Dict[str, Any]]:
    """Load reply, rescore, save back. Returns the updated record (or None)."""
    try:
        record = load_json(reply_path)
    except Exception as exc:
        logger.warning(f"skip {reply_path}: unreadable ({exc})")
        return None
    if not isinstance(record, dict):
        return None

    level = _level_from_path(reply_path)
    if level is None or level not in question_indices:
        logger.warning(f"skip {reply_path}: no question index for level {level}")
        return None

    q_stem = _question_stem_from_reply_stem(reply_path.stem)
    q = question_indices[level].get(q_stem)
    if not q:
        logger.warning(f"skip {reply_path}: no question {q_stem!r} for level {level}")
        return None

    new_format = infer_answer_format(q)
    gt = q.get("answer")
    pred = record.get("answer")
    acceptance_bounds = q.get("acceptance_bounds")
    question_text = q.get("question") or ""

    # Free-form scoring requires the judge LLM. By default we NEVER call it, # only recompute if there's already a saved judge score, or the user
    # explicitly passed --rerun-judge. Otherwise leave score as None.
    new_provenance: Optional[str] = None
    if new_format == "free_form":
        saved_judge = record.get("llm_judge_score")
        if saved_judge is not None:
            new_score = float(saved_judge)
            new_judge = (float(saved_judge), record.get("llm_judge_reason"))
            new_provenance = "judge"
        elif rerun_judge:
            new_score, new_judge, new_provenance = score_prediction(
                new_format, pred, gt, acceptance_bounds, question_text, judge_model)
        else:
            new_score = None
            new_judge = None
    else:
        new_score, new_judge, new_provenance = score_prediction(
            new_format, pred, gt, acceptance_bounds, question_text, judge_model)

    record["answer_format"] = new_format
    record["question_type"] = str(q.get("template_type") or record.get("question_type") or "unknown")
    record["ground_truth"] = gt
    record["score"] = new_score
    if new_provenance is not None:
        record["parse_provenance"] = new_provenance
    if new_judge is not None:
        record["llm_judge_score"] = new_judge[0]
        record["llm_judge_reason"] = new_judge[1]

    if not dry_run:
        save_json(reply_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore existing reply JSONs with corrected format inference.")
    parser.add_argument("--replies-root", type=Path, default=Path("output/replies"),
                        help="Directory containing level*/<model>/*_answer.json (default: output/replies)")
    parser.add_argument("--questions-root", type=Path, default=Path("output/questions"),
                        help="Directory containing level*/* question JSONs (default: output/questions)")
    parser.add_argument("--judge-model", type=str, default="gpt-5.1",
                        help="LLM-as-judge model for free-form (only used if --rerun-judge)")
    parser.add_argument("--rerun-judge", action="store_true",
                        help="Re-invoke the LLM judge for free-form answers (costs API calls)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    question_indices: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for lvl_dir in sorted(args.questions_root.glob("level*")):
        m = LEVEL_RE.match(lvl_dir.name)
        if not m:
            continue
        lvl = int(m.group(1))
        question_indices[lvl] = build_question_index(lvl_dir)
        logger.info(f"level {lvl}: {len(question_indices[lvl])} question payloads from {lvl_dir}")

    if not question_indices:
        parser.error(f"No question dirs found under {args.questions_root}")

    format_counts: Dict[str, int] = {}
    updated = 0
    unchanged = 0
    skipped = 0
    for reply_path in sorted(args.replies_root.rglob("*_answer.json")):
        if reply_path.name.startswith("_"):
            continue
        try:
            before = load_json(reply_path)
        except Exception:
            skipped += 1
            continue
        before_score = before.get("score") if isinstance(before, dict) else None
        record = rescore_reply(
            reply_path, question_indices,
            judge_model=args.judge_model,
            dry_run=args.dry_run,
            rerun_judge=args.rerun_judge)
        if record is None:
            skipped += 1
            continue
        format_counts[record["answer_format"]] = format_counts.get(record["answer_format"], 0) + 1
        if before_score != record.get("score"):
            updated += 1
        else:
            unchanged += 1

    logger.info("=" * 50)
    logger.info(f"Files updated: {updated} | unchanged: {unchanged} | skipped: {skipped}")
    logger.info(f"Inferred format distribution: {dict(sorted(format_counts.items()))}")
    if args.dry_run:
        logger.info("(dry run, no files written)")


if __name__ == "__main__":
    main()
