"""Post-hoc batch scoring of L4 free-form replies with a 3-judge ensemble.

Walks ``output/replies/level4/<model_slug>/*_answer.json``, picks out
free-form items whose template is **troubleshooting** (id=1) or
**optimization** (id=2), and grades them with three judges in parallel:
gpt-5.1-1 (foundry batch), claude-sonnet-4.6 + deepseek-v3.2 (bedrock
batch). Each judge submits ONE batch covering every item; results are
joined by ``custom_id`` and median-voted to produce the final score.

Per-reply writeback (in place):
  * ``llm_judge_score``, median across the 3 valid votes (snapped to
                          {0, 0.5, 1}).
  * ``llm_judge_reason``, short aggregation summary.
  * ``llm_judge_votes``, per-judge breakdown ``{model: {score, reason}}``
                          for audit.

Usage::

    python -m scripts.score_replies_batch \
        --replies-root output/replies \
        --questions-root output/questions \
        --levels 4

By default this is a no-op for any item already scored unless ``--rescore``
is passed; the existing single-judge ``llm_judge_score`` is otherwise
overwritten on every run.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.scoring.multi_judge import (
    DEFAULT_JUDGE_MODELS, JudgeItem, score_all)

logger = logging.getLogger(__name__)

# Only these two L4 templates have free-form rubrics. Templates 3 and 4
# (ranking) score deterministically and don't need the judge.
JUDGED_TEMPLATE_IDS = {1, 2}

def _question_path_for(reply_file: Path, questions_root: Path) -> Optional[Path]:
    """Derive the question JSON path from a reply filename.

    Handles both filename formats produced by the pipeline:
      - numeric index:  ``level4_0042_2_answer.json``
      - UUID (HF eval): ``level4_<uuid>_2_answer.json``

    The question stem is always ``{reply_stem_without_tid}``, i.e. we strip
    ``_answer.json``, then strip the last ``_<tid>`` token.
    """
    name = reply_file.name
    if not name.endswith("_answer.json"):
        return None
    custom_id = name[: -len("_answer.json")]   # e.g. "level4_<uuid>_0"
    question_stem = custom_id.rsplit("_", 1)[0]  # e.g. "level4_<uuid>"
    m = re.match(r"^level(\d+)_", question_stem)
    if not m:
        return None
    level = m.group(1)
    return questions_root / f"level{level}" / f"{question_stem}.json"


def _has_failed_judge(reply: dict, judges: List[str]) -> bool:
    """Return True if any of ``judges`` has a None score in llm_judge_votes."""
    votes = reply.get("llm_judge_votes") or {}
    return any(
        (votes.get(j) or {}).get("score") is None
        for j in judges
    )


def _collect_items(
    replies_root: Path,
    questions_root: Path,
    levels: List[int],
    rescore: bool,
    judges: Optional[List[str]] = None) -> Tuple[List[JudgeItem], Dict[str, Path]]:
    """Walk the replies tree, return judge items + a {custom_id: reply_path}
    map for writeback.

    ``custom_id`` is namespaced as ``<model_slug>__<reply_custom_id>`` so the
    same prompt scored across multiple inference models stays distinct in the
    judge batch.

    Skip logic (in order):
      * Always skip items with no matching question or wrong template.
      * ``--rescore``: include everything.
      * Otherwise: skip items where all requested judges already have a valid
        (non-None) score in ``llm_judge_votes``.
    """
    items: List[JudgeItem] = []
    cid_to_path: Dict[str, Path] = {}
    skipped_no_q = skipped_wrong_template = skipped_already = 0

    for level in levels:
        level_dir = replies_root / f"level{level}"
        if not level_dir.is_dir():
            logger.warning(f"no replies dir for level {level}: {level_dir}")
            continue
        for model_dir in sorted(level_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_slug = model_dir.name
            for reply_file in sorted(model_dir.glob("level*_*_answer.json")):
                try:
                    reply = json.loads(reply_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(f"unreadable reply {reply_file}: {exc}")
                    continue
                if reply.get("answer_format") != "free_form":
                    continue
                q_path = _question_path_for(reply_file, questions_root)
                if q_path is None or not q_path.exists():
                    skipped_no_q += 1
                    continue
                try:
                    q = json.loads(q_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(f"unreadable question {q_path}: {exc}")
                    continue
                tid = q.get("template_id")
                if tid not in JUDGED_TEMPLATE_IDS:
                    skipped_wrong_template += 1
                    continue
                if not rescore:
                    # Skip only if all requested judges already have valid scores.
                    if judges and not _has_failed_judge(reply, judges):
                        skipped_already += 1
                        continue
                    elif not judges and reply.get("llm_judge_votes"):
                        skipped_already += 1
                        continue

                cid = f"{model_slug}__{reply.get('custom_id') or reply_file.stem}"
                # Use the question-template text only (e.g. "Given the sensor
                # stream below, does the machine show signs of anomalous
                # behavior? ..."). NEVER fall back to ``reply['prompt']``, # that contains the full sensor timeseries context, which is
                # both wasteful in the judge prompt and not what we want the
                # judge to read.
                items.append(JudgeItem(
                    custom_id=cid,
                    template_id=int(tid),
                    question=q.get("question") or "(question text not available)",
                    prediction=reply.get("answer") or "",
                    reference=str(reply.get("ground_truth") or q.get("answer") or ""),
                    root_cause=q.get("root_cause")))
                cid_to_path[cid] = reply_file

    logger.info(
        f"collected {len(items)} items "
        f"(skipped: {skipped_no_q} no-question, "
        f"{skipped_wrong_template} wrong-template, "
        f"{skipped_already} already-scored)"
    )
    return items, cid_to_path


def _writeback(
    cid_to_path: Dict[str, Path],
    results) -> int:
    """Merge llm_judge_{score,reason,votes} back into each reply file.

    New votes are merged into any existing ``llm_judge_votes`` dict so that a
    partial retry (e.g. only gpt-5.1-1 after it failed) does not overwrite
    votes from judges that already succeeded. The ensemble score and reason are
    recomputed from the merged vote set.
    Returns the count of files updated.
    """
    from src.scoring.multi_judge import _aggregate, JudgeVote
    n = 0
    for cid, res in results.items():
        path = cid_to_path.get(cid)
        if path is None:
            continue
        try:
            reply = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"writeback: unreadable {path}: {exc}")
            continue

        # Merge: start from existing votes, overwrite with new ones.
        existing = reply.get("llm_judge_votes") or {}
        merged = dict(existing)
        for model, vote in res.votes.items():
            merged[model] = {"score": vote.score, "reason": vote.reason}

        # Recompute ensemble score from merged votes.
        merged_votes = {
            m: JudgeVote(score=v.get("score"), reason=v.get("reason", ""))
            for m, v in merged.items()
        }
        final_score, final_reason = _aggregate(merged_votes)

        reply["llm_judge_score"] = final_score
        reply["llm_judge_reason"] = final_reason
        reply["llm_judge_votes"] = merged
        path.write_text(json.dumps(reply, indent=2), encoding="utf-8")
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--replies-root", type=Path, default=Path("output/replies"))
    parser.add_argument("--questions-root", type=Path, default=Path("output/questions"))
    parser.add_argument("--levels", nargs="+", type=int, default=[4],
                        help="Levels to score. Default: 4 (only L4 has free-form rubric items).")
    parser.add_argument("--judges", nargs="+", default=list(DEFAULT_JUDGE_MODELS),
                        help="Judge model names. Default: gpt-5.1-1 claude-sonnet-4.6 deepseek-v3.2.")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--sync-concurrency", type=int, default=0,
                        help="Use concurrent sync calls for foundry judges instead of batch. "
                             "Set to e.g. 20 when the Azure batch upload endpoint is broken.")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score items that already have llm_judge_votes.")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Submit judge batches sequentially instead of in parallel.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    from src.evaluation.test_gpt_5mini import load_dotenv_file
    load_dotenv_file(args.env_file)

    items, cid_to_path = _collect_items(
        args.replies_root.resolve(),
        args.questions_root.resolve(),
        args.levels,
        args.rescore,
        judges=args.judges)
    if not items:
        logger.info("nothing to score")
        return

    logger.info(
        f"submitting {len(items)} items × {len(args.judges)} judges "
        f"({'parallel' if not args.no_parallel else 'sequential'})"
    )
    results = score_all(
        items,
        judges=tuple(args.judges),
        max_output_tokens=args.max_output_tokens,
        poll_interval=args.poll_interval,
        parallel_judges=not args.no_parallel,
        sync_concurrency=args.sync_concurrency)

    n_written = _writeback(cid_to_path, results)
    n_scored = sum(1 for r in results.values() if r.final_score is not None)
    logger.info(
        f"done: wrote {n_written} reply files | "
        f"final scores assigned: {n_scored}/{len(results)}"
    )


if __name__ == "__main__":
    main()
