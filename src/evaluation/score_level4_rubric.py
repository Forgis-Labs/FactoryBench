"""Deterministic rubric scorer for Level 4 free-form answers.

No LLM is called. Each reply is scored 0.0 / 0.5 / 1.0 based on textual matches
against the gold answer fields.

  Troubleshooting (template 1):
    - 1.0 : reply text covers the gold corrective protocol (token-overlap above
            PROTOCOL_THRESHOLD against the gold answer's content tokens).
    - 0.5 : reply mentions the gold root_cause (snake_case → space-separated)
            but does not match the protocol.
    - 0.0 : neither.

  Optimization (template 2):
    - 1.0 : reply mentions the misconfigured parameter values AND covers the
            gold protocol.
    - 0.5 : reply mentions the misconfigured parameter values (extracted as
            numeric+unit tokens from the gold answer) but doesn't match the
            protocol.
    - 0.0 : neither.

Usage:
    python -m src.evaluation.score_level4_rubric \
        --replies-root output/replies output/replies_noised \
        --questions-root output/questions
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Stopwords kept short, the goal is to drop words that carry no protocol meaning
# (the / a / and / etc.) without over-pruning. Anything else with length >= 3 is
# treated as content.
STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "that", "this", "from", "into", "onto",
    "are", "was", "were", "has", "have", "had", "but", "not", "any", "all",
    "you", "your", "its", "their", "they", "them", "his", "her", "our",
    "out", "off", "via", "per", "than", "then", "when", "what", "which",
    "where", "who", "how", "such", "also", "yet", "may", "might", "could",
    "would", "should", "shall", "can", "will", "just", "more", "most", "very",
    "much", "some", "few", "each", "every", "both", "either", "neither",
}

PROTOCOL_THRESHOLD = 0.40  # fraction of gold content tokens that must appear in reply

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
# Numeric value with optional unit (e.g. "1.5 kg", "55 mm", "100°C")
_NUM_UNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+|°[CFK])?")


def _content_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    out: Set[str] = set()
    for m in _TOKEN_RE.finditer(text.lower()):
        tok = m.group(0)
        if tok in STOPWORDS:
            continue
        out.add(tok)
    return out


def _root_cause_keywords(root_cause: Optional[str]) -> List[str]:
    """Turn a snake_case root_cause id into a list of words to check for."""
    if not root_cause:
        return []
    return [w for w in re.split(r"[_\s\-]+", root_cause.lower()) if len(w) >= 3 and w not in STOPWORDS]


def _gold_numbers(text: str) -> List[str]:
    """Extract distinct numeric values from gold (with units when present)."""
    seen: Set[str] = set()
    out: List[str] = []
    for m in _NUM_UNIT_RE.finditer(text):
        num = m.group(1)
        try:
            float(num)
        except ValueError:
            continue
        # Skip integers that look like incidental indices (e.g. "1.", "0")
        # Keep them only if a unit is attached.
        unit = (m.group(2) or "").strip()
        key = f"{num}{unit}".lower()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _root_cause_present(reply: str, root_cause: Optional[str]) -> bool:
    if not root_cause:
        return False
    reply_l = reply.lower()
    # Direct identifier hit (snake_case form).
    if root_cause.lower() in reply_l:
        return True
    # Or every keyword from the root_cause name appears somewhere.
    keywords = _root_cause_keywords(root_cause)
    if not keywords:
        return False
    return all(kw in reply_l for kw in keywords)


def _protocol_match(reply: str, gold_answer: str) -> Tuple[bool, float]:
    gold_tokens = _content_tokens(gold_answer)
    if not gold_tokens:
        return False, 0.0
    reply_tokens = _content_tokens(reply)
    if not reply_tokens:
        return False, 0.0
    overlap = len(gold_tokens & reply_tokens) / len(gold_tokens)
    return overlap >= PROTOCOL_THRESHOLD, overlap


def _params_present(reply: str, gold_answer: str) -> Tuple[bool, List[str], List[str]]:
    """Return (all_present, present, missing). True iff every gold number value
    (e.g. '1.5kg') appears in the reply (as a number; the unit is optional)."""
    gold_nums = _gold_numbers(gold_answer)
    if not gold_nums:
        return False, [], []
    reply_l = reply.lower()
    present: List[str] = []
    missing: List[str] = []
    for token in gold_nums:
        # Match the bare numeric prefix; we don't require the unit (units in the
        # reply may differ in spacing / case).
        m = re.match(r"-?\d+(?:\.\d+)?", token)
        num = m.group(0) if m else token
        if num in reply_l:
            present.append(token)
        else:
            missing.append(token)
    return (not missing), present, missing


def score_one(question: Dict[str, Any], reply: Dict[str, Any]) -> Optional[Tuple[float, str]]:
    template_type = str(question.get("template_type") or "").lower()
    if template_type not in ("troubleshooting", "optimization"):
        return None
    reply_text = str(reply.get("answer") or "")
    gold = str(question.get("answer") or "")
    rc = question.get("root_cause")
    protocol_ok, overlap = _protocol_match(reply_text, gold)

    if template_type == "troubleshooting":
        rc_ok = _root_cause_present(reply_text, rc)
        if protocol_ok:
            return 1.0, f"protocol covered (token overlap={overlap:.2f})"
        if rc_ok:
            return 0.5, f"root cause '{rc}' present; protocol not covered (overlap={overlap:.2f})"
        return 0.0, f"root cause not found and protocol not covered (overlap={overlap:.2f})"

    # optimization
    params_ok, present, missing = _params_present(reply_text, gold)
    if params_ok and protocol_ok:
        return 1.0, f"all parameters {present} present; protocol covered (overlap={overlap:.2f})"
    if params_ok:
        return 0.5, f"parameters {present} present but protocol not covered (overlap={overlap:.2f})"
    if protocol_ok:
        return 0.5, f"protocol covered but parameters {missing} missing (overlap={overlap:.2f})"
    return 0.0, f"parameters {missing} missing and protocol not covered (overlap={overlap:.2f})"


def index_questions(questions_root: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for p in (questions_root / "level4").rglob("*.json"):
        try:
            idx[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return idx


def _question_stem_from_reply(reply_stem: str) -> str:
    # 'level4_0083_83_answer' -> 'level4_0083'
    parts = reply_stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else reply_stem


def process_replies_root(
    replies_root: Path,
    questions: Dict[str, Dict[str, Any]],
    overwrite: bool,
    dry_run: bool) -> Tuple[int, int, int]:
    n_scored = 0
    n_skipped = 0
    n_lookup_fail = 0
    l4_dir = replies_root / "level4"
    if not l4_dir.is_dir():
        logger.warning(f"No level4 dir under {replies_root}; skipping")
        return 0, 0, 0
    for model_dir in sorted(l4_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name == "baseline_simple":
            continue
        for reply_path in sorted(model_dir.glob("*_answer.json")):
            try:
                reply = json.loads(reply_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not overwrite and reply.get("rubric_score") is not None:
                n_skipped += 1
                continue
            q_stem = _question_stem_from_reply(reply_path.stem)
            q = questions.get(q_stem)
            if q is None:
                n_lookup_fail += 1
                continue
            result = score_one(q, reply)
            if result is None:
                continue
            score, reason = result
            reply["score"] = score
            reply["rubric_score"] = score
            reply["llm_judge_score"] = score  # downstream tooling reads this
            reply["llm_judge_reason"] = reason
            reply["answer_format"] = "free_form_rubric"
            if not dry_run:
                reply_path.write_text(json.dumps(reply, indent=2, ensure_ascii=False), encoding="utf-8")
            n_scored += 1
    return n_scored, n_skipped, n_lookup_fail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replies-root", nargs="+", type=Path, default=[Path("output/replies")])
    parser.add_argument("--questions-root", type=Path, default=Path("output/questions"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-score items that already have rubric_score (default: skip)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    questions = index_questions(args.questions_root)
    logger.info(f"Indexed {len(questions)} L4 questions from {args.questions_root}")
    grand = 0
    for root in args.replies_root:
        if not root.exists():
            logger.warning(f"Skipping {root}: does not exist")
            continue
        scored, skipped, lookup_fail = process_replies_root(
            root, questions, args.overwrite, args.dry_run)
        logger.info(f"{root}: scored {scored}, skipped {skipped}, lookup-failed {lookup_fail}")
        grand += scored
    logger.info(f"Done. Total rubric scores written: {grand}")


if __name__ == "__main__":
    main()
