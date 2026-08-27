"""Re-score a SageMaker eval's level_X_predictions.jsonl using the official
FactoryBench scoring cascade.

Exact-match (what eval_factorybench.py logs live) is the wrong metric for most
templates, rankings want Kendall-tau, T/F wants per-bit agreement, numerics
want tolerance bands, and L4 free-form needs an LLM judge. This script wraps
the existing ``src.scoring.cascade`` so the numbers match what the rest of the
repo reports.

Usage:
    # Deterministic only (offline, no API): scores L1/L2/L3 properly; L4 free-form
    # samples are returned as `unparseable` with score=None.
    python finetuning/score_predictions.py baseline_bm/

    # Add LLM judge for L4 free-form + escalation on parse failures.
    # Requires Azure Foundry / Bedrock credentials wired into src.evaluation.
    python finetuning/score_predictions.py baseline_bm/ --judge

    # Score a single file
    python finetuning/score_predictions.py baseline_bm/level_2_predictions.jsonl

Outputs:
    <input>_scored.jsonl     ─ per-sample row + {score, provenance, reason}
    <input>_summary.json     ─ aggregate stats (mean, format breakdown...)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

# Make `src.scoring.*` importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.scoring.cascade import parse_only, score as cascade_score  # noqa: E402
from src.scoring.types import ParseResult  # noqa: E402


# -----------------------------------------------------------------------------
# answer_format inference, inlined from src.evaluation.run_foundry_eval so we
# don't drag in the full Foundry/Bedrock imports just to dispatch on format.
# -----------------------------------------------------------------------------

def infer_answer_format(q: dict[str, Any]) -> str:
    """Derive answer_format from level / template_type / options / answer shape."""
    level = q.get("level")
    tid = q.get("template_id")
    template_type = str(q.get("template_type") or "").lower()
    options = q.get("options") or {}
    ans = q.get("ground_truth") if "ground_truth" in q else q.get("answer")

    if level == 4:
        if tid in (3, 4) or "ranking" in template_type:
            return "ranking"
        return "free_form"
    if "ranking" in template_type:
        return "ranking"

    if isinstance(ans, str):
        s = ans.strip()
        s_upper = s.upper()
        if s and set(s_upper) <= {"T", "F"} and len(s_upper) >= 2:
            return "multiple_choice_multi_select"
        if len(s_upper) >= 3 and set(s_upper) <= set("ABCD"):
            return "ranking"
        if len(s) == 1 and s_upper in "ABCDEFGH" and isinstance(options, dict) and options:
            return "multiple_choice_single_select"
        if s.startswith("["):
            return "tensor"
    if isinstance(ans, list):
        return "tensor"
    if isinstance(ans, (int, float)):
        return "numerical"
    if isinstance(ans, str):
        try:
            float(ans)
            return "numerical"
        except ValueError:
            pass
    return "free_form"


# -----------------------------------------------------------------------------
# Scoring driver
# -----------------------------------------------------------------------------

def score_file(path: Path, judge=None) -> dict:
    """Score one level_X_predictions.jsonl. Returns the summary dict."""
    out_path = path.with_name(path.stem + "_scored.jsonl")
    sum_path = path.with_name(path.stem + "_summary.json")

    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = 0
    by_format: dict[str, list[float]] = defaultdict(list)
    by_provenance: dict[str, int] = defaultdict(int)
    unparseable_examples: list[dict] = []
    scores_all: list[float] = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in rows:
            af = infer_answer_format(r)
            pred = r.get("answer") or ""
            gt = r.get("ground_truth")
            ab = r.get("acceptance_bounds")

            if af == "free_form":
                if judge is None:
                    res = ParseResult(None, None, "unparseable",
                                      "free_form needs --judge")
                else:
                    res = cascade_score(
                        answer_format="free_form",
                        prediction=pred,
                        ground_truth=gt,
                        question=r.get("question"),
                        acceptance_bounds=ab,
                        judge=judge)
            else:
                # Deterministic first; escalate via judge only if available.
                if judge is None:
                    res = parse_only(
                        answer_format=af,
                        prediction=pred,
                        ground_truth=gt,
                        acceptance_bounds=ab)
                else:
                    res = cascade_score(
                        answer_format=af,
                        prediction=pred,
                        ground_truth=gt,
                        question=r.get("question"),
                        acceptance_bounds=ab,
                        judge=judge)

            n += 1
            by_provenance[res.provenance] += 1
            if res.score is not None:
                by_format[af].append(res.score)
                scores_all.append(res.score)
            elif len(unparseable_examples) < 5:
                unparseable_examples.append({
                    "id": r.get("id"), "format": af,
                    "gt": gt, "pred": pred[:200], "reason": res.reason,
                })

            out_f.write(json.dumps({
                **r,
                "answer_format": af,
                "score": res.score,
                "provenance": res.provenance,
                "score_reason": res.reason,
            }) + "\n")

    mean_overall = sum(scores_all) / len(scores_all) if scores_all else None
    by_format_mean = {
        af: {
            "n": len(v),
            "mean": round(sum(v) / len(v), 4),
            "exact_n": sum(1 for s in v if s >= 0.999),
            "exact_rate": round(sum(1 for s in v if s >= 0.999) / len(v), 4),
        }
        for af, v in sorted(by_format.items())
    }

    summary = {
        "file": str(path),
        "n": n,
        "scored_n": len(scores_all),
        "unparseable_n": n - len(scores_all),
        "mean_score": round(mean_overall, 4) if mean_overall is not None else None,
        "by_answer_format": by_format_mean,
        "by_provenance": dict(by_provenance),
        "unparseable_examples": unparseable_examples,
        "scored_path": str(out_path),
    }
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+",
                   help="JSONL files or directories containing level_*_predictions.jsonl")
    p.add_argument("--judge", action="store_true",
                   help="Enable LLM judge for free-form + parse-failure "
                        "escalation. Requires Azure Foundry / Bedrock creds.")
    p.add_argument("--judge-model", default=None,
                   help="Override judge model id (default: gpt-5-mini).")
    args = p.parse_args()

    files: list[Path] = []
    for raw in args.paths:
        pp = Path(raw)
        if pp.is_dir():
            files.extend(sorted(Path(p_) for p_ in glob.glob(
                str(pp / "level_*_predictions.jsonl"))))
        elif pp.is_file():
            files.append(pp)
        else:
            print(f"[skip] no file/dir: {raw}", file=sys.stderr)
    if not files:
        raise SystemExit("No prediction files found.")

    judge = None
    if args.judge:
        from src.scoring.judge import LLMJudge
        kw = {}
        if args.judge_model:
            kw["model"] = args.judge_model
        judge = LLMJudge(**kw)
        print(f"[judge] enabled, model={judge.model}")

    overall_n = 0
    overall_scored = 0
    overall_sum = 0.0
    print()
    for f in files:
        print(f"=== {f}")
        s = score_file(f, judge=judge)
        print(json.dumps({
            "n": s["n"], "scored": s["scored_n"],
            "unparseable": s["unparseable_n"],
            "mean": s["mean_score"],
            "by_format": s["by_answer_format"],
        }, indent=2))
        if s["unparseable_examples"]:
            print("  unparseable examples:")
            for ex in s["unparseable_examples"]:
                print(f"    [{ex['format']}] gt={ex['gt']!r}  "
                      f"pred={ex['pred']!r}  reason={ex['reason']!r}")
        overall_n += s["n"]
        overall_scored += s["scored_n"]
        if s["mean_score"] is not None:
            overall_sum += s["mean_score"] * s["scored_n"]
        print()

    if overall_scored:
        print(f"OVERALL: {overall_scored}/{overall_n} scored,  "
              f"mean = {overall_sum / overall_scored:.4f}")


if __name__ == "__main__":
    main()
