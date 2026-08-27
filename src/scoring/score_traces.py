"""Post-hoc scoring of Opik traces.

Pull predictions from Opik, score them with the cascade (parser -> judge),
and write the new `accuracy` + `parse_provenance` feedback scores back to
Opik. Also dumps a parquet for local analysis.

Usage:
    python -m src.scoring.score_traces --since 2026-04-17T03:58:00Z
    python -m src.scoring.score_traces --since 2026-04-17T03:58:00Z --no-write-back
    python -m src.scoring.score_traces --since 2026-04-17T03:58:00Z --limit 20 --dry-run

When the trace metadata's ``answer_format`` is wrong (e.g. it was logged
before a generator/inference fix), pass ``--questions-dir output/questions/levelN``
to reload the declared ``answer_format.type`` from the local QA files and
re-score each trace under the correct format:

    python -m src.scoring.score_traces --since 2026-04-30T00:00:00Z \\
        --questions-dir output/questions/level1 --overwrite

The script is idempotent: traces that already have a `parse_provenance`
feedback score are skipped unless `--overwrite` is passed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv  # python-dotenv (optional)
except ImportError:
    load_dotenv = None

from src.scoring.cascade import score
from src.scoring.judge import DEFAULT_JUDGE_MODEL, LLMJudge

if load_dotenv is not None:
    load_dotenv()
else:
    # Fallback: project's own .env loader (no python-dotenv dependency).
    from pathlib import Path as _Path
    from src.evaluation.test_gpt_5mini import load_dotenv_file
    _env = _Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        load_dotenv_file(_env)
logger = logging.getLogger(__name__)

# How we surface provenance in Opik (numeric so the existing notebook can
# pivot on it directly).
PROVENANCE_TO_NUMERIC = {
    "strict": 0.0,
    "lenient": 1.0,
    "judge": 2.0,
    "unparseable": 3.0,
}

MODEL_ALIASES = {
    "gpt-5-mini-2025-08-07":     "gpt-5.1",
    "gpt-5.1-2025-11-13":        "gpt-5.1",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
}

# Flush feedback-score writes to Opik in batches of this many.
WRITE_BATCH_SIZE = 50

# Rough token cost for gpt-5-mini, used only for the cost estimate at the end.
# Source: existing pricing in src/evaluation/test_gpt_5mini.py.
GPT5_MINI_PRICE_IN_PER_1K = 0.00025
GPT5_MINI_PRICE_OUT_PER_1K = 0.002
JUDGE_AVG_INPUT_TOKENS = 1500
JUDGE_AVG_OUTPUT_TOKENS = 50


def _normalise_model(raw: str | None) -> str:
    raw = (raw or "unknown").lower()
    return MODEL_ALIASES.get(raw, raw)


def _existing_provenance(trace) -> str | None:
    """Return the existing parse_provenance feedback score (numeric) as text."""
    for fb in (trace.feedback_scores or []):
        if fb.name == "parse_provenance":
            return str(fb.value)
    return None


def _question_text(trace) -> str | None:
    """Best-effort extraction of the question text from the trace input/metadata."""
    inp = trace.input or {}
    meta = trace.metadata or {}
    q = inp.get("question")
    if isinstance(q, str) and q.strip():
        return q
    full_ctx = (meta.get("full_telemetry_context") or {})
    if isinstance(full_ctx, dict):
        text = full_ctx.get("text")
        if isinstance(text, str) and text.strip():
            return text
    return None


def _load_qa_format_index(questions_dir: Path) -> dict[str, str]:
    """Map ``question`` text -> declared ``answer_format.type`` from local QA files.

    Used to override the ``answer_format`` recorded in Opik metadata when it
    was inferred wrong at log-time (e.g. tensor answers serialised as
    ``"a_b_c"`` were misclassified as free_form before the
    ``infer_answer_format`` fix). Indexed by question text because trace
    metadata doesn't always carry a usable ``qa_pair_id``.
    """
    import json
    index: dict[str, str] = {}
    if not questions_dir.is_dir():
        raise FileNotFoundError(f"--questions-dir not found: {questions_dir}")
    for path in questions_dir.rglob("*.json"):
        try:
            qa = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(qa, dict):
            continue
        af = qa.get("answer_format")
        question = qa.get("question")
        if isinstance(af, dict) and isinstance(question, str):
            af_type = str(af.get("type") or "").strip().lower()
            if af_type:
                index[question] = af_type
    logger.info("Loaded %d (question -> answer_format) entries from %s",
                len(index), questions_dir)
    return index


def _flush_batch(client, batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    try:
        client.log_traces_feedback_scores(scores=batch)
    except Exception as e:
        logger.warning("Opik batch write failed (%d items): %s", len(batch), e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc scoring of Opik traces.")
    parser.add_argument("--since", required=True,
                        help='ISO timestamp filter, e.g. "2026-04-17T03:58:00Z"')
    parser.add_argument("--max-traces", type=int, default=100_000,
                        help="Cap on traces fetched from Opik")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM judge; unparseable cases stay as score=None.")
    parser.add_argument("--no-write-back", action="store_true",
                        help="Don't write feedback scores back to Opik (local-only).")
    parser.add_argument("--output", type=Path, default=Path("outputs/scored_traces.parquet"),
                        help="Local output for analysis.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N traces (debug).")
    parser.add_argument("--answer-format", default=None,
                        help="Only score traces with this answer_format "
                             "(e.g. 'ranking'). Useful for targeted re-scores after "
                             "a parser change.")
    parser.add_argument("--questions-dir", type=Path, default=None,
                        help="Directory of QA JSON files (e.g. output/questions/level1). "
                             "When set, the trace's answer_format is overridden by the "
                             "declared answer_format.type in the matching QA file. Use "
                             "this after fixing a misclassified answer_format to "
                             "re-score affected traces under the correct parser.")
    parser.add_argument("--only-overrides", action="store_true",
                        help="With --questions-dir, only score traces whose answer_format "
                             "was actually overridden (skip the rest). Useful for surgical "
                             "re-scores that don't touch correctly-scored traces.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-score traces that already have parse_provenance.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print decisions without scoring or writing.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    project = os.getenv("OPIK_PROJECT_NAME", "FactoryBench")
    workspace = os.getenv("OPIK_WORKSPACE", "anon-workspace")

    judge: LLMJudge | None = None
    if not args.no_judge and not args.dry_run:
        judge = LLMJudge(model=args.judge_model)
        logger.info("Judge enabled: model=%s", args.judge_model)
    else:
        logger.info("Judge DISABLED (no escalation for unparseable cases)")

    import opik
    client = opik.Opik(project_name=project, workspace=workspace)
    logger.info("Fetching traces project=%s workspace=%s since=%s", project, workspace, args.since)
    traces = client.search_traces(
        project_name=project,
        max_results=args.max_traces,
        filter_string=f'start_time > "{args.since}"')
    logger.info("Fetched %d traces", len(traces))

    if args.limit:
        traces = traces[: args.limit]
        logger.info("Limiting to first %d traces", args.limit)

    qa_format_index: dict[str, str] = {}
    if args.questions_dir is not None:
        qa_format_index = _load_qa_format_index(args.questions_dir)

    rows: list[dict[str, Any]] = []
    pending_writes: list[dict[str, Any]] = []
    provenance_counts: Counter[str] = Counter()
    by_model_provenance: dict[str, Counter[str]] = defaultdict(Counter)
    n_judge_free_form = 0
    n_judge_escalation = 0
    n_skipped_already_scored = 0
    n_skipped_no_override = 0
    n_missing_data = 0
    n_format_overridden = 0

    for i, tr in enumerate(traces):
        meta = tr.metadata or {}
        inp = tr.input or {}
        prediction = inp.get("prediction")
        ground_truth = inp.get("ground_truth")
        answer_format = meta.get("answer_format", "unknown")
        model_name = _normalise_model(meta.get("model"))

        if prediction is None or ground_truth is None:
            n_missing_data += 1
            continue

        # Override answer_format from local QA index (when set), fixes traces
        # logged before infer_answer_format learned to honour answer_format.type.
        was_overridden = False
        if qa_format_index:
            q_text = _question_text(tr)
            declared = qa_format_index.get(q_text or "")
            if declared and declared != answer_format:
                answer_format = declared
                n_format_overridden += 1
                was_overridden = True

        if args.only_overrides and not was_overridden:
            n_skipped_no_override += 1
            continue

        if args.answer_format and answer_format != args.answer_format:
            continue

        if not args.overwrite and _existing_provenance(tr) is not None:
            n_skipped_already_scored += 1
            continue

        question = _question_text(tr)
        acceptance_bounds = meta.get("acceptance_bounds")  # rarely present today

        result = score(
            answer_format=answer_format,
            prediction=str(prediction),
            ground_truth=ground_truth,
            question=question,
            acceptance_bounds=acceptance_bounds,
            judge=judge)

        # Telemetry
        provenance_counts[result.provenance] += 1
        by_model_provenance[model_name][result.provenance] += 1
        if result.provenance == "judge":
            if answer_format == "free_form":
                n_judge_free_form += 1
            else:
                n_judge_escalation += 1

        rows.append({
            "trace_id": tr.id,
            "model": model_name,
            "answer_format": answer_format,
            "ground_truth": str(ground_truth),
            "prediction": str(prediction),
            "score": result.score,
            "parsed": str(result.parsed) if result.parsed is not None else None,
            "provenance": result.provenance,
            "reason": result.reason,
        })

        if args.dry_run:
            logger.info(
                "[%d/%d] %s %s score=%s prov=%s",
                i + 1, len(traces), model_name, answer_format, result.score, result.provenance)
            continue

        # Queue feedback writes for batch flush
        if not args.no_write_back and result.score is not None:
            reason_acc = (result.reason or f"parsed={result.parsed!r} gt={ground_truth!r}")[:500]
            pending_writes.extend([
                {
                    "id": tr.id, "name": "accuracy",
                    "value": float(result.score), "reason": reason_acc,
                    "project_name": project,
                },
                {
                    "id": tr.id, "name": "parse_provenance",
                    "value": PROVENANCE_TO_NUMERIC[result.provenance],
                    "reason": result.provenance,
                    "project_name": project,
                },
            ])
            if len(pending_writes) >= WRITE_BATCH_SIZE * 2:  # 2 scores per trace
                _flush_batch(client, pending_writes)
                pending_writes.clear()

        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d traces | judge calls: %d (free-form) + %d (escalation)",
                i + 1, len(traces), n_judge_free_form, n_judge_escalation)

    # Flush remaining writes
    if not args.dry_run and pending_writes:
        _flush_batch(client, pending_writes)
        pending_writes.clear()

    # Persist locally
    if rows and not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_parquet(args.output, index=False)
            logger.info("Wrote %d rows to %s", len(rows), args.output)
        except ImportError:
            # Fallback to JSON if pandas/pyarrow not available.
            import json as _json
            json_path = args.output.with_suffix(".json")
            json_path.write_text(_json.dumps(rows, indent=2, default=str), encoding="utf-8")
            logger.info("pandas not available; wrote JSON to %s", json_path)

    # Final summary
    print("\n=== Scoring summary ===")
    print(f"Traces fetched : {len(traces)}")
    print(f"Scored         : {len(rows)}")
    print(f"Skipped (already scored, --overwrite to redo): {n_skipped_already_scored}")
    print(f"Skipped (missing prediction/ground_truth)    : {n_missing_data}")
    if args.only_overrides:
        print(f"Skipped (no answer_format override)          : {n_skipped_no_override}")
    if qa_format_index:
        print(f"answer_format overridden from --questions-dir: {n_format_overridden}")
    print(f"Judge calls    : {n_judge_free_form} (free-form) + {n_judge_escalation} (escalation)")

    if not args.no_judge and (n_judge_free_form + n_judge_escalation) > 0:
        n_calls = n_judge_free_form + n_judge_escalation
        cost_in = (n_calls * JUDGE_AVG_INPUT_TOKENS / 1000.0) * GPT5_MINI_PRICE_IN_PER_1K
        cost_out = (n_calls * JUDGE_AVG_OUTPUT_TOKENS / 1000.0) * GPT5_MINI_PRICE_OUT_PER_1K
        print(f"Estimated judge cost (gpt-5-mini): ${cost_in + cost_out:.4f}")

    print("\nProvenance distribution:")
    total = sum(provenance_counts.values()) or 1
    for prov, n in provenance_counts.most_common():
        print(f"  {prov:<13} {n:>5} ({100 * n / total:5.1f}%)")

    print("\nProvenance per model:")
    print(f"  {'model':<28} {'strict':>7} {'lenient':>7} {'judge':>7} {'unparseable':>12}")
    for model, counts in sorted(by_model_provenance.items()):
        print(
            f"  {model:<28} "
            f"{counts.get('strict', 0):>7} "
            f"{counts.get('lenient', 0):>7} "
            f"{counts.get('judge', 0):>7} "
            f"{counts.get('unparseable', 0):>12}"
        )


if __name__ == "__main__":
    sys.exit(main())
