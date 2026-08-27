"""Repair the counterfactual onset timestep quoted in FactoryBench Level 3 questions.

The L3 generator quoted the event onset on the raw episode clock while the
context time series it ships was shifted to its own base. The number in the
question text therefore pointed at a timestep that does not exist in the window
the model can see.

The repair rewrites only that number, to `first_context_timestamp + event_time_ms`
(`provenance.event_time_ms` is the onset measured from the start of the context
window, and is already correct in every released row). Context, options, answers
and acceptance bounds are untouched.

Usage:
    python fix_l3_timestep.py --repo FactoryBench/FactoryBench            # dry run
    python fix_l3_timestep.py --repo FactoryBench/FactoryBench --push
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path.cwd() / ".env")
load_dotenv()

SPLITS = ("train", "validation", "test")
REMOTE_TMPL = "factorybench_qa/level_3/{split}.jsonl"
TS_IN_CONTEXT = re.compile(r"t=(\d+):")
TS_IN_QUESTION = re.compile(r"(occurs at timestep )(\d+)( ms)")


def context_start_ms(context) -> int | None:
    stamps = [int(x) for x in TS_IN_CONTEXT.findall(json.dumps(context))]
    return min(stamps) if stamps else None


def context_end_ms(context) -> int | None:
    stamps = [int(x) for x in TS_IN_CONTEXT.findall(json.dumps(context))]
    return max(stamps) if stamps else None


def event_time_ms(item) -> int | None:
    prov = item.get("provenance")
    if isinstance(prov, str):
        prov = json.loads(prov)
    if not isinstance(prov, dict):
        return None
    value = prov.get("event_time_ms")
    return int(value) if value is not None else None


def repair_item(item) -> tuple[dict, str]:
    """Return (item, status). Status is one of ok / changed / skipped:<why>."""
    question = item.get("question", "")
    match = TS_IN_QUESTION.search(question)
    if not match:
        return item, "skipped:no-timestep-in-question"

    start = context_start_ms(item.get("context"))
    end = context_end_ms(item.get("context"))
    onset = event_time_ms(item)
    if start is None or onset is None:
        return item, "skipped:missing-anchor"

    correct = start + onset
    if not start <= correct <= end:
        return item, "skipped:repair-out-of-window"
    if int(match.group(2)) == correct:
        return item, "ok"

    item["question"] = TS_IN_QUESTION.sub(
        lambda m: f"{m.group(1)}{correct}{m.group(3)}", question, count=1
    )
    return item, "changed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="FactoryBench/FactoryBench")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--push", action="store_true", help="Commit the fixed splits back to the repo.")
    parser.add_argument(
        "--token-env",
        default=None,
        help="Env var holding the token. The the authors/* repos need HF_TOKEN; "
             "FactoryBench/FactoryBench needs HF_WRITE_TOKEN.",
    )
    args = parser.parse_args()

    if args.token_env:
        token = os.getenv(args.token_env)
        if not token:
            raise SystemExit(f"{args.token_env} is not set in the environment or .env")
    else:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise SystemExit("No Hugging Face token in the environment or .env")
    api = HfApi(token=token)

    args.workdir.mkdir(parents=True, exist_ok=True)
    totals = {"ok": 0, "changed": 0}
    staged = []

    for split in SPLITS:
        remote = REMOTE_TMPL.format(split=split)
        local = Path(
            hf_hub_download(
                args.repo, remote, repo_type="dataset", token=token,
                local_dir=args.workdir / "download",
            )
        )
        items = [json.loads(line) for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]

        counts: dict[str, int] = {}
        fixed = []
        for item in items:
            item, status = repair_item(item)
            counts[status] = counts.get(status, 0) + 1
            fixed.append(item)

        for key in ("ok", "changed"):
            totals[key] += counts.get(key, 0)
        print(f"{split}: n={len(items)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

        out = args.workdir / "fixed" / remote
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for item in fixed:
                handle.write(json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n")
        staged.append((out, remote))

    print(f"\ntotal: {totals['changed']} rewritten, {totals['ok']} already correct")

    if not args.push:
        print("dry run, nothing pushed. Re-run with --push to commit.")
        return

    for path, remote in staged:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="L3: quote the counterfactual onset on the same clock as the context window",
        )
        print(f"pushed {remote}")


if __name__ == "__main__":
    main()
