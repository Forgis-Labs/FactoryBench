"""Enforce "segments strictly after the context" on the released ranking items.

Templates L2.1 and L3.1 show a context window and ask the model to rank four
signal segments in the order they appear as the event manifests. The segments
were sampled from ``post_event_rows``, which is not disjoint from the context:
the window is centred on the event onset and extends past it, and at L3 the
baseline and counterfactual episodes are identical until they diverge. Measured
on the release, 696 of 5,336 option segments (13%) appear verbatim inside their
own context, letting the ordering be recovered by matching values against the
stream instead of reasoning about propagation. 304 of 1,334 items are affected.

The generator now samples only from rows strictly later than the last context
timestamp (``rows_strictly_after_context``). This script repairs the already
released items the only way possible without the source episodes: it truncates
each affected context to end immediately before the earliest row that any of
its own segments reproduces. Options, answers and mappings are untouched, so
ground truth is unchanged and item counts stay stable.

Constant segments are ignored when locating the overlap: a stationary robot
produces identical rows and matching one is coincidence, not leakage.

Usage:
    python scripts/trim_ranking_context_overlap.py --workdir <dir>
    python scripts/trim_ranking_context_overlap.py --workdir <dir> --push
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv(Path.cwd() / ".env")
load_dotenv()

REPO = "FactoryBench/FactoryBench"
SPLITS = ("train", "validation", "test")
LEVELS = (2, 3)
RANK_Q = re.compile(r"rank the signal segments", re.I)
ROW_SEP = " | "
MIN_ROWS = 5  # never truncate a context below this


def _kv(part: str) -> dict:
    return {k: v for k, _, v in (t.partition("=") for t in part.split(", ") if "=" in t)}


def option_rows(value) -> list:
    return [_kv(p) for p in str(value).split(ROW_SEP)]


def context_rows(ctx) -> list:
    return [_kv(str(r).partition(": ")[2]) for r in (ctx.get("time_series") or [])]


def earliest_overlap(item) -> int | None:
    """Index of the first context row reproduced by any option segment."""
    ctx = item.get("context")
    if not isinstance(ctx, dict):
        return None
    rows = context_rows(ctx)
    if not rows:
        return None
    acr = set((ctx.get("time_series_format") or {}).get("acronym_mapping") or {}) - {"tm"}
    ctx_sig_cache: dict[tuple, list] = {}
    best = None
    for value in (item.get("options") or {}).values():
        seg = option_rows(value)
        if not seg:
            continue
        shared = tuple(sorted(set(seg[0]) & acr))
        if not shared:
            continue
        sig = [tuple(r.get(k) for k in shared) for r in seg]
        if len(set(sig)) == 1:
            continue  # constant segment: a match would be coincidence
        if shared not in ctx_sig_cache:
            ctx_sig_cache[shared] = [tuple(r.get(k) for k in shared) for r in rows]
        csig = ctx_sig_cache[shared]
        for i in range(max(0, len(csig) - len(sig) + 1)):
            if csig[i:i + len(sig)] == sig:
                best = i if best is None else min(best, i)
                break
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token-env", default="HF_WRITE_TOKEN")
    args = ap.parse_args()

    token = os.getenv(args.token_env)
    if not token:
        raise SystemExit(f"{args.token_env} not set")
    api = HfApi(token=token)
    args.workdir.mkdir(parents=True, exist_ok=True)

    staged = []
    seen = trimmed = floored = 0
    kept_lengths = []
    for lvl in LEVELS:
        for split in SPLITS:
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            local = Path(hf_hub_download(REPO, remote, repo_type="dataset", token=token,
                                         local_dir=args.workdir / "download"))
            out_rows = []
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if RANK_Q.search(item.get("question", "")):
                    seen += 1
                    cut = earliest_overlap(item)
                    if cut is not None:
                        if cut < MIN_ROWS:
                            cut = MIN_ROWS
                            floored += 1
                        ts = item["context"].get("time_series") or []
                        if cut < len(ts):
                            item["context"]["time_series"] = ts[:cut]
                            trimmed += 1
                            kept_lengths.append(cut)
                out_rows.append(item)
            out = args.workdir / "fixed" / remote
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
            staged.append((out, remote))
            print(f"  L{lvl}/{split}: {len(out_rows)} rows")

    kept_lengths.sort()
    print(f"\nranking items seen={seen}  trimmed={trimmed}  floored_at_{MIN_ROWS}={floored}")
    if kept_lengths:
        mid = kept_lengths[len(kept_lengths) // 2]
        print(f"context rows kept in trimmed items: min={kept_lengths[0]} median={mid} max={kept_lengths[-1]}")

    if not args.push:
        print("dry run, nothing pushed. Re-run with --push.")
        return
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message="L2.1/L3.1: trim context so ranking segments come strictly after it")
        print(f"pushed {remote}")


if __name__ == "__main__":
    main()
