"""Restrict released ranking option segments to the context's channel set.

Templates L2.1 and L3.1 encoded their option segments from the full episode row
while the context was filtered to the template's ``important_features``. Every
one of the 1,334 released ranking items therefore showed a context defined over
~19 channels and asked the model to order segments carrying ~98, of which 79
appeared in no acronym mapping anywhere in the item. The segments and the
context have to describe the same signals for the question to be answerable.

The generator now filters the ranking pool through
``filter_rows_to_template_features``. This script repairs the released items by
dropping, from each option, every ``key=value`` token whose acronym is not in
that item's own context mapping. Row structure (the ``|`` separators), option
labels, answers and the context are untouched.

Guard: if filtering would make two options identical the item becomes
unanswerable, so it is left alone and reported.

Usage:
    python scripts/align_ranking_option_features.py --workdir <dir>
    python scripts/align_ranking_option_features.py --workdir <dir> --push
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


def filter_option(value: str, allowed: set) -> str:
    """Keep only tokens whose acronym is in ``allowed``, preserving row layout."""
    out_rows = []
    for part in str(value).split(ROW_SEP):
        toks = [t for t in part.split(", ")
                if "=" in t and t.partition("=")[0].strip() in allowed]
        out_rows.append(", ".join(toks))
    return ROW_SEP.join(out_rows)


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
    seen = changed = skipped_dup = no_map = 0
    before_keys, after_keys = [], []
    for lvl in LEVELS:
        for split in SPLITS:
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            local = Path(hf_hub_download(REPO, remote, repo_type="dataset", token=token,
                                         local_dir=args.workdir / "download"))
            rows_out = []
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if RANK_Q.search(item.get("question", "")):
                    seen += 1
                    ctx = item.get("context") or {}
                    mapping = (ctx.get("time_series_format") or {}).get("acronym_mapping") or {}
                    allowed = set(mapping)
                    opts = item.get("options") or {}
                    if not allowed or not opts:
                        no_map += 1
                    else:
                        first = str(next(iter(opts.values()))).split(ROW_SEP)[0]
                        before_keys.append(len([t for t in first.split(", ") if "=" in t]))
                        new = {k: filter_option(v, allowed) for k, v in opts.items()}
                        if len(set(new.values())) != len(new):
                            skipped_dup += 1
                        else:
                            item["options"] = new
                            changed += 1
                            f2 = str(next(iter(new.values()))).split(ROW_SEP)[0]
                            after_keys.append(len([t for t in f2.split(", ") if "=" in t]))
                rows_out.append(item)
            out = args.workdir / "fixed" / remote
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for r in rows_out:
                    fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
            staged.append((out, remote))
            print(f"  L{lvl}/{split}: {len(rows_out)} rows")

    def med(x):
        return sorted(x)[len(x) // 2] if x else 0
    print(f"\nranking items seen={seen} rewritten={changed} "
          f"skipped_would_duplicate={skipped_dup} no_mapping={no_map}")
    print(f"channels per option timestep: before median={med(before_keys)} -> after median={med(after_keys)}")

    if not args.push:
        print("dry run, nothing pushed. Re-run with --push.")
        return
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message="L2.1/L3.1: align option segment channels with the context mapping")
        print(f"pushed {remote}")


if __name__ == "__main__":
    main()
