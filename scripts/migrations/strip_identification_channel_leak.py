"""Strip the robot-fingerprint channels from the released identification items.

Templates L1.6 and L2.10 ("What robot does this sensor data originate from?")
request 30 channels, but each source dataset only carries a subset, so the
surviving channel-name set identified the robot outright: 4 distinct signatures
across 11,979 items, each mapping to exactly one robot, and a lookup fitted on
train predicts test at 100% without reading a single value.

The fix keeps only the channels every dataset renders 100% of the time
(feedback_pos_0..5 and setpoint_pos_0..5), removing them from the acronym
mapping, from every encoded row, and from any constant-features note. The three
dropped groups are each near-perfect discriminators on their own:

    feedback_speed_*       absent for KUKA, present elsewhere
    est_contact_force_*    present for aursad, essentially nowhere else
    effort_target_torque_* renders in only 8% of factorywave items

Ground truth is untouched: the robot is still the robot, and the three machines
stay separable from joint position alone on physical grounds (Yu 5 logs radians,
UR3e joint 5 spans ~480 deg, KUKA sits between), which is the evidence the
template is meant to test.

Usage:
    python scripts/strip_identification_channel_leak.py --workdir <dir>
    python scripts/strip_identification_channel_leak.py --workdir <dir> --push
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
TARGETS = {(1, 6), (2, 10)}
KEEP = {f"feedback_pos_{i}" for i in range(6)} | {f"setpoint_pos_{i}" for i in range(6)}
KEEP_ALWAYS = {"timestamp_ms"}
IDENT = re.compile(r"which robot|what robot|which machine|what machine", re.I)


def strip_context(ctx):
    """Drop non-KEEP channels from a context. Returns (ctx, changed)."""
    if not isinstance(ctx, dict):
        return ctx, False
    changed = False
    holders = [ctx] + [v for v in ctx.values() if isinstance(v, dict) and "time_series" in v]
    for h in holders:
        tf = h.get("time_series_format")
        if not isinstance(tf, dict):
            continue
        mapping = tf.get("acronym_mapping")
        if not isinstance(mapping, dict):
            continue
        drop_acr = {a for a, c in mapping.items() if c not in KEEP and c not in KEEP_ALWAYS}
        if not drop_acr:
            continue
        tf["acronym_mapping"] = {a: c for a, c in mapping.items() if a not in drop_acr}
        rows = h.get("time_series")
        if isinstance(rows, list):
            new_rows = []
            for row in rows:
                s = str(row)
                # rows look like "t=0: fp0=1.2, fs0=0.3, ..."
                head, sep, body = s.partition(": ")
                if not sep:
                    new_rows.append(row)
                    continue
                kept = [tok for tok in body.split(", ")
                        if tok.split("=", 1)[0].strip() not in drop_acr]
                new_rows.append(f"{head}{sep}{', '.join(kept)}")
            h["time_series"] = new_rows
        notes = h.get("notes")
        if isinstance(notes, dict) and isinstance(notes.get("constant_features"), dict):
            cf = {c: v for c, v in notes["constant_features"].items()
                  if c in KEEP or c in KEEP_ALWAYS}
            if cf:
                notes["constant_features"] = cf
            else:
                notes.pop("constant_features", None)
                if not notes:
                    h.pop("notes", None)
        changed = True
    return ctx, changed


def is_target(item) -> bool:
    if (item.get("level"), item.get("template_id")) in TARGETS:
        return True
    # belt and braces: template ids could drift, so also match on the question
    return bool(IDENT.search(item.get("question", "")))


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

    staged, totals = [], {"scanned": 0, "targeted": 0, "changed": 0}
    for lvl in (1, 2):
        for split in SPLITS:
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            local = Path(hf_hub_download(REPO, remote, repo_type="dataset", token=token,
                                         local_dir=args.workdir / "download"))
            out_rows = []
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                totals["scanned"] += 1
                if is_target(item):
                    totals["targeted"] += 1
                    ctx, changed = strip_context(item.get("context"))
                    item["context"] = ctx
                    if changed:
                        totals["changed"] += 1
                out_rows.append(item)
            out = args.workdir / "fixed" / remote
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for r in out_rows:
                    fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
            staged.append((out, remote))
            print(f"  L{lvl}/{split}: {len(out_rows)} rows")

    print(f"\nscanned={totals['scanned']} identification={totals['targeted']} "
          f"contexts_stripped={totals['changed']}")

    if not args.push:
        print("dry run, nothing pushed. Re-run with --push.")
        return
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message="L1.6/L2.10: drop robot-fingerprinting channels from context")
        print(f"pushed {remote}")


if __name__ == "__main__":
    main()
