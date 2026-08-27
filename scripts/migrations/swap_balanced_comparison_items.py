"""Replace the two-robot comparison items with class-balanced regenerations.

L1.3 and L2.8 show two telemetry streams and ask four independent yes/no
questions: different robots (A), different anomalous states (B), different
tasks (C), same task at a different phase (D). Every one of the eight
marginals was lopsided in the release:

    L1.3   P(A)=0.503  P(B)=0.000  P(C)=0.689  P(D)=0.298
    L2.8   P(A)=0.300  P(B)=0.953  P(C)=0.617  P(D)=0.294

B was degenerate in both directions for the same reason. L1 drew both streams
from the nominal pool, so their anomalous states could never differ and the
proposition carried no information at all across 2,407 items. L2 always made
the first stream faulty and the second one usually a different fault, so it was
true almost always. The rest follows the shape of the corpus: 7,553 UR3
episodes against 1,302 KUKA, and more cross-task than same-task pairs.

The consequence is that a model that ignored the time series entirely and
answered the per-position majority scored 0.723 on L1.3 and 0.744 on L2.8
against a chance rate of 0.500, and one fixed four-letter string scored 35.5%
and 32.8% against 6.25%.

These items cannot be repaired in place the way the option-support items were.
The skew is a property of which two episodes get compared, so fixing it means
picking different episodes, which means new context, new window and a new
answer: a different item. So this script swaps rather than edits. Each
replacement lands at the same position in the same split, so per-level counts
and the split proportions the paper reports are unchanged, but the ids are new
and any score previously reported on an L1.3 or L2.8 item no longer applies.

Replacements come from the generators themselves, run with ``--template-ids``,
so the ground truth is produced by exactly the path that produces every other
item. The balancing lives in ``utils.pair_balance``: items are aimed at one of
eight cells of (robot, anomaly, task) in turn, no two items compare the same
two episodes, and for L2 both streams stay faulty so "different anomalous
states" means two different faults rather than healthy against faulty.

Usage:
    python scripts/swap_balanced_comparison_items.py \
        --l1-pool <dir> --l2-pool <dir> --workdir <dir>
    ... --push
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from huggingface_hub import HfApi, hf_hub_download

REPO = "FactoryBench/FactoryBench"
SPLITS = ("train", "validation", "test")
# (level, template_id) -> which pool supplies its replacements
TARGETS = {(1, 3): "l1", (2, 8): "l2"}
# Only items whose two streams both come from one of these sources can be
# swapped. aursad and vorausad episodes are not on this machine (vorausad is
# gone entirely, aursad exists only as un-normalized parquet), so items that
# touch them cannot be regenerated and are left exactly as released. That caps
# what the swap can achieve: it reaches 7.6% of L1.3 and 59.5% of L2.8, and the
# remaining skew has to come out by downsampling the untouched items later.
SWAPPABLE_SOURCES = frozenset({"factorywave", "factorywave_kuka"})


def load_pool(pool_dir: Path, template_id: int) -> List[Dict[str, Any]]:
    """Generated items for one template, in filename order."""
    out = []
    for path in sorted(pool_dir.glob("level*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if item.get("template_id") == template_id:
            out.append(item)
    return out


def marginals(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """P(A..D), the majority-class score and the best fixed-string score."""
    answers = [it.get("answer", "") for it in items]
    answers = [a for a in answers if isinstance(a, str) and len(a) == 4]
    n = len(answers)
    if not n:
        return {}
    stats: Dict[str, float] = {}
    majority = 0.0
    for i, label in enumerate("ABCD"):
        true_count = sum(a[i] == "T" for a in answers)
        stats[f"P({label})"] = true_count / n
        majority += max(true_count, n - true_count) / n
    stats["majority_class"] = majority / 4
    stats["top_string"] = collections.Counter(answers).most_common(1)[0][1] / n
    stats["n"] = n
    return stats


def show(tag: str, stats: Dict[str, float]) -> None:
    if not stats:
        print(f"  {tag}: no scorable items")
        return
    print(f"  {tag}: n={int(stats['n'])}  "
          + "  ".join(f"{k}={stats[k]:.3f}" for k in ("P(A)", "P(B)", "P(C)", "P(D)"))
          + f"  majority={stats['majority_class']:.3f}"
          f"  top-string={stats['top_string']:.1%}")


def is_swappable(item: Dict[str, Any]) -> bool:
    """True when both streams come from a source we can still regenerate."""
    prov = item.get("provenance") or {}
    return {prov.get("dataset_a"), prov.get("dataset_b")} <= SWAPPABLE_SOURCES


def pair_key(item: Dict[str, Any]) -> Tuple[Any, Any]:
    prov = item.get("provenance") or {}
    return tuple(sorted((prov.get("episode_a"), prov.get("episode_b")), key=str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1-pool", type=Path, required=True,
                    help="Output dir of level1 --template-ids 3")
    ap.add_argument("--l2-pool", type=Path, required=True,
                    help="Output dir of level2 --template-ids 8")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token-env", default="HF_WRITE_TOKEN")
    args = ap.parse_args()

    token = os.getenv(args.token_env)
    if not token and args.push:
        raise SystemExit(f"{args.token_env} not set")
    api = HfApi(token=token) if token else None
    args.workdir.mkdir(parents=True, exist_ok=True)

    pools = {"l1": load_pool(args.l1_pool, 3), "l2": load_pool(args.l2_pool, 8)}
    print("replacement pools:")
    for name, tid in (("l1", 3), ("l2", 8)):
        print(f"  {name} (template {tid}): {len(pools[name])} items")
        show("    balance", marginals(pools[name]))

    # A pool that runs dry mid-file would leave the split half swapped, so
    # check supply against demand before writing anything.
    cursors = {"l1": 0, "l2": 0}
    demand: collections.Counter = collections.Counter()
    downloads: Dict[Tuple[int, str], Path] = {}
    for (lvl, tid), pool_name in TARGETS.items():
        for split in SPLITS:
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            local = Path(hf_hub_download(REPO, remote, repo_type="dataset",
                                         token=token,
                                         local_dir=args.workdir / "download"))
            downloads[(lvl, split)] = local
            for line in local.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("template_id") == tid and is_swappable(item):
                    demand[pool_name] += 1
    print("\ndemand vs supply:")
    short = False
    for name in ("l1", "l2"):
        print(f"  {name}: need {demand[name]}, have {len(pools[name])}")
        if len(pools[name]) < demand[name]:
            short = True
    if short:
        raise SystemExit("replacement pool too small; generate more before swapping")

    before: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    kept: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    fresh: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    after: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    staged: List[Tuple[Path, str]] = []
    swapped = 0
    for (lvl, tid), pool_name in TARGETS.items():
        for split in SPLITS:
            remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
            rows = []
            for line in downloads[(lvl, split)].read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("template_id") == tid and item.get("level") == lvl:
                    if is_swappable(item):
                        before[pool_name].append(item)
                        item = pools[pool_name][cursors[pool_name]]
                        cursors[pool_name] += 1
                        swapped += 1
                        fresh[pool_name].append(item)
                    else:
                        kept[pool_name].append(item)
                    after[pool_name].append(item)
                rows.append(item)
            out = args.workdir / "swapped" / remote
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
            staged.append((out, remote))
            print(f"  L{lvl}/{split}: {len(rows)} rows, "
                  f"{sum(1 for r in rows if r.get('template_id') == tid)} swapped")

    print(f"\nswapped {swapped} items")
    for name, label in (("l1", "L1.3"), ("l2", "L2.8")):
        print(f"{label}:  swapped={len(before[name])}  left as released={len(kept[name])}")
        show("swapped items, as released", marginals(before[name]))
        show("swapped items, regenerated", marginals(fresh[name]))
        show("untouched, cannot regen   ", marginals(kept[name]))
        show("template overall, after   ", marginals(after[name]))
        keys = [pair_key(it) for it in after[name]]
        print(f"  distinct episode pairs: {len(set(keys))}/{len(keys)}")
        if name == "l2":
            faults = [(it.get("provenance") or {}).get("fault_label_a") for it in after[name]]
            faults += [(it.get("provenance") or {}).get("fault_label_b") for it in after[name]]
            print(f"  healthy streams: {sum(1 for f in faults if not f)}")

    if not args.push:
        print("\ndry run, nothing pushed. Re-run with --push.")
        return 0
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message="L1.3/L2.8: swap in class-balanced comparison items")
        print(f"pushed {remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
