"""Build FactoryBench-Lite: a balanced 3,000-item subsample of the full set.

Lite is not a random draw. The point is that every axis a solver could exploit
is flat within it, so a score on Lite means the same thing a score on the full
set would mean, without the 70,915-item cost.

Balancing happens twice:

* across templates, so no template dominates the headline number;
* inside each template, on whatever dimension actually defines its answer.

That second one is the substance. A template's answer distribution can be
balanced overall and still be lopsided in the ways that matter: the anomaly
templates can be even across letters while over-representing one fault class,
and the level 4 optimization template can be even across letters while asking
about payload mass nine times for every TCP frame. So each template declares a
stratum key, and selection round-robins across strata rather than sampling
uniformly.

Strata by family:

  multi-select T/F   the four-letter answer string, which balances the
                     per-position T/F rates as a side effect
  anomaly ID         the correct anomaly class
  robot ID           the correct robot
  L4 diagnosis       the correct root-cause category
  L4 optimization    the misconfigured parameter and its variant
  ranking            the answer permutation
  numeric            (fault, task) of the source episode

Where a stratum is genuinely scarce (fault 22 in the optimization set, KUKA at
~14% of the corpus) exact equality is not reachable. Round-robin takes what
exists rather than over-sampling a thin class, and the achieved distribution is
reported per template so the residual imbalance is visible rather than implied.

Usage:
    python scripts/build_factorybench_lite.py --size 3000
    python scripts/build_factorybench_lite.py --size 3000 --push
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download

REPO = "FactoryBench/FactoryBench"
LEVELS = (1, 2, 3, 4)

MULTISELECT = {(1, 3), (2, 2), (2, 3), (2, 8), (3, 2), (3, 3)}
RANKING = {(2, 1), (2, 9), (3, 1)}
ANOMALY_ID = {(2, 7)}
ROBOT_ID = {(1, 6), (2, 10)}
L4_DIAGNOSIS = {(4, 1)}
L4_OPTIMIZATION = {(4, 2)}


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def stratum_of(item: Dict[str, Any]) -> str:
    """The dimension this template must be balanced on."""
    key = (item.get("level"), item.get("template_id"))
    answer = item.get("answer")
    options = item.get("options") or {}
    prov = item.get("provenance") or {}

    if key in MULTISELECT or key in RANKING:
        return str(answer)
    if key in ANOMALY_ID or key in ROBOT_ID:
        # the option text, not the letter: letters are shuffled per item, so
        # balancing on them would leave the underlying classes lopsided
        return str(options.get(str(answer), answer))[:120]
    if key in L4_DIAGNOSIS:
        # L4 is free-form: there are no options and the answer is the protocol
        # text itself, so balance on the root cause that protocol belongs to.
        return str(item.get("root_cause") or answer)[:120]
    if key in L4_OPTIMIZATION:
        return str(answer)[:120]
    # numeric and window templates: spread over the source condition. The
    # extrapolation templates carry no fault_label in provenance, so keying on
    # it alone collapsed all their items into one stratum and balanced nothing;
    # the queried signal is the axis that actually varies there.
    bounds = item.get("acceptance_bounds") or {}
    signal = bounds.get("signal")
    if signal:
        return str(signal)
    return f"{prov.get('fault_label')}|{prov.get('task')}"


def balance_positions(items: List[Dict[str, Any]], quota: int,
                      rng: random.Random) -> List[Dict[str, Any]]:
    """Pick multi-select items so each of the four positions sits at 50% True.

    Equal counts per answer string is not the same thing. The reachable strings
    do not carry the four propositions evenly: proposition D is deliberately
    rarer than the others in the full set, so drawing equally from each string
    still left D near 0.35. Lite exists to flatten exactly that, so selection
    is greedy on the position totals instead: take the item that most reduces
    the largest deviation from a 50/50 column.
    """
    pool = list(items)
    rng.shuffle(pool)
    chosen: List[Dict[str, Any]] = []
    trues = [0, 0, 0, 0]
    while pool and len(chosen) < quota:
        target = (len(chosen) + 1) / 2.0
        best, best_cost = None, None
        # Full scan, not a window. A scarce proposition can be concentrated in
        # a handful of items: L1.3 holds 94 items with B true out of 2,407,
        # because 92% of that template is aursad/vorausad which cannot be
        # regenerated locally and where B is false throughout. A bounded scan
        # kept missing them and left B at 0.13.
        for idx, item in enumerate(pool):
            answer = str(item.get("answer") or "")
            if len(answer) != 4:
                continue
            cost = sum(abs(trues[i] + (answer[i] == "T") - target) for i in range(4))
            if best_cost is None or cost < best_cost:
                best, best_cost = idx, cost
        if best is None:
            break
        item = pool.pop(best)
        answer = str(item["answer"])
        for i in range(4):
            trues[i] += (answer[i] == "T")
        chosen.append(item)
    return chosen


def round_robin(strata: Dict[str, List[Any]], quota: int, rng: random.Random) -> List[Any]:
    """Take one item per stratum in turn until the quota is met.

    Exhausted strata drop out, so a thin class contributes everything it has
    and no more, and the surplus goes to classes that can supply it.
    """
    pools = {k: list(v) for k, v in strata.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    chosen: List[Any] = []
    while len(chosen) < quota and pools:
        for key in list(pools):
            if len(chosen) >= quota:
                break
            if pools[key]:
                chosen.append(pools[key].pop())
            if not pools[key]:
                del pools[key]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--workdir", type=Path, default=Path("lite_build"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--token-env", default="HF_WRITE_TOKEN")
    ap.add_argument("--env-file", type=Path, default=Path(".env"))
    args = ap.parse_args()

    load_env(args.env_file)
    token = os.getenv(args.token_env)
    if args.push and not token:
        raise SystemExit(f"{args.token_env} not found")
    rng = random.Random(args.seed)
    args.workdir.mkdir(parents=True, exist_ok=True)

    # The release is one file per level; there is no split to preserve.
    by_template: Dict[Tuple[int, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    total = 0
    for level in LEVELS:
        path = hf_hub_download(REPO, f"factorybench_qa/level_{level}.jsonl",
                               repo_type="dataset", token=token)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            by_template[(level, item.get("template_id"))].append(item)
            total += 1
    print(f"full set: {total} items across {len(by_template)} templates")

    quota = args.size // len(by_template)
    remainder = args.size - quota * len(by_template)
    print(f"target {args.size}: {quota} per template, {remainder} spread over the largest\n")

    selected: List[Dict[str, Any]] = []
    report: List[Tuple[str, int, int, float]] = []
    order = sorted(by_template, key=lambda k: -len(by_template[k]))
    for i, key in enumerate(sorted(by_template)):
        want = quota + (1 if order.index(key) < remainder else 0)
        if key in MULTISELECT:
            picked = balance_positions(by_template[key], want, rng)
        else:
            strata: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
            for item in by_template[key]:
                strata[stratum_of(item)].append(item)
            picked = round_robin(strata, want, rng)
        selected.extend(picked)
        counts = collections.Counter(stratum_of(x) for x in picked)
        spread = (min(counts.values()) / max(counts.values())) if counts else 0.0
        report.append((f"L{key[0]}.{key[1]}", len(picked), len(strata), spread))

    print(f"{'template':10s} {'picked':>7} {'strata':>7} {'min/max per stratum':>21}")
    for name, n, k, spread in report:
        print(f"{name:10s} {n:7d} {k:7d} {spread:20.2f}")
    print(f"\nselected {len(selected)} items")

    # write one file per level, mirroring the flat release layout
    out_root = args.workdir / "factorybench_lite"
    staged: List[Tuple[Path, str]] = []
    for level in LEVELS:
        rows = [x for x in selected if x.get("level") == level]
        out = out_root / f"level_{level}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                row = dict(row)
                row["lite"] = True
                fh.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        staged.append((out, f"factorybench_lite/level_{level}.jsonl"))
        print(f"  L{level} {len(rows):5d}")

    ids = sorted(str(x.get("id")) for x in selected)
    (args.workdir / "lite_ids.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")

    if not args.push:
        print("\ndry run, nothing pushed. Re-run with --push.")
        return 0
    api = HfApi(token=token)
    for path, remote in staged:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"FactoryBench-Lite: balanced {len(selected)}-item subsample")
        print(f"pushed {remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
