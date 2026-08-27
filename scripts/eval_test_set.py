"""Strict-batch evaluation of the HF test split across all models and levels.

Pipeline:

  1. **Pull**: download ``factorynet_qa_<folder>/level_<N>/test.jsonl`` from HF.
  2. **Unpack**: write each item as ``<out>/questions/level<N>/<id>.json`` so the
     existing prompt builder (which expects a directory of JSONs) works as-is.
  3. **Build prompts**: ``src.question_generation.build_prompts_from_questions``
     emits one prompt JSON per item under ``<out>/prompts/level<N>/``.
  4. **Chunked batch eval**: per (level, model) cell, the orchestrator splits
     the prompts into chunks of ``--chunk-size`` (default 1000) and spawns one
     subprocess per chunk against the right backend (``run_foundry_eval`` for
     foundry, ``run_aws_eval`` for bedrock). Each chunk is one batch job.
     Tail chunks smaller than ``--sync-threshold`` (default 100) fall back to
     sync, because Bedrock's ``CreateModelInvocationJob`` rejects batches
     under 100 records. ``supports_batch=False`` models (e.g. OpenRouter) run
     every chunk sync regardless.
  5. **Replies**: ``<out>/replies/level<N>/<model_slug>/*_answer.json`` plus a
     ``_summary.json`` per (level, model).

Logging:
  * Orchestrator prints clean timestamped status lines to ``<out>/run.log``, just phase transitions, per-cell start/end, errors. **Tail this for an
    at-a-glance view of the whole job.**
  * Each (level, model) subprocess's verbose output (HTTP traffic, batch
    polling, TQDM bars) goes to ``<out>/logs/level<N>_<model_slug>.err.log``
    so you can drill into one cell without drowning in 16-way interleaving.

Usage::

    python -m scripts.eval_test_set \\
        --hf-dataset-folder factorynet_qa_150k \\
        --levels 1 2 3 4 \\
        --models gpt-5.1-1 claude-sonnet-4.6 mistral-large-3 deepseek-v3.2 \\
        --output-root output/test_eval

Pass ``--no-judge`` to skip the LLM-as-judge layer (free-form items get
``score=None`` instead of being judged).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download

from src.config import MODEL_NAMES, MODELS, get_provider
from src.pipeline.upload_qa_pairs import DEFAULT_REPO_ID, resolve_hf_token

logger = logging.getLogger(__name__)

_STATUS_LOCK = threading.Lock()
_STATUS_FILE: Optional[Path] = None


def _status(msg: str) -> None:
    """Timestamped one-line status to stdout + run.log (thread-safe)."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    with _STATUS_LOCK:
        print(line, flush=True)
        if _STATUS_FILE is not None:
            with _STATUS_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _model_slug(model: str) -> str:
    return model.replace(".", "_").replace("/", "_")


def _download_test_jsonl(
    repo_id: str, dataset_folder: str, level: int
) -> Optional[Path]:
    remote = f"{dataset_folder}/level_{level}/test.jsonl"
    api = HfApi(token=resolve_hf_token())
    try:
        api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        logger.error(f"L{level}: cannot list repo {repo_id}: {exc}")
        return None
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename=remote, repo_type="dataset"
        )
        return Path(local)
    except Exception as exc:
        logger.error(f"L{level}: download failed for {remote}: {exc}")
        return None


def _unpack_jsonl_to_json_dir(jsonl: Path, out_dir: Path) -> int:
    """Write each JSONL line as a separate JSON file (one QA per file).
    Returns the number of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            qid = item.get("id") or f"item_{n:06d}"
            # Sanitize for filename safety; ids are uuids in practice.
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(qid))
            out_path = out_dir / f"level{item.get('level','X')}_{safe}.json"
            with out_path.open("w", encoding="utf-8") as out:
                json.dump(item, out, indent=2)
            n += 1
    return n


def _build_prompts(level_q_dir: Path, level_p_dir: Path, kg_path: str) -> None:
    cmd = [
        sys.executable, "-m", "src.question_generation.build_prompts_from_questions",
        "--input", str(level_q_dir),
        "--output", str(level_p_dir),
        "--machines", kg_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _eval_one(
    *,
    level: int,
    model: str,
    questions_dir: Path,
    prompts_dir: Path,
    replies_dir: Path,
    logs_dir: Path,
    no_judge: bool,
    cost_limit: Optional[float],
    max_output_tokens: Optional[int],
    poll_interval: int,
    chunk_size: int,
    sync_threshold: int,
    sync_concurrency: int = 1) -> Dict[str, Any]:
    """Run one (level, model) cell, broken into chunks of size ``chunk_size``.

    The eval module's ``--batch-number`` / ``--batch-size`` slice the prompt
    list, so we invoke the subprocess once per chunk to spread a level across
    multiple smaller batch jobs (rather than one giant batch). Tail chunks
    smaller than ``sync_threshold`` are run with ``--no-batch`` because
    Bedrock's ``CreateModelInvocationJob`` rejects batches under 100 records.
    """
    provider = get_provider(model)
    if provider == "foundry":
        eval_module = "src.evaluation.run_foundry_eval"
    elif provider in ("bedrock", "sagemaker"):
        eval_module = "src.evaluation.run_aws_eval"
    else:
        raise SystemExit(f"unknown provider for model {model!r}: {provider}")

    supports_batch = bool(MODELS.get(model, {}).get("supports_batch", False))
    n_items = len(list(prompts_dir.glob("*.json")))
    n_chunks = max(1, (n_items + chunk_size - 1) // chunk_size)

    cell_log = logs_dir / f"level{level}_{_model_slug(model)}.err.log"
    cell_log.write_text(
        f"# {datetime.now().isoformat()}  cell L{level} {model}: "
        f"{n_items} items, {n_chunks} chunk(s) of up to {chunk_size}\n",
        encoding="utf-8")
    _status(
        f"  ▶ L{level} {model}: {n_items} items in {n_chunks} chunk(s) "
        f"({eval_module.split('.')[-1]})  log={cell_log.name}"
    )

    agg = {"completed": 0, "failed": 0, "skipped": 0}
    cell_t0 = time.time()
    cell_n_errors = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_items)
        chunk_n = chunk_end - chunk_start

        # Bedrock's CreateModelInvocationJob requires ≥100 records. Tail
        # chunks below that threshold fall back to sync. For non-batch
        # providers, all chunks are sync regardless.
        force_sync_tail = supports_batch and chunk_n < sync_threshold
        chunk_uses_batch = supports_batch and not force_sync_tail
        if not supports_batch:
            mode = "sync"
        elif force_sync_tail:
            mode = f"sync(tail<{sync_threshold})"
        else:
            mode = "batch"

        chunk_summary_path = replies_dir / f"_summary_chunk{chunk_idx}.json"
        cmd = [
            sys.executable, "-u", "-m", eval_module,
            "--input", str(prompts_dir),
            "--output-dir", str(replies_dir),
            "--questions", str(questions_dir),
            "--model", model,
            "--summary-file", str(chunk_summary_path),
            "--poll-interval", str(poll_interval),
            "--batch-number", str(chunk_idx),
            "--batch-size", str(chunk_size),
        ]
        if chunk_uses_batch:
            cmd.append("--strict-batch")
        elif supports_batch:
            cmd.append("--no-batch")  # explicit sync for the small tail
        if eval_module.endswith("run_foundry_eval"):
            cmd.extend(["--eval-level", f"level_{level}_test"])
        if no_judge:
            cmd.append("--no-judge")
        if cost_limit is not None:
            cmd.extend(["--cost-limit", str(cost_limit)])
        if max_output_tokens is not None:
            cmd.extend(["--max-output-tokens", str(max_output_tokens)])
        if not chunk_uses_batch and sync_concurrency > 1:
            cmd.extend(["--concurrency", str(sync_concurrency)])

        _status(
            f"     L{level} {model} chunk {chunk_idx + 1}/{n_chunks} "
            f"[{chunk_start}:{chunk_end}] ({chunk_n} items, {mode})"
        )
        t0 = time.time()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        with cell_log.open("a", encoding="utf-8") as logfh:
            logfh.write(
                f"\n# === chunk {chunk_idx + 1}/{n_chunks} "
                f"[{chunk_start}:{chunk_end}] ({chunk_n} items, {mode}) ===\n"
            )
            logfh.flush()
            proc = subprocess.run(cmd, stdout=logfh, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t0

        if proc.returncode != 0:
            cell_n_errors += 1
            _status(
                f"     ✗ L{level} {model} chunk {chunk_idx + 1}/{n_chunks}: "
                f"exit {proc.returncode} in {dt:.0f}s"
            )
            continue

        if chunk_summary_path.exists():
            try:
                cs = json.loads(chunk_summary_path.read_text(encoding="utf-8"))
                for k in ("completed", "failed", "skipped"):
                    agg[k] += int(cs.get(k, 0) or 0)
            except Exception:
                pass
        _status(
            f"     ✓ L{level} {model} chunk {chunk_idx + 1}/{n_chunks} "
            f"done in {dt:.0f}s"
        )

    cell_dt = time.time() - cell_t0

    summary_path = replies_dir / "_summary.json"
    summary: Dict[str, Any] = {
        "level": level, "model": model,
        "n_items": n_items, "n_chunks": n_chunks, "n_errors": cell_n_errors,
        "wall_clock_s": round(cell_dt, 1),
        **agg,
    }
    if cell_n_errors > 0:
        summary["error"] = f"{cell_n_errors}/{n_chunks} chunks failed"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if cell_n_errors > 0:
        _status(
            f"  ✗ L{level} {model}: {cell_n_errors}/{n_chunks} chunks failed | "
            f"ok={agg['completed']} fail={agg['failed']} skip={agg['skipped']} "
            f"time={cell_dt:.0f}s, see {cell_log}"
        )
    else:
        _status(
            f"  ✓ L{level} {model}: ok={agg['completed']} fail={agg['failed']} "
            f"skip={agg['skipped']} time={cell_dt:.0f}s"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--hf-dataset-folder", type=str, required=True,
                        help="e.g. factorynet_qa_150k")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--output-root", type=Path, default=Path("output/test_eval"))
    parser.add_argument("--kg-repo", type=str, default="the authors/FactoryBench-KnowledgeGraph",
                        help="HF repo for the knowledge graph used by the prompt builder.")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge for free-form items.")
    parser.add_argument("--cost-limit", type=float, default=None,
                        help="USD cap forwarded to each (level, model) eval run.")
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Items per batch job. A cell with N items runs ceil(N/chunk_size) "
                             "subprocesses, each submitting one batch.")
    parser.add_argument("--sync-threshold", type=int, default=100,
                        help="Tail chunks smaller than this fall back to sync. Set to match "
                             "Bedrock's CreateModelInvocationJob 100-record minimum (default).")
    parser.add_argument("--model-concurrency", type=int, default=4,
                        help="Number of (level, model) eval subprocesses to run in parallel.")
    parser.add_argument("--sync-concurrency", type=int, default=1,
                        help="Number of concurrent sync API calls within a chunk (for "
                             "sync-only providers like Together AI). Default: 1 (sequential).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    unknown = [m for m in args.models if m not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; supported: {MODEL_NAMES}")

    out = args.output_root.resolve()
    q_root = out / "questions"
    p_root = out / "prompts"
    r_root = out / "replies"
    logs_dir = out / "logs"
    for d in (q_root, p_root, r_root, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    global _STATUS_FILE
    _STATUS_FILE = out / "run.log"
    _status(f"== eval_test_set start: dataset={args.hf_dataset_folder} "
            f"levels={args.levels} models={args.models} cost_limit={args.cost_limit}")
    _status(f"   out={out}  cell_logs={logs_dir}")

    # ---- Phase 1: pull test.jsonl + unpack -------------------------------
    _status("PHASE 1: pull test.jsonl from HF + unpack to per-item JSON")
    level_to_q_dir: Dict[int, Path] = {}
    for lvl in args.levels:
        jsonl = _download_test_jsonl(args.repo_id, args.hf_dataset_folder, lvl)
        if jsonl is None:
            _status(f"  L{lvl}: skipped (no test.jsonl)")
            continue
        q_dir = q_root / f"level{lvl}"
        n = _unpack_jsonl_to_json_dir(jsonl, q_dir)
        _status(f"  L{lvl}: unpacked {n} test items")
        if n > 0:
            level_to_q_dir[lvl] = q_dir

    if not level_to_q_dir:
        raise SystemExit("No test items found. Aborting.")

    # ---- Phase 2: download KG + build prompts ----------------------------
    _status("PHASE 2: download KG + build prompts")
    kg_path = hf_hub_download(
        repo_id=args.kg_repo, repo_type="dataset", filename="machines.json"
    )
    level_to_p_dir: Dict[int, Path] = {}
    for lvl, q_dir in level_to_q_dir.items():
        p_dir = p_root / f"level{lvl}"
        _build_prompts(q_dir, p_dir, kg_path)
        n_prompts = len(list(p_dir.glob("*.json")))
        _status(f"  L{lvl}: {n_prompts} prompts built")
        level_to_p_dir[lvl] = p_dir

    # ---- Phase 3: strict-batch eval per (level, model) -------------------
    tasks = []
    for lvl in sorted(level_to_p_dir):
        for model in args.models:
            replies_dir = r_root / f"level{lvl}" / _model_slug(model)
            replies_dir.mkdir(parents=True, exist_ok=True)
            tasks.append((lvl, model, replies_dir))
    _status(f"PHASE 3: chunked batch eval | {len(tasks)} cells | "
            f"chunk_size={args.chunk_size} sync_threshold={args.sync_threshold} | "
            f"concurrency={args.model_concurrency}")

    results: List[Dict[str, Any]] = []
    done_count = 0
    total = len(tasks)
    if args.model_concurrency <= 1:
        for lvl, model, replies_dir in tasks:
            results.append(_eval_one(
                level=lvl, model=model,
                questions_dir=level_to_q_dir[lvl],
                prompts_dir=level_to_p_dir[lvl],
                replies_dir=replies_dir,
                logs_dir=logs_dir,
                no_judge=args.no_judge,
                cost_limit=args.cost_limit,
                max_output_tokens=args.max_output_tokens,
                poll_interval=args.poll_interval,
                chunk_size=args.chunk_size,
                sync_threshold=args.sync_threshold,
                sync_concurrency=args.sync_concurrency))
            done_count += 1
            _status(f"     progress: {done_count}/{total} cells done")
    else:
        with ThreadPoolExecutor(max_workers=args.model_concurrency) as ex:
            futures = {
                ex.submit(
                    _eval_one,
                    level=lvl, model=model,
                    questions_dir=level_to_q_dir[lvl],
                    prompts_dir=level_to_p_dir[lvl],
                    replies_dir=replies_dir,
                    logs_dir=logs_dir,
                    no_judge=args.no_judge,
                    cost_limit=args.cost_limit,
                    max_output_tokens=args.max_output_tokens,
                    poll_interval=args.poll_interval,
                    chunk_size=args.chunk_size,
                    sync_threshold=args.sync_threshold): (lvl, model)
                for (lvl, model, replies_dir) in tasks
            }
            for fut in as_completed(futures):
                lvl, model = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    _status(f"  ✗ L{lvl} {model}: exception: {exc}")
                    results.append({"level": lvl, "model": model, "error": str(exc)})
                done_count += 1
                _status(f"     progress: {done_count}/{total} cells done")

    # ---- Phase 4: aggregate summary --------------------------------------
    _status("PHASE 4: aggregate summary")
    agg_path = out / "_eval_summary.json"
    agg_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    totals = {"completed": 0, "failed": 0, "skipped": 0}
    total_cost = 0.0
    n_errors = 0
    for r in results:
        for k in ("completed", "failed", "skipped"):
            totals[k] += int(r.get(k, 0) or 0)
        try:
            total_cost += float(r.get("total_cost_usd") or r.get("cost_usd") or 0.0)
        except Exception:
            pass
        if "error" in r:
            n_errors += 1
    _status(
        f"DONE: {len(results)} cells | completed={totals['completed']} "
        f"failed={totals['failed']} skipped={totals['skipped']} "
        f"errors={n_errors} total_cost=${total_cost:.2f}"
    )
    _status(f"  per-cell summary: {agg_path}")
    _status(f"  replies tree:     {r_root}")


if __name__ == "__main__":
    main()
