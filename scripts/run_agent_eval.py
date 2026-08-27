"""Evaluate the GPT-5.1 agentic baseline on a FactoryBench prompt tree.

Iterates over ``<prompts-dir>/*.json`` produced by
``build_prompts_from_questions``, instantiates the four tools per item
(signal_stats, forecast, run_python, retrieve_manual), runs the
ReAct-style ``Agent`` loop, scores with the same ``score_prediction``
used by the zero-shot runners, and writes replies in the identical
JSON shape so downstream aggregators / judge scripts pick them up
without any changes.

Only Foundry chat is supported for the driver model right now
(GPT-5.1). Judges are optional (default off), free-form scoring runs
via the same 3-judge batch script as the other panel cells.

Usage:
    python -m scripts.run_agent_eval \\
        --input output/kuka_eval/prompts/level4 \\
        --output-dir output/kuka_eval/replies/level4/gpt-5.1-agent \\
        --questions output/kuka_qa/level4 \\
        --model gpt-5.1-1 --cost-limit 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load .env before any auth resolution, matches what run_foundry_eval does.
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

import numpy as np

from src.agentic.agent import Agent, BedrockAgent
from src.agentic.tools import (
    ForecastTool, ManualRAGTool, PythonSandboxTool, SignalStatsTool)
from src.config import MODELS, get_upstream_model_id
from src.evaluation.run_foundry_eval import (
    _openai_client, build_question_index, infer_answer_format,
    resolve_api_key, resolve_endpoint, score_prediction)
from src.evaluation.test_gpt_5mini import _estimate_cost as estimate_cost

logger = logging.getLogger(__name__)


# ── time-series parsing ────────────────────────────────────────────────
_TS_ROW_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([\-0-9.eE+]+)")


def _parse_ts(question: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract the {channel: values[]} dict from a question's context."""
    ctx = question.get("context") or {}
    rows = ctx.get("time_series") or []
    ts: Dict[str, List[float]] = {}
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        # dict form: each row is {channel: value...}
        for row in rows:
            for k, v in row.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ts.setdefault(k, []).append(float(v))
        return {k: np.asarray(v, dtype=float) for k, v in ts.items()}
    # string form: each row is "t=<ts>: k1=v1, k2=v2..."
    for row in rows:
        if not isinstance(row, str):
            continue
        for m in _TS_ROW_RE.finditer(row):
            k, v = m.group(1), m.group(2)
            try:
                ts.setdefault(k, []).append(float(v))
            except ValueError:
                continue
    return _with_aliases({k: np.asarray(v, dtype=float) for k, v in ts.items()}, question)


def _with_aliases(ts: Dict[str, np.ndarray], question: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Also key each channel by its expanded name.

    Rows render channels as acronyms (``ec1``) while the prompt carries the
    expansion (``effort_current_1``), and the agent tends to ask for whichever
    it read last. Unaliased, those calls come back as "channel not in the
    item's time series" and burn tool budget, which cost 19 of 127 forecast
    calls in a 200-item run. Aliases point at the same array, so either name
    resolves.
    """
    ctx = question.get("context") or {}
    fmt = ctx.get("time_series_format")
    mapping = fmt.get("acronym_mapping") if isinstance(fmt, dict) else None
    if not isinstance(mapping, dict):
        return ts
    for acro, full in mapping.items():
        if acro in ts and full not in ts:
            ts[str(full)] = ts[acro]
    return ts


# ── one-item worker ────────────────────────────────────────────────────
def _process_one(
    prompt_path: Path,
    questions: Dict[str, Any],
    client,
    driver_model: str,
    upstream_model_id: str,
    rag: ManualRAGTool,
    output_dir: Path,
    max_tool_calls: int,
    overwrite: bool) -> Optional[Dict[str, Any]]:
    custom_id = prompt_path.stem
    out_path = output_dir / f"{custom_id}_answer.json"
    if out_path.exists() and not overwrite:
        return {"skipped": True, "custom_id": custom_id}
    with open(prompt_path, encoding="utf-8") as fh:
        prompt_payload = json.load(fh)
    prompt_text = prompt_payload.get("prompt") or ""
    # Question lookup: try every plausible stem transform.
    # Prompt naming shapes seen in this repo:
    #   (a) "<uuid>_<idx>", KUKA subset (stage_kuka_from_hf), strip _<idx>
    #   (b) "level<N>_<uuid>_<idx>", some builds, strip both prefix + suffix
    #   (c) "level<N>_<uuid>", paper prompts, strip prefix, no suffix
    # UUIDs contain hyphens, not underscores, so any underscore in custom_id is
    # either the level prefix separator or the _<idx> separator.
    candidates = [custom_id]
    # strip trailing _<idx> if idx is short-numeric (avoids splitting inside UUID)
    if re.search(r"_\d+$", custom_id):
        candidates.append(re.sub(r"_\d+$", "", custom_id))
    # strip leading level<N>_
    for cand in list(candidates):
        m = re.match(r"^level\d+_(.+)$", cand)
        if m:
            candidates.append(m.group(1))
    question = None
    for cand in candidates:
        if cand in questions:
            question = questions[cand]
            break
    if question is None:
        logger.warning(f"no question for {custom_id!r}")
        return {"error": "question missing", "custom_id": custom_id}
    ts_dict = _parse_ts(question)
    tools = [
        SignalStatsTool(ts_dict),
        ForecastTool(ts_dict),
        PythonSandboxTool(ts_dict, timeout_s=10),
        rag,
    ]
    if (MODELS.get(driver_model) or {}).get("provider") == "bedrock":
        agent = BedrockAgent(client=client, model=upstream_model_id, tools=tools,
                             max_tool_calls=max_tool_calls)
    else:
        agent = Agent(client=client, model=upstream_model_id, tools=tools,
                      max_tool_calls=max_tool_calls)
    t0 = time.time()
    result = agent.answer(prompt_text)
    elapsed = time.time() - t0

    answer_text = result["answer"]
    fmt = infer_answer_format(question)
    try:
        score, _judge = score_prediction(
            answer_format=fmt,
            prediction=answer_text,
            ground_truth=question.get("answer"),
            acceptance_bounds=question.get("acceptance_bounds"),
            question_text=prompt_text,
            judge_model="",  # judge is a post-hoc batch step, not per-item
        )
    except Exception as exc:
        score = None
        result["error"] = (result.get("error") or "") + f" | score failed: {exc}"

    usage = result.get("usage") or {}
    cost = estimate_cost(driver_model, usage.get("prompt_tokens", 0),
                         usage.get("completion_tokens", 0))

    reply = {
        "custom_id":         custom_id,
        "prompt_file":       str(prompt_path),
        "prompt":            prompt_text,
        "answer":            answer_text,
        "ground_truth":      question.get("answer"),
        "score":             score,
        "llm_judge_score":   None,
        "llm_judge_reason":  None,
        "answer_format":     fmt,
        "question_type":     question.get("template_type"),
        "model":             f"{driver_model}-agent",
        "agent_trace":       result.get("trace"),
        "agent_n_tool_calls": result.get("n_tool_calls"),
        "agent_error":       result.get("error"),
        "usage":             usage,
        "estimated_cost":    cost,
        "wall_clock_s":      round(elapsed, 2),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(reply, fh, indent=2)
    return {"score": score, "cost": cost, "elapsed_s": elapsed, "custom_id": custom_id}


# ── main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="prompt directory (from build_prompts_from_questions)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="where to write <custom_id>_answer.json files")
    ap.add_argument("--questions", type=Path, required=True,
                    help="matching questions directory (with gold answers + time series)")
    ap.add_argument("--model", default="gpt-5.1-1",
                    help="driver model registered in src.config.MODELS (default: gpt-5.1-1)")
    ap.add_argument("--cost-limit", type=float, default=20.0,
                    help="USD cap; runner stops after cumulative estimate crosses this")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="how many items to run in parallel (default 2; agents are stateful)")
    ap.add_argument("--max-tool-calls", type=int, default=6)
    ap.add_argument("--manuals-root", type=Path, default=Path("data/manuals"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N prompts (for smoke tests)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    model_cfg = MODELS.get(args.model) or {}
    if model_cfg.get("provider") == "bedrock":
        import os as _os
        import boto3  # type: ignore
        region = _os.environ.get(model_cfg.get("region_env", ""), "")
        if not region:
            raise SystemExit(f"missing region env {model_cfg.get('region_env')!r} for {args.model}")
        client = boto3.client("bedrock-runtime", region_name=region)
        upstream_model_id = get_upstream_model_id(args.model)
        logger.info(f"driver={args.model!r} → bedrock modelId={upstream_model_id!r} @ region={region}")
    else:
        api_key = resolve_api_key(args.model)
        base_url, _api_style, api_version = resolve_endpoint(args.model)
        client = _openai_client(base_url, api_key, api_version=api_version)
        upstream_model_id = get_upstream_model_id(args.model)
        logger.info(f"driver={args.model!r} → upstream={upstream_model_id!r} @ {base_url[:60]}...")

    questions = build_question_index(args.questions)
    logger.info(f"indexed {len(questions)} questions")

    rag = ManualRAGTool(index_root=args.manuals_root)
    if rag._loaded_err:
        logger.warning(f"manual RAG disabled: {rag._loaded_err}")

    prompt_files = sorted(args.input.glob("*.json"))
    if args.limit:
        prompt_files = prompt_files[:args.limit]
    logger.info(f"processing {len(prompt_files)} prompts with concurrency={args.concurrency}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    done = fail = skip = 0
    def _run(pp):
        return _process_one(
            pp, questions, client, args.model, upstream_model_id, rag,
            args.output_dir, args.max_tool_calls, args.overwrite)

    if args.concurrency <= 1:
        for pp in prompt_files:
            res = _run(pp)
            if res is None: fail += 1; continue
            if res.get("skipped"): skip += 1; continue
            if "error" in res and res["error"]: fail += 1; continue
            done += 1
            total_cost += float(res.get("cost", 0.0))
            logger.info(f"✓ [{done + fail + skip}/{len(prompt_files)}] {res['custom_id']} "
                        f"| score={res.get('score')} | cost={res.get('cost'):.4f} | cum=${total_cost:.2f}")
            if total_cost >= args.cost_limit:
                logger.warning(f"cost cap ${args.cost_limit} reached; stopping")
                break
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run, pp): pp for pp in prompt_files}
            for fut in as_completed(futures):
                res = fut.result()
                if res is None: fail += 1; continue
                if res.get("skipped"): skip += 1; continue
                if "error" in res and res["error"]: fail += 1; continue
                done += 1
                total_cost += float(res.get("cost", 0.0))
                logger.info(f"✓ [{done + fail + skip}/{len(prompt_files)}] {res['custom_id']} "
                            f"| score={res.get('score')} | cost={res.get('cost'):.4f} | cum=${total_cost:.2f}")
                if total_cost >= args.cost_limit:
                    logger.warning(f"cost cap ${args.cost_limit} reached; will drain remaining futures then stop")

    logger.info(f"Done. completed={done} failed={fail} skipped={skip} total_cost=${total_cost:.2f}")
    summary = {
        "input": str(args.input), "output_dir": str(args.output_dir),
        "model": args.model, "processed": done, "failed": fail, "skipped": skip,
        "total_cost_usd": round(total_cost, 4),
    }
    with open(args.output_dir / "_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
