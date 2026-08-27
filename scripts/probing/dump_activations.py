"""Dump per-layer residual-stream activations for a set of FactoryBench items.

Runs a frozen open-weight LLM over each prompt in a directory of question JSONs
and saves the last-token residual-stream activation from every transformer
layer to a Parquet file. The result feeds the linear-probing pipeline.

Usage:
    python scripts/probing/dump_activations.py \\
        --model Qwen/Qwen3-4B \\
        --questions output/test_eval/questions/level1 \\
        --out output/probing/qwen3_4b/level1_activations.parquet \\
        --limit 1000

Notes:
    * Requires transformers, torch, pyarrow.
    * Uses `output_hidden_states=True` so activations from every layer are
      captured. Only the *last* non-padding token's activation is kept.
    * Free-form Level 4 prompts are handled the same way but the concept
      label supplied by `build_labels.py` differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: install torch + transformers before running this script.", file=sys.stderr)
    sys.exit(1)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: install pyarrow before running this script.", file=sys.stderr)
    sys.exit(1)


def _build_prompt(q: dict) -> str:
    """Reconstruct the model-facing prompt from a question JSON."""
    parts = [q.get("question", "")]
    opts = q.get("options") or {}
    if isinstance(opts, dict) and opts:
        for k in sorted(opts.keys()):
            parts.append(f"{k}. {opts[k]}")
    ts = (q.get("context") or {}).get("time_series") or []
    if ts:
        parts.append("Time series:")
        parts.extend(ts)
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--questions", required=True, help="Directory of level*_*.json question files")
    ap.add_argument("--out", required=True, help="Output Parquet path")
    ap.add_argument("--limit", type=int, default=None, help="Cap on number of items")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-length", type=int, default=16384)
    args = ap.parse_args()

    q_dir = Path(args.questions)
    q_paths = sorted(q_dir.glob("*.json"))
    if args.limit is not None:
        q_paths = q_paths[: args.limit]
    if not q_paths:
        print(f"no question JSONs in {q_dir}", file=sys.stderr)
        return 1

    print(f"loading {args.model} on {args.device} ({args.dtype})")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    torch_dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=args.device,
        output_hidden_states=True,
    )
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    rows_id: list[str] = []
    rows_level: list[int] = []
    rows_template: list[int] = []
    rows_prompt_len: list[int] = []
    layer_columns: dict[int, list[bytes]] = {}

    with torch.no_grad():
        for i, qp in enumerate(q_paths):
            try:
                with open(qp, encoding="utf-8") as fh:
                    q = json.load(fh)
            except Exception as e:
                print(f"[skip] {qp.name}: {e}", file=sys.stderr)
                continue
            prompt = _build_prompt(q)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)
            # hidden_states is a tuple of length num_layers+1 (embedding + each layer),
            # each (batch=1, seq_len, hidden). Take the last token from each.
            hs = out.hidden_states
            last_idx = enc["input_ids"].shape[1] - 1
            for ell, h in enumerate(hs):
                vec = h[0, last_idx].to(torch.float32).cpu().numpy().tobytes()
                layer_columns.setdefault(ell, []).append(vec)
            rows_id.append(q.get("id", qp.stem))
            rows_level.append(int(q.get("level", 0)))
            rows_template.append(int(q.get("template_id", 0)))
            rows_prompt_len.append(int(enc["input_ids"].shape[1]))
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(q_paths)}")

    table = pa.table({
        "id": rows_id,
        "level": rows_level,
        "template_id": rows_template,
        "prompt_len": rows_prompt_len,
        **{f"layer_{ell}": pa.array(v, type=pa.binary()) for ell, v in layer_columns.items()},
    })
    pq.write_table(table, args.out, compression="zstd")
    print(f"wrote {len(rows_id)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
