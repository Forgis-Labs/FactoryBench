"""Convert FactoryBench text-formatted time series into the per-channel
<ts_*>-token prompt that Shrike / BearingModel were pretrained on.

FactoryBench encodes each timestep as a single string of comma-separated
``feature=value`` pairs (optionally prefixed by ``t=<ms>: ``). ``context.time_series``
is a list of those strings (one per timestep). For level-2 / level-3 ranking
questions the options A/B/C/D are themselves multi-timestep TS strings, with
timesteps separated by `` | ``.

To match the Shrike/Bearing pretraining format, we need to:
  1. Parse each row into a {feature: value} dict.
  2. Group across timesteps to get one 1-D signal per feature → "channel".
  3. Push each channel through wrapper.tokenize_ts() to get integer codes.
  4. Emit ``{channel_name} (mean=…, std=…) <ts_start> <ts_X> <ts_Y> ... <ts_end>``
     for each channel, mirroring shrike.model.shrike.Shrike.build_text.

The ``<ts_*>`` strings tokenize to single token IDs because the LLM tokenizer
was extended at model-load time with N codebook entries as special tokens.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


# Hard cap on channels included per signal block, protects long-context
# blowups on samples with 100+ features. Channels beyond this are dropped
# in declaration order (i.e. we keep the FIRST K).
DEFAULT_MAX_CHANNELS = 64


def parse_ts_row(s: str) -> dict[str, float]:
    """Parse a single FactoryBench TS row into {feature: float}.

    Tolerates the optional ``t=<int>: `` timestamp prefix; if present, the
    timestamp is RETAINED as a synthetic channel called ``_time`` so that
    questions referencing absolute timestamps (L1 windowing, L3 counterfactual
    insertion points) still have positional grounding after TOTEM encoding.
    Skips non-numeric values and silently drops malformed pairs.
    """
    s = s.strip()
    out: dict[str, float] = {}
    if s.startswith("t=") and ":" in s:
        ts_part, s = s.split(":", 1)
        try:
            out["_time"] = float(ts_part[2:].strip())
        except ValueError:
            pass
        s = s.strip()
    for kv in s.split(","):
        kv = kv.strip()
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def rows_to_channels(rows: Sequence[str], max_channels: int = DEFAULT_MAX_CHANNELS
                     ) -> tuple[list[str], list[list[float]]]:
    """Convert N rows of ``feat=val, …`` strings into per-channel signals.

    Returns (channel_names, channel_signals) where each signal is a length-N
    list of floats (missing values default to 0.0). Channels are returned in
    first-seen order; only the first ``max_channels`` are kept.
    """
    parsed = [parse_ts_row(r) for r in rows]
    seen: dict[str, None] = {}
    for d in parsed:
        for k in d:
            if k not in seen:
                seen[k] = None
                if len(seen) >= max_channels:
                    break
        if len(seen) >= max_channels:
            break
    keys = list(seen)
    sigs = [[d.get(k, 0.0) for d in parsed] for k in keys]
    return keys, sigs


def _codes_block(ch_name: str, signal: list[float], wrapper) -> str:
    """Tokenize one channel via wrapper.tokenize_ts() and wrap in the canonical
    per-channel format used by Shrike.build_text and BearingModel._build_content."""
    if not signal:
        return f"{ch_name} (mean=0.0000, std=0.0000) <ts_start> <ts_end>"
    sig_t = torch.tensor(signal, dtype=torch.float32)
    # TOTEM/FSQ tokenizers expect a 1-D float tensor on CPU; they return
    # int codes. Anything shorter than compression_factor gets padded inside.
    codes = wrapper.tokenize_ts(sig_t)
    code_text = " ".join(f"<ts_{c}>" for c in codes if c >= 0)
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    return (f"{ch_name} (mean={mean:.4f}, std={std:.4f}) "
            f"<ts_start> {code_text} <ts_end>")


def _ts_section(rows: Sequence[str], wrapper, acronym_map: dict[str, str],
                max_channels: int) -> str:
    """Build the multi-channel TS block: one line per channel."""
    keys, sigs = rows_to_channels(rows, max_channels=max_channels)
    out_lines = []
    for k, sig in zip(keys, sigs):
        desc = acronym_map.get(k, k)
        out_lines.append(_codes_block(desc, sig, wrapper))
    return "\n".join(out_lines)


def _option_is_ts(v) -> bool:
    """True if an option value looks like a TS string (has ``=`` and either
    ``,`` or `` | ``). Otherwise treat as plain text (e.g. "True"/"False")."""
    if not isinstance(v, str):
        return False
    return "=" in v and ("," in v or " | " in v)


def build_ts_prompt(sample: dict, wrapper, max_channels: int = DEFAULT_MAX_CHANNELS
                    ) -> tuple[str, str]:
    """Build (prompt, answer) for a FactoryBench sample in Shrike's wire format.

    The returned prompt is byte-equivalent to what
    ``finetuning/_factorybench_src/train_factorybench.py:_format_chatml`` would
    produce, EXCEPT every block of TS data (context + each TS-shaped option)
    is replaced by ``<ts_start> <ts_X> ... <ts_end>`` per channel. Non-TS
    options (e.g. plain "True"/"False") pass through as plain text.
    """
    question = sample["question"]
    context = sample.get("context", {}) or {}
    ts_format = context.get("time_series_format", {}) or {}
    acronym_map = ts_format.get("acronym_mapping", {}) or {}
    ts_rows = context.get("time_series", []) or []
    options = sample.get("options", {}) or {}

    parts: list[str] = []

    # Keep the feature-mapping cheat sheet, it's small and helps the
    # model line up channel descriptions with the acronyms in the question.
    if acronym_map:
        mapping_str = ", ".join(
            f"{k}={v}" for k, v in list(acronym_map.items())[:10])
        if len(acronym_map) > 10:
            mapping_str += f"... ({len(acronym_map)} total)"
        parts.append(f"Feature mapping: {mapping_str}")

    # Main signal, per-channel <ts_*> blocks
    if isinstance(ts_rows, list) and ts_rows:
        parts.append("Time series data:\n" + _ts_section(
            ts_rows, wrapper, acronym_map, max_channels))

    user_content = "\n\n".join(parts) + ("\n\n" if parts else "") + question

    # Options: tokenize the ones that look like TS, leave the rest as text
    if options:
        opt_lines = []
        for k, v in options.items():
            if _option_is_ts(v):
                opt_rows = [r for r in v.split(" | ") if r.strip()]
                ts_text = _ts_section(opt_rows, wrapper, acronym_map, max_channels)
                opt_lines.append(f"  {k}:\n{ts_text}")
            else:
                opt_lines.append(f"  {k}: {v}")
        user_content += "\n\nOptions:\n" + "\n".join(opt_lines)

    prompt = (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    answer = f"{sample.get('answer', '')}<|im_end|>"
    return prompt, answer
