"""signal_stats - per-channel summary of the item's time series.

Input: list of channel names (subset of what the item exposes), plus
optional [t_start, t_end] index bounds. Returns per-channel mean, std,
min, max, p05/p50/p95, monotonic-trend sign, first-derivative extrema.
Also a ``residual(a, b)`` helper for the SCE schema.

Tool interface: the agent-loop constructs one instance per question,
seeding it with the parsed time-series dict (channel -> np.ndarray).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SignalStatsTool:
    NAME = "signal_stats"

    def __init__(self, ts: Dict[str, np.ndarray]):
        # ts already parsed from the item's ``context.time_series`` rows.
        self.ts = {k: np.asarray(v, dtype=float) for k, v in ts.items()
                   if not (isinstance(v, np.ndarray) and v.dtype == object)}

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": (
                    "Summary statistics for one or more time-series channels of the "
                    "current question. Use this instead of trying to read numeric "
                    "values from a long TS prompt directly. Supports optional "
                    "index-window bounds and a residual(a, b) view for setpoint "
                    "vs feedback (SCE-schema residual)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "channel names to summarise; must exist in the item's time series",
                        },
                        "t_start": {"type": "integer", "description": "start row index (inclusive), default 0"},
                        "t_end":   {"type": "integer", "description": "end row index (exclusive), default len(series)"},
                        "residual_pairs": {
                            "type": "array",
                            "items": {
                                "type": "array", "items": {"type": "string"},
                                "minItems": 2, "maxItems": 2,
                            },
                            "description": "optional list of [setpoint_channel, feedback_channel] pairs; returns residual stats per pair",
                        },
                    },
                    "required": ["channels"],
                },
            },
        }

    def __call__(
        self,
        channels: List[str],
        t_start: Optional[int] = None,
        t_end: Optional[int] = None,
        residual_pairs: Optional[List[List[str]]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {"channels": {}}
        missing = [c for c in channels if c not in self.ts]
        if missing:
            out["warning"] = f"channels not in time series: {missing}"
            channels = [c for c in channels if c in self.ts]
        for c in channels:
            v = self.ts[c]
            if t_start is not None or t_end is not None:
                v = v[slice(t_start, t_end)]
            out["channels"][c] = self._summ(v)
        if residual_pairs:
            out["residuals"] = {}
            for pair in residual_pairs:
                if len(pair) != 2 or pair[0] not in self.ts or pair[1] not in self.ts:
                    continue
                sp, fb = self.ts[pair[0]], self.ts[pair[1]]
                n = min(len(sp), len(fb))
                res = fb[:n] - sp[:n]
                if t_start is not None or t_end is not None:
                    res = res[slice(t_start, t_end)]
                out["residuals"][f"{pair[1]} - {pair[0]}"] = self._summ(res)
        return out

    @staticmethod
    def _summ(v: np.ndarray) -> Dict[str, float]:
        if v.size == 0 or not np.all(np.isfinite(v)):
            v = v[np.isfinite(v)]
        if v.size == 0:
            return {"n": 0}
        dv = np.diff(v) if v.size >= 2 else np.array([0.0])
        return {
            "n":            int(v.size),
            "mean":         float(np.mean(v)),
            "std":          float(np.std(v)),
            "min":          float(np.min(v)),
            "max":          float(np.max(v)),
            "p05":          float(np.percentile(v, 5)),
            "p50":          float(np.percentile(v, 50)),
            "p95":          float(np.percentile(v, 95)),
            "first_val":    float(v[0]),
            "last_val":     float(v[-1]),
            "trend_sign":   int(np.sign(v[-1] - v[0])),
            "d_max":        float(np.max(np.abs(dv))),
            "d_mean":       float(np.mean(dv)),
        }
