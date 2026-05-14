"""Side-by-side comparison of multiple :class:`Result` objects.

Typical use after running several models on the same split::

    from factorybench import evaluate, compare

    results = {
        "gpt-5.1":            evaluate(model="gpt-5.1",            split="mini"),
        "claude-sonnet-4.6":  evaluate(model="claude-sonnet-4.6",  split="mini"),
        "my-model":           evaluate(model="my-model",           split="mini"),
    }
    comp = compare(results)
    print(comp.to_markdown())
    print(comp.to_latex())
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .result import ItemResult, Result


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #

def compare(
    results: Mapping[str, "Result | str | Path"] | Iterable["Result | str | Path"],
) -> "Comparison":
    """Build a :class:`Comparison` from saved runs or :class:`Result` objects.

    ``results`` may be:

    * A mapping from ``model_name -> Result`` (use the explicit key as the column).
    * A mapping from ``model_name -> path-to-result-json`` (loaded for you).
    * An iterable of :class:`Result` / paths (each column is named from the file
      or the result's ``model_name``).
    """
    by_name: dict[str, Result] = {}

    if isinstance(results, Mapping):
        for name, value in results.items():
            by_name[name] = _coerce(value)
    else:
        for value in results:
            r = _coerce(value)
            by_name[_default_name(value, r)] = r

    if not by_name:
        raise ValueError("compare() needs at least one result")
    return Comparison(by_name)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

class Comparison:
    """Rows = metric (one per level + Overall + parse failures); cols = models."""

    _LEVELS = (1, 2, 3, 4)

    def __init__(self, results: Mapping[str, Result]):
        self.results: dict[str, Result] = dict(results)

    # -- core grid ----------------------------------------------------------- #

    def to_dataframe(self) -> pd.DataFrame:
        """Long-form per-(model, level) table for arbitrary analysis."""
        rows = []
        for model, r in self.results.items():
            per_level = r.by_level()
            for lvl in self._LEVELS:
                rows.append({
                    "model": model,
                    "level": f"L{lvl}",
                    "score": per_level.get(f"L{lvl}", float("nan")),
                    "n_items": sum(1 for it in r.items if it.level == lvl),
                    "parse_failures": sum(
                        1 for it in r.items if it.level == lvl and it.parse_error is not None
                    ),
                })
            rows.append({
                "model": model,
                "level": "Overall",
                "score": r.score,
                "n_items": len(r.items),
                "parse_failures": len(r.parse_failures()),
            })
        return pd.DataFrame(rows)

    def score_grid(self) -> pd.DataFrame:
        """Wide-form grid (rows=metric, cols=model)."""
        df = self.to_dataframe()
        grid = df.pivot(index="level", columns="model", values="score")
        # Preserve the original column order (Python 3.7+ dict order).
        grid = grid[list(self.results.keys())]
        # And our preferred row order.
        order = [f"L{lvl}" for lvl in self._LEVELS] + ["Overall"]
        grid = grid.reindex(order)
        grid.index.name = "Level"
        return grid

    # -- renderers ----------------------------------------------------------- #

    def to_markdown(self, *, decimals: int = 4, bold_best: bool = True) -> str:
        return _render_table(
            self.score_grid(),
            decimals=decimals,
            bold_best=bold_best,
            row_label_header="Level",
            cell_fmt=_md_cell,
            bold_fmt=_md_bold,
            separator_row=_md_separator,
        )

    def to_latex(
        self, *, decimals: int = 4, bold_best: bool = True, booktabs: bool = True
    ) -> str:
        grid = self.score_grid()
        cols = grid.columns.tolist()
        col_spec = "l" + "c" * len(cols)
        header = " & ".join(["Level"] + [_latex_escape(c) for c in cols]) + r" \\"

        lines = [f"\\begin{{tabular}}{{{col_spec}}}"]
        if booktabs:
            lines.append(r"\toprule")
        lines.append(header)
        if booktabs:
            lines.append(r"\midrule")
        else:
            lines.append(r"\hline")

        for level_label, row in grid.iterrows():
            best = _row_argmax(row)
            cells = []
            for col in cols:
                val = row[col]
                cells.append(_latex_cell(val, decimals=decimals, bold=bold_best and col == best))
            lines.append(f"{level_label} & " + " & ".join(cells) + r" \\")

        if booktabs:
            lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    def to_json(self, *, decimals: int = 4) -> str:
        """JSON dump suitable for downstream tooling."""
        grid = self.score_grid()
        payload = {
            "models": list(self.results.keys()),
            "grid": {
                row: {
                    col: (None if _is_nan(val) else round(float(val), decimals))
                    for col, val in grid.loc[row].items()
                }
                for row in grid.index
            },
            "parse_failures": {
                m: len(r.parse_failures()) for m, r in self.results.items()
            },
        }
        return json.dumps(payload, indent=2)

    def __repr__(self) -> str:
        models = ", ".join(self.results.keys())
        return f"Comparison(models=[{models}])"


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #

def _coerce(value) -> Result:
    if isinstance(value, Result):
        return value
    if isinstance(value, (str, Path)):
        return _load_result_json(value)
    raise TypeError(
        f"compare() entries must be Result, str, or Path; got {type(value).__name__}"
    )


def _default_name(source, result: Result) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).stem
    return result.model_name or "model"


def _load_result_json(path: str | Path) -> Result:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [
        ItemResult(
            id=row["id"],
            level=row["level"],
            template_id=row["template_id"],
            template_type=row["template_type"],
            answer_format=row["answer_format"],
            dataset=row.get("dataset"),
            fault_id=row.get("fault_id"),
            raw_output=row.get("raw_output"),
            parsed=row.get("parsed"),
            score=float(row["score"]) if row.get("score") is not None else float("nan"),
            chance=float(row.get("chance", 0.0)),
            parse_error=row.get("parse_error"),
            judge_votes=row.get("judge_votes"),
        )
        for row in (raw.get("items") or [])
    ]
    return Result(
        model_name=raw.get("model_name") or "model",
        items=items,
        wall_time=timedelta(seconds=raw.get("wall_time_seconds") or 0.0),
        cost=raw.get("cost", 0.0),
        judges=raw.get("judges") or [],
    )


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def _render_table(grid, *, decimals, bold_best, row_label_header, cell_fmt, bold_fmt, separator_row):
    cols = grid.columns.tolist()
    lines = ["| " + " | ".join([row_label_header] + list(cols)) + " |"]
    lines.append(separator_row(len(cols) + 1))
    for level_label, row in grid.iterrows():
        best = _row_argmax(row)
        cells = []
        for col in cols:
            val = row[col]
            text = cell_fmt(val, decimals=decimals)
            if bold_best and col == best and not _is_nan(val):
                text = bold_fmt(text)
            cells.append(text)
        lines.append("| " + " | ".join([level_label] + cells) + " |")
    return "\n".join(lines)


def _md_cell(val, *, decimals: int) -> str:
    if _is_nan(val):
        return "--"
    return f"{float(val):.{decimals}f}"


def _md_bold(text: str) -> str:
    return f"**{text}**"


def _md_separator(n_cols: int) -> str:
    return "|" + "|".join(["---"] * n_cols) + "|"


def _latex_cell(val, *, decimals: int, bold: bool) -> str:
    if _is_nan(val):
        return "--"
    text = f"{float(val):.{decimals}f}"
    if bold:
        return r"\textbf{" + text + "}"
    return text


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("_", r"\_")
         .replace("#", r"\#")
    )


def _row_argmax(row: pd.Series) -> str | None:
    """Return the column name with the largest finite value, or None if all NaN."""
    best_val = -math.inf
    best_col = None
    for col, val in row.items():
        if _is_nan(val):
            continue
        f = float(val)
        if f > best_val:
            best_val = f
            best_col = col
    return best_col


def _is_nan(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False
