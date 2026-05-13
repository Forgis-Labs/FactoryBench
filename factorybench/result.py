"""``Result`` object returned from ``factorybench.evaluate``."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .score import chance_corrected
from .types import AnswerFormat


@dataclass
class ItemResult:
    """Per-item outcome row."""

    id: str
    level: int
    template_id: int
    template_type: str
    answer_format: str
    dataset: str | None
    fault_id: int | None
    raw_output: str | None
    parsed: Any
    score: float
    chance: float
    parse_error: str | None = None
    # L4 only -- per-judge votes when the item went through a JudgePanel.
    judge_votes: list[dict] | None = None


@dataclass
class Result:
    """Aggregated evaluation result with structured accessors."""

    model_name: str
    items: list[ItemResult] = field(default_factory=list)
    wall_time: timedelta = field(default_factory=lambda: timedelta(0))
    cost: float = 0.0
    # Judge specs used for L4 scoring (empty if no L4 items were judged).
    judges: list[str] = field(default_factory=list)

    # ----- aggregate accessors -----

    @property
    def score(self) -> float:
        """Chance-corrected mean score across all scored items."""
        scored = self._scored()
        if not scored:
            return float("nan")
        mean_acc = sum(r.score for r in scored) / len(scored)
        mean_chance = sum(r.chance for r in scored) / len(scored)
        return chance_corrected(mean_acc, mean_chance)

    @property
    def raw_accuracy(self) -> float:
        """Uncorrected mean per-item score."""
        scored = self._scored()
        if not scored:
            return float("nan")
        return sum(r.score for r in scored) / len(scored)

    def by_level(self) -> dict[str, float]:
        return self._groupby(lambda r: f"L{r.level}")

    def by_template(self) -> dict[str, float]:
        return self._groupby(lambda r: f"L{r.level}.{r.template_id}")

    def by_answer_format(self) -> dict[str, float]:
        return self._groupby(lambda r: r.answer_format)

    def by_dataset(self) -> dict[str, float]:
        """Aggregate by ``provenance.dataset`` (proxy for robot family)."""
        return self._groupby(lambda r: r.dataset or "unknown")

    # Kept for spec-compatibility with usage.md; same as by_dataset for now.
    by_robot = by_dataset

    def by_fault_category(self) -> dict[str, float]:
        return self._groupby(
            lambda r: str(r.fault_id) if r.fault_id is not None else "unknown"
        )

    def parse_failures(self) -> list[ItemResult]:
        return [r for r in self.items if r.parse_error is not None]

    # ----- L4 / judge accessors -----

    def l4_items(self) -> list[ItemResult]:
        """Return only the items that went through the judge panel."""
        return [r for r in self.items if r.judge_votes is not None]

    def fleiss_kappa(self) -> float:
        """Inter-judge agreement (Fleiss' kappa) across all L4 items.

        Returns ``nan`` when there are fewer than two L4 items or fewer than
        two valid votes per item.
        """
        from .judges import fleiss_kappa  # local import to avoid a cycle

        rows = [
            [v["score"] for v in (r.judge_votes or [])]
            for r in self.l4_items()
        ]
        return fleiss_kappa(rows)

    def judge_mode(self) -> str | None:
        """Returns ``"paper-default"`` / ``"single-judge"`` / ``"custom"``,
        or ``None`` if no judges were used.
        """
        from .judges import PAPER_DEFAULT_JUDGES

        if not self.judges:
            return None
        if tuple(self.judges) == PAPER_DEFAULT_JUDGES:
            return "paper-default"
        if len(self.judges) == 1:
            return "single-judge"
        return "custom"

    # ----- export -----

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.items])

    def save(self, path: str | Path) -> None:
        payload = {
            "model_name": self.model_name,
            "score": _coerce_nan(self.score),
            "raw_accuracy": _coerce_nan(self.raw_accuracy),
            "wall_time_seconds": self.wall_time.total_seconds(),
            "cost": self.cost,
            "judges": self.judges,
            "judge_mode": self.judge_mode(),
            "fleiss_kappa": _coerce_nan(self.fleiss_kappa()) if self.judges else None,
            "by_level": self.by_level(),
            "by_template": self.by_template(),
            "by_answer_format": self.by_answer_format(),
            "by_dataset": self.by_dataset(),
            "items": [asdict(r) for r in self.items],
        }
        Path(path).write_text(json.dumps(payload, indent=2, default=str))

    # ----- repr -----

    def __repr__(self) -> str:
        parts = [
            f"Result(model={self.model_name!r}",
            f"n={len(self.items)}",
            f"score={_fmt(self.score)}",
            f"raw_acc={_fmt(self.raw_accuracy)}",
            f"parse_failures={len(self.parse_failures())})",
        ]
        return ", ".join(parts)

    # ----- internals -----

    def _scored(self) -> list[ItemResult]:
        return [
            r for r in self.items
            if r.parse_error is None and not _is_nan(r.score)
        ]

    def _groupby(self, key) -> dict[str, float]:
        buckets: dict[str, list[ItemResult]] = {}
        for r in self._scored():
            buckets.setdefault(key(r), []).append(r)
        out = {}
        for k, rows in buckets.items():
            mean_acc = sum(r.score for r in rows) / len(rows)
            mean_chance = sum(r.chance for r in rows) / len(rows)
            out[k] = chance_corrected(mean_acc, mean_chance)
        return dict(sorted(out.items()))


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _coerce_nan(x: float) -> float | None:
    return None if _is_nan(x) else x


def _fmt(x: float) -> str:
    if _is_nan(x):
        return "nan"
    return f"{x:.4f}"
