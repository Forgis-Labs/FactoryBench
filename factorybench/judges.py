"""LLM-as-judge ensemble for scoring Level 4 (free-form) items.

L4 items are open-ended root-cause / remediation answers. Scoring requires
asking one or more judge models to compare a model's prediction against a
reference answer using a fixed rubric.

Usage from Python:

    from factorybench import evaluate
    from factorybench.judges import JudgePanel, PAPER_DEFAULT_JUDGES

    result = evaluate(
        model="my-model",
        level="L4",
        judges=JudgePanel(PAPER_DEFAULT_JUDGES),
    )

Usage from the CLI:

    factorybench evaluate --model my-model --level L4 --judges paper-default
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import get_model
from .types import Item

# Bump if you change the rubric prompt -- the cache is keyed on this.
RUBRIC_VERSION = "v1"

# Models named in the paper-default ensemble. CLI/Python users can override.
PAPER_DEFAULT_JUDGES: tuple[str, ...] = (
    "gpt-5.1",
    "claude-sonnet-4.6",
    "deepseek-v3.2",
)

# Provider key required per judge (used by precheck_credentials).
_JUDGE_KEY_REQUIREMENTS = {
    "gpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_SCORE_RE = re.compile(r"(?<![0-9.])(?:0\.5|0|1\.0|1|0\.0)(?![0-9])")


RUBRIC_PROMPT = """You are an evaluator scoring an industrial-robot troubleshooting answer against a reference answer written by domain experts.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}
{root_cause_block}
CANDIDATE ANSWER:
{prediction}

Score the candidate on a 0 / 0.5 / 1 scale:
- 1.0  -- captures the same essential conclusion as the reference (same root cause AND reasonable remediation; or for optimization items, the same parameter / adjustment).
- 0.5  -- partially correct (right diagnosis but missing or wrong remediation, or vice versa).
- 0.0  -- does not match the reference.

Respond with ONLY one of: 0, 0.5, or 1. No prose."""


@dataclass
class JudgeVote:
    """One judge's score for one item."""

    judge: str
    score: float  # 0.0, 0.5, or 1.0; NaN if the judge output could not be parsed
    raw: str
    parse_error: str | None = None
    cached: bool = False


@dataclass
class L4Score:
    """Aggregated L4 score for one item across an ensemble of judges."""

    score: float  # median of valid judge votes (NaN if all judges failed)
    votes: list[JudgeVote] = field(default_factory=list)

    @property
    def valid_votes(self) -> list[float]:
        return [v.score for v in self.votes if not math.isnan(v.score)]


# --------------------------------------------------------------------------- #
# JudgePanel
# --------------------------------------------------------------------------- #

class JudgePanel:
    """An ensemble of judges that scores L4 items by majority/median vote.

    Args:
        judges: Iterable of judge specs. Each is resolved through
            :func:`factorybench.registry.get_model` — so the same string forms
            used for the model under test (``"gpt-5.1"``, ``"claude-sonnet-4.6"``,
            registered names, ``"mock-judge"``, ...) all work here.
        cache_dir: Where to cache judge votes. Default: ``~/.cache/factorybench/judges``.
            Pass ``None`` to disable caching.
        rubric_version: Override the rubric version used for cache keys. Bump
            when you change the rubric template so old cached votes are not
            silently reused.
        cache_only: If True, refuse new API calls and return only cached votes.
            Items missing a cached vote get ``parse_error='judge_cache_miss'``.
        concurrency: Number of judges to call in parallel per item. Default:
            ``min(len(judges), 8)``. Set to 1 to force sequential.
    """

    def __init__(
        self,
        judges,
        *,
        cache_dir: str | Path | None = "default",
        rubric_version: str = RUBRIC_VERSION,
        cache_only: bool = False,
        concurrency: int | None = None,
    ):
        self.judge_specs: list[str] = list(judges)
        if not self.judge_specs:
            raise ValueError("JudgePanel requires at least one judge")
        self.rubric_version = rubric_version
        if cache_dir == "default":
            cache_dir = Path.home() / ".cache" / "factorybench" / "judges"
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # When True, this panel refuses to make new API calls and returns only
        # cached votes; items missing a cached vote get parse_error='judge_cache_miss'.
        self.cache_only: bool = cache_only
        # Run judges in parallel by default -- each judge is a different provider,
        # so there's no shared rate limit to worry about. Cap at 8 to keep thread
        # counts sane for big ensembles.
        if concurrency is None:
            concurrency = min(len(self.judge_specs), 8)
        self.concurrency: int = max(1, int(concurrency))
        # Lazy-resolved adapter instances per judge spec, guarded for thread safety.
        self._adapters: dict[str, Any] = {}
        self._adapter_lock = threading.Lock()
        # Running usage tally per judge across the panel's lifetime.
        # ``model`` is the underlying provider model id (e.g. ``claude-sonnet-4-6``)
        # learned from the adapter's usage payload; falls back to the spec name
        # when the adapter doesn't report one.
        self._usage: dict[str, dict] = {
            spec: {"model": spec, "input_tokens": 0, "output_tokens": 0, "calls": 0}
            for spec in self.judge_specs
        }
        self._usage_lock = threading.Lock()

    # -- public API ---------------------------------------------------------- #

    @property
    def is_paper_default(self) -> bool:
        return tuple(self.judge_specs) == PAPER_DEFAULT_JUDGES

    @property
    def is_single_judge(self) -> bool:
        return len(self.judge_specs) == 1

    def score(self, item: Item, prediction: str) -> L4Score:
        """Score one prediction against ``item.answer`` using every judge."""
        rendered = self._render_prompt(item, prediction)
        if self.concurrency <= 1 or len(self.judge_specs) <= 1:
            votes = [self._score_one(spec, rendered, item, prediction) for spec in self.judge_specs]
        else:
            votes: list[JudgeVote] = [None] * len(self.judge_specs)  # type: ignore[assignment]
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._score_one, spec, rendered, item, prediction): i
                    for i, spec in enumerate(self.judge_specs)
                }
                for fut in futures:
                    votes[futures[fut]] = fut.result()
        return L4Score(score=self._aggregate(votes), votes=votes)

    # -- internals ----------------------------------------------------------- #

    def _aggregate(self, votes: list[JudgeVote]) -> float:
        valid = [v.score for v in votes if not math.isnan(v.score)]
        if not valid:
            return float("nan")
        return statistics.median(valid)

    def _score_one(self, spec: str, rendered_prompt: str, item: Item, prediction: str) -> JudgeVote:
        cache_key = self._cache_key(spec, item.id, prediction)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return JudgeVote(judge=spec, score=cached["score"], raw=cached["raw"], cached=True)

        if self.cache_only:
            return JudgeVote(
                judge=spec, score=float("nan"), raw="", parse_error="judge_cache_miss",
            )

        adapter = self._get_adapter(spec)
        try:
            call = getattr(adapter, "predict_with_usage", None)
            if callable(call):
                raw, usage = call(rendered_prompt)
            else:
                raw, usage = adapter.predict(rendered_prompt), None
        except Exception as exc:  # surface to caller as parse_error so the run continues
            return JudgeVote(judge=spec, score=float("nan"), raw="", parse_error=f"{type(exc).__name__}: {exc}")

        self._record_usage(spec, usage)

        score, perr = _parse_judge_score(raw)
        vote = JudgeVote(judge=spec, score=score, raw=raw, parse_error=perr)
        if perr is None:
            self._cache_put(cache_key, {"score": score, "raw": raw, "judge": spec, "timestamp": time.time()})
        return vote

    def _record_usage(self, spec: str, usage: dict | None) -> None:
        if not usage:
            return
        with self._usage_lock:
            bucket = self._usage.setdefault(
                spec, {"model": spec, "input_tokens": 0, "output_tokens": 0, "calls": 0}
            )
            bucket["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            bucket["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            bucket["calls"] += 1
            reported_model = usage.get("model")
            if reported_model:
                bucket["model"] = str(reported_model)

    def usage_by_judge(self) -> dict[str, dict]:
        """Snapshot of accumulated token usage per judge across this panel's calls."""
        with self._usage_lock:
            return {spec: dict(stats) for spec, stats in self._usage.items()}

    def _get_adapter(self, spec: str):
        with self._adapter_lock:
            if spec not in self._adapters:
                self._adapters[spec] = get_model(spec)
            return self._adapters[spec]

    def _render_prompt(self, item: Item, prediction: str) -> str:
        rc = (item.root_cause or "").strip()
        root_cause_block = f"\nREFERENCE ROOT CAUSE: {rc}\n" if rc else ""
        return RUBRIC_PROMPT.format(
            question=item.question.strip(),
            reference=str(item.answer).strip(),
            root_cause_block=root_cause_block,
            prediction=str(prediction).strip(),
        )

    def _cache_key(self, judge_spec: str, item_id: str, prediction: str) -> str:
        h = hashlib.sha256()
        h.update(self.rubric_version.encode("utf-8"))
        h.update(b"\0")
        h.update(judge_spec.encode("utf-8"))
        h.update(b"\0")
        h.update(item_id.encode("utf-8"))
        h.update(b"\0")
        h.update(prediction.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, key: str, payload: dict) -> None:
        if self.cache_dir is None:
            return
        (self.cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Helpers used by the CLI and evaluate()
# --------------------------------------------------------------------------- #

def parse_judges_flag(value: str) -> JudgePanel:
    """Parse the CLI ``--judges`` argument into a :class:`JudgePanel`.

    Accepts:
      * ``"paper-default"`` -- the paper-faithful three-judge ensemble.
      * A comma-separated list of judge specs (e.g. ``"gpt-5.1,claude-sonnet-4.6"``).
    """
    value = value.strip()
    if value == "paper-default":
        return JudgePanel(PAPER_DEFAULT_JUDGES)
    specs = [s.strip() for s in value.split(",") if s.strip()]
    if not specs:
        raise ValueError(f"could not parse --judges={value!r}")
    return JudgePanel(specs)


def precheck_credentials(panel: JudgePanel) -> list[str]:
    """Return a list of missing-env-var error lines for the panel's judges.

    Empty list means every judge has its credential available. (Custom-registered
    judges and ``mock`` skip the check.)
    """
    missing: list[str] = []
    for spec in panel.judge_specs:
        env = _required_env_for_spec(spec)
        if env and not os.environ.get(env):
            missing.append(f"  - {env} (required for {spec})")
    return missing


def _required_env_for_spec(spec: str) -> str | None:
    for prefix, env in _JUDGE_KEY_REQUIREMENTS.items():
        if spec.startswith(prefix):
            return env
    return None


def _parse_judge_score(raw: str) -> tuple[float, str | None]:
    """Extract a 0 / 0.5 / 1 score from a judge's free-form output."""
    if raw is None:
        return float("nan"), "judge returned None"
    text = raw.strip()
    # Fast path -- exact one-token answer.
    if text in {"0", "0.0", "0.5", "1", "1.0"}:
        return float(text), None
    m = _SCORE_RE.search(text)
    if m is None:
        return float("nan"), f"could not extract 0/0.5/1 from {text!r}"
    return float(m.group(0)), None


# --------------------------------------------------------------------------- #
# Fleiss' kappa for inter-judge agreement
# --------------------------------------------------------------------------- #

def fleiss_kappa(vote_matrix: list[list[float]]) -> float:
    """Fleiss' kappa for N items rated by k judges into 3 categories {0, 0.5, 1}.

    ``vote_matrix[i]`` is the list of valid scores for item ``i``. Items with
    fewer than two valid votes are skipped. Returns NaN if fewer than two
    scorable items remain.
    """
    categories = (0.0, 0.5, 1.0)
    rows = []
    n_per_row = []
    for votes in vote_matrix:
        valid = [v for v in votes if not math.isnan(v)]
        if len(valid) < 2:
            continue
        counts = [valid.count(c) for c in categories]
        rows.append(counts)
        n_per_row.append(sum(counts))
    if len(rows) < 2:
        return float("nan")

    N = len(rows)
    # If row counts vary (some items had failed judges), fall back to a per-row
    # normalized version: P_i = (sum_j n_ij^2 - n) / (n(n-1)), averaged.
    P_i_values = []
    p_j_sums = [0.0] * len(categories)
    for counts, n in zip(rows, n_per_row):
        if n < 2:
            continue
        sq = sum(c * c for c in counts)
        P_i_values.append((sq - n) / (n * (n - 1)))
        for j, c in enumerate(counts):
            p_j_sums[j] += c / n
    P_bar = sum(P_i_values) / len(P_i_values)
    p_j = [s / N for s in p_j_sums]
    Pe = sum(p * p for p in p_j)
    denom = 1 - Pe
    if denom <= 0:
        return float("nan")
    return (P_bar - Pe) / denom
