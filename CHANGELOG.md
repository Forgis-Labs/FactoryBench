# Changelog

Notable changes per release. The `--dataset-version` flag pins the HuggingFace
dataset revision -- entries here that mention "dataset vX.Y" refer to that.

## 0.0.10

- Fix: `factorybench evaluate --template L2.7 --max-items N` now filters by
  template *before* capping, so the natural reading ("first N items of L2.7")
  works. Previously the cap was applied first and the template filter then
  found 0 items in the slice.
- Add: `Result.load(path)` classmethod -- the documented counterpart to
  `Result.save(path)`. `compare()` and `evaluate --resume` are routed through it.
- usage.md: dropped the stale `--strict false` advice (parser is already
  lenient by design), the `comparison.render()` heatmap (markdown/latex/df
  cover the paper use case), and corrected the `mini` split description
  (50/level x N levels in scope, not a fixed 200).

## 0.0.9

- Exact pre-run token counting via the new `factorybench[tokenizers]` extra
  (uses `tiktoken`; falls back to the `len(text)/4` heuristic when not
  installed). `CostEstimate.precise_tokens` exposes which mode was used.
- Price table extracted to `factorybench/prices.json` with a version stamp.
  User overrides at `~/.config/factorybench/prices.json` are layered on top
  at import time. New `factorybench prices list | source` CLI group.
- Automated test suite (106 tests, zero network calls). Covers parse, score,
  result, judges, cost, compare, cache, data, tokens, evaluate helpers.

## 0.0.8

- `factorybench info` overhaul: pinned dataset revision (HF commit SHA),
  judge-cache stats, full price table, registered models, Python version.
  Add `--json` for paste-into-bug-report. New `factorybench cache stats |
  clear` subcommands. New `factorybench.cache` helpers (`judge_cache_stats`,
  `clear_judge_cache`, `DEFAULT_JUDGE_CACHE_DIR`).

## 0.0.7

- Actual cost tracking. `Result.cost` is no longer always `0.0`: it's populated
  from real `response.usage` reported by the OpenAI / Anthropic SDKs and from
  judge calls in `JudgePanel`. `Result.tokens_used` carries the breakdown
  (candidate vs each judge) and survives `save()` / `load()`. CLI summary now
  prints `actual cost: $X.XX (estimate was $Y.YY)`. Adapters gain an optional
  `predict_with_usage(prompt) -> (str, usage_dict | None)` method; user models
  that only implement `predict()` continue to work unchanged.

## 0.0.6

- Parallel judges within an L4 item: `JudgePanel` defaults to
  `concurrency=min(len(judges), 8)`. CLI: `--judge-concurrency N`. ~3x wall-time
  reduction per L4 item when running the paper-default ensemble.

## 0.0.5

- Cost preview (`factorybench cost`, `evaluate --dry-run`, `evaluate -y/--yes`,
  Python `estimate_cost`, `set_price`). Heuristic token count
  (`len(text)/4`); ~+/-25% accurate. Auto-prompt before any L4 run or runs
  estimated above $1.

## 0.0.4

- `compare(results)` + `factorybench compare` CLI: model x level table in
  markdown / latex / json with best-per-row bolding. Handles absent levels.

## 0.0.3

- Long-eval robustness: `--concurrency N` (thread pool for candidate calls),
  `--resume FILE` (reuse already-scored items from a prior result JSON),
  `--judge-cache-only` (re-aggregate against a saved judge cache without
  re-billing).

## 0.0.2

- CLI (`factorybench info | list | evaluate | export | score`), built-in
  adapters for OpenAI (`gpt-*`), Anthropic (`claude-*`), DeepSeek (`deepseek-*`),
  and a `mock` adapter. Optional install extras: `[openai]`, `[anthropic]`, `[all]`.

## 0.0.1

- First cut: loader, prompt renderer, deterministic scorer for L1-L3,
  `register_model` + `evaluate` Python API, chance-corrected aggregate, Result
  object with `by_level / by_template / by_answer_format / by_dataset` accessors.
  L4 raises `NotImplementedError`.
