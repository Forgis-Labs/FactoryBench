# Changelog

Notable changes per release. The `--dataset-version` flag pins the HuggingFace
dataset revision -- entries here that mention "dataset vX.Y" refer to that.

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
