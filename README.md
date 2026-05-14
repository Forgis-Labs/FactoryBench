# factorybench

Minimal Python library for loading and scoring the **FactoryBench** test split
([`FactoryBench/FactoryBench`](https://huggingface.co/datasets/FactoryBench/FactoryBench) on the Hugging Face Hub).

This is a focused first cut: it implements the **dataset loader**, **prompt renderer**,
and **deterministic scorer** for Levels 1–3. Level 4 (free-form decision making) needs
an LLM-as-judge ensemble and is intentionally out of scope here.

## Install

```bash
pip install -e .                       # base install (mock model only)
pip install -e ".[openai]"             # adds OpenAI / DeepSeek (gpt-*, deepseek-*)
pip install -e ".[anthropic]"          # adds Anthropic (claude-*)
pip install -e ".[all]"                # everything
```

Python 3.10+.

## Quick start (Python)

```python
from factorybench import register_model, evaluate

@register_model("dummy")
class Dummy:
    def predict(self, prompt: str) -> str:
        return "0"

result = evaluate(model="dummy", level="L1", max_items=20)
print(result.score)
print(result.by_level())
print(result.by_answer_format())
df = result.to_dataframe()
```

`evaluate(...)` accepts:

- `model`: a registered model name, a built-in provider spec (`"mock"`, `"gpt-5.1"`,
  `"claude-sonnet-4.6"`, `"deepseek-chat"`, ...), or any object with `predict(prompt) -> str`
  (optionally `predict_batch(prompts) -> list[str]`).
- `level`: `"L1" | "L2" | "L3" | "all"` (L4 is rejected — needs a judge ensemble).
- `split`: `"test"` (default), `"validation"`, `"train"`, or `"mini"` (50 items per level).
- `max_items`: cap per level (useful for smoke tests).
- `progress`: show a tqdm bar (default `True`).

## CLI

```bash
factorybench info                                            # version, dataset revision, cache, providers, prices
factorybench info --json                                     # machine-readable (paste into bug reports)
factorybench cache stats                                     # judge cache size + age
factorybench cache clear -y                                  # wipe the judge cache
factorybench list models                                     # registered + built-in
factorybench list levels                                     # item counts per level
factorybench list templates --level L2                       # templates within a level

# online evaluation against a provider
factorybench evaluate --model claude-sonnet-4.6 --split mini
factorybench evaluate --model gpt-5.1 --level L2 --max-items 50
factorybench evaluate --model mock --level all --split mini  # no API needed

# user-registered model from a script file
factorybench evaluate --model my-model --import ./my_models.py --split mini

# offline workflow (run inference yourself)
factorybench export --level L2 --output prompts.jsonl
# ... run your model over prompts.jsonl, writing {item_id, prediction} JSONL ...
factorybench score --predictions predictions.jsonl --level L2 --output scores.json
```

Built-in provider specs map to:

| Pattern        | Adapter                             | Required env var      |
|----------------|-------------------------------------|-----------------------|
| `mock`         | Canned-response (no API)            | --                    |
| `gpt-*`        | OpenAI chat completions             | `OPENAI_API_KEY`      |
| `claude-*`     | Anthropic Messages API              | `ANTHROPIC_API_KEY`   |
| `deepseek-*`   | DeepSeek (OpenAI-compatible)        | `DEEPSEEK_API_KEY`    |

## Scoring

Each item is scored 0--1, then aggregated with a chance-corrected accuracy:

```text
score = (mean_accuracy - mean_chance) / (1 - mean_chance)
```

Chance rates per answer format:

| Format                | Chance                                |
|-----------------------|---------------------------------------|
| `single_letter_mcq`   | `1 / #options`                        |
| `four_letter_tf`      | `0.5` (per-element TF)                |
| `four_letter_ranking` | `1/24` (24 permutations of A,B,C,D)   |
| `scalar_range`        | `0`                                   |
| `scalar_margin`       | `0`                                   |
| `tensor_margin`       | `0`                                   |
| `scalar_exact`        | `0`                                   |

## Level 4 (LLM-as-judge)

Levels 1--3 are scored deterministically. **Level 4** items are open-ended root-cause
and remediation answers, so they need an LLM-as-judge. The judge pipeline is opt-in:
you supply the judges and pay for their API calls.

```bash
# Paper-faithful 3-judge ensemble (GPT-5.1 / Claude Sonnet 4.6 / DeepSeek V3.2).
factorybench evaluate --model my-model --level L4 --judges paper-default

# Custom ensemble.
factorybench evaluate --model my-model --level L4 --judges gpt-5.1,claude-sonnet-4.6

# Single-judge mode (cheaper; not paper-comparable -- flagged in output).
factorybench evaluate --model my-model --level L4 --judges deepseek-v3.2
```

The CLI **prechecks** required env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`DEEPSEEK_API_KEY`) before any call is made -- if a key is missing, it exits with a
clean error naming exactly which variable to set.

Each L4 item's votes and the aggregate (median) are preserved in the saved JSON,
and the result reports inter-judge agreement (Fleiss' kappa):

```python
result.l4_items()       # ItemResults that went through the panel
result.fleiss_kappa()   # agreement across all L4 items
result.judge_mode()     # 'paper-default' | 'single-judge' | 'custom'
```

Judge calls are cached at `~/.cache/factorybench/judges/`, keyed on
`(item_id, prediction_hash, judge_model, rubric_version)`. Re-runs with the same
predictions and judge set make zero new API calls.

**Judges run in parallel by default** (one thread per judge in the panel, capped
at 8). Override with `--judge-concurrency N` -- set `1` to force sequential, or
raise it for very large ensembles. Each judge is a different provider, so they
don't share rate limits and parallelism is a free ~3x speedup on the
paper-default ensemble.

By default, `--level all` skips L4 unless `--judges` is also given.

## Long-eval robustness

Three flags meant for real (long, expensive) runs:

```bash
# Run candidate-model predict() in parallel. API adapters are I/O-bound; threads
# are the right tool. Order-preserving. Default 1 (sequential). Ignored when
# the model defines predict_batch.
factorybench evaluate --model gpt-5.1 --level all --concurrency 8

# If a run crashes, restart it. --resume points at a Result JSON from a prior
# run; items that were scored without a parse_error are reused verbatim, and only
# the remaining items are sent to the model (and judges).
factorybench evaluate --model gpt-5.1 --level L4 --judges paper-default \
    --output results/my-run.json
# ... crash midway ...
factorybench evaluate --model gpt-5.1 --level L4 --judges paper-default \
    --output results/my-run.json --resume results/my-run.json

# Re-aggregate against a saved judge cache without paying for any new judge call.
# Items missing a cached vote are reported with parse_error="judge_cache_miss".
factorybench evaluate --model my-model --level L4 --judges paper-default \
    --judge-cache-only
```

Same three options are available on the Python API:

```python
result = fb.evaluate(
    model="gpt-5.1",
    level="L4",
    judges="paper-default",
    concurrency=8,
    resume_from="results/my-run.json",
    judge_cache_only=False,
)
```

## Cost preview

Before kicking off a paid run, get an order-of-magnitude estimate:

```bash
# Standalone preview (no API calls).
factorybench cost --model claude-sonnet-4.6 --level L4 --judges paper-default

# evaluate --dry-run: same preview, exits 0 without running.
factorybench evaluate --model claude-sonnet-4.6 --level L4 \
    --judges paper-default --dry-run

# Any L4 run (or run with estimated total > $1) auto-prompts. Skip with --yes
# in CI / scripted runs.
factorybench evaluate --model claude-sonnet-4.6 --level L4 --judges paper-default
# > Continue? [y/N]:

factorybench evaluate --model claude-sonnet-4.6 --level L4 --judges paper-default --yes
```

Python API:

```python
from factorybench import estimate_cost, set_price

# Override stale or missing prices ($/M tokens).
set_price("my-internal-model", input_per_m=0.5, output_per_m=2.0)

est = estimate_cost(
    model="claude-sonnet-4.6",
    level="L4",
    judges="paper-default",
    max_items=100,
)
print(est.format())
est.to_dict()
```

### Actual cost after the run

The built-in OpenAI / Anthropic adapters report each call's real
`response.usage` to `factorybench`. After the run, `Result.cost` carries the
**actual** dollar total and `Result.tokens_used` shows the breakdown:

```python
result = evaluate(model="claude-sonnet-4.6", level="L4", judges="paper-default", split="mini")
result.cost              # 1.4032 (real, not heuristic)
result.tokens_used
# {"candidate": {"model": "claude-sonnet-4-6", "input_tokens": ..., "output_tokens": ..., "calls": ...},
#  "judges": {"gpt-5.1": {"model": "gpt-5.1", "input_tokens": ..., "output_tokens": ..., "calls": ...},
#             "claude-sonnet-4.6": {...}, "deepseek-v3.2": {...}}}
```

The CLI summary prints both numbers side by side:

```text
actual cost         : $1.4032  (estimate was $1.8200)
```

This works for any adapter that implements
`predict_with_usage(prompt) -> (str, {"input_tokens": ..., "output_tokens": ..., "model": "..."})`.
User-registered models that only implement plain `predict()` keep running and
just report `cost: 0.0` (no spend tracked).

## Comparing models

After running several models on the same split, drop a model x level table into
your paper:

```python
from factorybench import evaluate, compare

results = {
    "gpt-5.1":            evaluate(model="gpt-5.1",           split="mini"),
    "claude-sonnet-4.6":  evaluate(model="claude-sonnet-4.6", split="mini"),
    "my-model":           evaluate(model="my-model",          split="mini"),
}
comp = compare(results)
print(comp.to_markdown())                # paste into a draft
print(comp.to_latex(booktabs=True))      # paste into LaTeX
comp.to_dataframe()                      # long-form pandas
```

Best-per-row is bolded automatically (`bold_best=False` to disable). Absent or
unscored levels render as `--`. `compare()` also accepts a dict of
``model_name -> path-to-result.json`` (loaded for you), or just an iterable of
paths (column names come from file stems).

From the CLI:

```bash
factorybench compare results/m0.json results/m1.json results/m2.json
factorybench compare results/*.json --format latex --output table.tex
factorybench compare m0.json m1.json --name gpt-5.1 --name claude-sonnet-4.6
```

## What's intentionally missing

- **Leaderboard `submit`** subcommand (needs a backend).
- Auto-fetched price table; the static snapshot is updated by hand. Use
  `set_price()` or the `FB_PRICES` env var to override.
- Prompt caching on the Anthropic adapter -- each FactoryBench item has a unique
  time-series body, and the shared prefix per level (description + acronym mapping)
  is well below the 2K / 4K minimum cacheable prefix on Sonnet / Opus. Adding
  `cache_control` here would write entries that are never read.
