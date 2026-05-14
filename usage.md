# Using FactoryBench

FactoryBench evaluates language models and time-series models on industrial machine understanding across four causal levels (state, intervention, counterfactual, decision). This guide walks you from a fresh install to a scored model.

## Contents

1. [Install](#install)
2. [Your first score](#your-first-score)
3. [Evaluate your own model](#evaluate-your-own-model)
4. [Offline workflow (export and score)](#offline-workflow)
5. [Comparing models](#comparing-models)
6. [Working with results](#working-with-results)
7. [Level 4 and the LLM-as-judge ensemble](#level-4-and-the-llm-as-judge-ensemble)
8. [Submitting to the leaderboard](#submitting-to-the-leaderboard)
9. [Useful flags and shortcuts](#useful-flags-and-shortcuts)
10. [Troubleshooting](#troubleshooting)

---

## Install

FactoryBench requires Python 3.10 or newer.

```bash
pip install factorybench
```

The base install pulls in everything needed for the CLI, scoring, and offline evaluation. Provider SDKs are optional extras so you only install what you actually use:

```bash
pip install "factorybench[openai]"       # GPT-5.1, DeepSeek (OpenAI-compatible)
pip install "factorybench[anthropic]"    # Claude Sonnet 4.6
pip install "factorybench[local]"        # HuggingFace transformers + torch
pip install "factorybench[vllm]"         # vLLM-served models
pip install "factorybench[all]"          # everything
```

Verify the install:

```bash
factorybench info
```

You should see the package version, the pinned dataset version, the cache directory, and which provider credentials are detected in your environment.

---

## Your first score

Set the API key for whichever provider you want to use:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

Run the `mini` split. It samples 50 items from each level you're evaluating — so `--level all` without judges runs L1+L2+L3 (150 items) and `--level all --judges paper-default` adds L4 (200 items). Either way it finishes in about a minute and costs a few cents:

```bash
factorybench evaluate --model gpt-5.1 --split mini
```

You will see a per-level progress bar followed by a results table with chance-corrected accuracy, per-format breakdowns, and parse-failure rates. The full results are written to `./results/`.

Once you trust the setup, run a single level:

```bash
factorybench evaluate --model gpt-5.1 --level L2
```

Or the entire benchmark:

```bash
factorybench evaluate --model gpt-5.1 --level all
```

For `--level all`, the CLI prints a cost and time estimate and asks for confirmation before making any API calls.

Note that running L4 requires opting into the LLM-as-judge ensemble; see [Level 4 and the LLM-as-judge ensemble](#level-4-and-the-llm-as-judge-ensemble) below.

---

## Evaluate your own model

FactoryBench evaluates anything that implements one method:

```python
def predict(self, prompt: str) -> str: ...
```

That is the entire contract. The library renders the prompt (machine description, acronym mapping, time-series rows, question, options), hands you the final string, and parses whatever string you return.

### Register a model

```python
# evaluate_my_model.py
from factorybench import register_model, evaluate

@register_model("my-model")
class MyModel:
    def __init__(self):
        from my_library import load_checkpoint
        self.model = load_checkpoint("./checkpoints/best.pt")

    def predict(self, prompt: str) -> str:
        return self.model.generate(prompt, max_tokens=512)

results = evaluate(model="my-model", level="L2")
print(results)
```

Run it:

```bash
python evaluate_my_model.py
```

Once registered, your model is also available from the CLI as long as the script is importable:

```bash
factorybench evaluate --model my-model --level L4
```

### Common patterns

**A remote HTTP endpoint:**

```python
import os, httpx
from factorybench import register_model

@register_model("our-internal-llm")
class InternalLLM:
    def predict(self, prompt: str) -> str:
        r = httpx.post(
            "https://llm.internal.corp/v1/complete",
            json={"prompt": prompt, "max_tokens": 512},
            headers={"Authorization": f"Bearer {os.environ['INTERNAL_TOKEN']}"},
            timeout=60,
        )
        return r.json()["text"]
```

**A local HuggingFace model:**

```python
from transformers import pipeline
from factorybench import register_model

@register_model("my-hf-model")
class HFModel:
    def __init__(self):
        self.pipe = pipeline("text-generation", model="meta-llama/Llama-3-8B-Instruct")

    def predict(self, prompt: str) -> str:
        return self.pipe(prompt, max_new_tokens=512)[0]["generated_text"]
```

**A model with native batching** (optional; speeds things up if your backend supports it):

```python
@register_model("my-batched-model")
class BatchedModel:
    def predict(self, prompt: str) -> str:
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts: list[str]) -> list[str]:
        return self.model.generate_batch(prompts, max_tokens=512)
```

If `predict_batch` is defined, FactoryBench uses it automatically.

---

## Offline workflow

For pipelines where you cannot or do not want to integrate the Python API directly, such as closed-source providers, Slurm clusters, batch inference services, or simply preferring your own runner, use the two-step export/score flow.

### Step 1: export prompts

```bash
factorybench export --level L2 --output prompts.jsonl
```

This writes one JSON object per line with the rendered prompt and metadata:

```json
{"item_id": "L2_0001", "prompt": "The following sensor data...", "answer_format": "single_select_mcq"}
{"item_id": "L2_0002", "prompt": "The following sensor data...", "answer_format": "tensor_6vec"}
```

### Step 2: run inference

Run your model on each prompt however you like, and produce a JSONL file with the predictions:

```json
{"item_id": "L2_0001", "prediction": "C"}
{"item_id": "L2_0002", "prediction": "[1.23, 0.87, -0.42, 0.15, 2.01, -0.03]"}
```

### Step 3: score

```bash
factorybench score --predictions predictions.jsonl --output scores.json
```

The scorer applies the same chance correction, parsing, and per-format rules as the integrated runner. Output schema is identical to `factorybench evaluate`, so downstream tooling treats both paths the same.

If your predictions cover Level 4 items, the score command requires the same `--judges` flag described in the next section.

---

## Comparing models

```python
from factorybench import evaluate, compare

results = {
    "gpt-5.1": evaluate(model="gpt-5.1", split="mini"),
    "claude-sonnet-4.6": evaluate(model="claude-sonnet-4.6", split="mini"),
    "my-model": evaluate(model="my-model", split="mini"),
}

comparison = compare(results)
print(comparison.to_markdown())  # paste into a paper draft
print(comparison.to_latex())     # paste into a LaTeX paper
comparison.to_dataframe()        # long-form pandas for custom plotting
```

The rendered table follows the layout of Figure 3a in the paper, so your numbers drop in next to the published baselines without reformatting. From the command line, `factorybench compare run1.json run2.json --format latex` does the same thing against saved Result files.

---

## Working with results

`evaluate(...)` returns a `Result` object with structured accessors:

```python
results = evaluate(model="gpt-5.1", level="L2")

results.score                  # 0.300, overall chance-corrected score
results.by_level()             # {"L2": 0.300}
results.by_template()          # per-template breakdown
results.by_answer_format()     # single-select / multi-select / ranking / tensor
results.by_robot()             # UR3 / KUKA KR10 / Yu-Cobot
results.by_fault_category()    # L2/L3 only
results.parse_failures()       # items whose output could not be parsed

results.to_dataframe()         # pandas DataFrame, one row per item
results.save("./out.json")     # full results in JSON
results.cost                   # $ spent on API calls
results.wall_time              # timedelta
```

Custom analysis is just pandas from there:

```python
df = results.to_dataframe()
df.groupby(["fault_category", "task"])["score"].mean()
```

---

## Level 4 and the LLM-as-judge ensemble

Levels 1, 2, and 3 are scored deterministically. **Level 4 is different.** L4 answers are free-form natural language (root-cause diagnoses and remediation procedures), so scoring requires an LLM-as-judge.

The judge pipeline is **opt-in**. By default, `factorybench evaluate` skips L4 unless you explicitly request it, because running it has two consequences you should accept knowingly:

1. **You must supply API keys for every judge in the ensemble.** The library does not ship credentials.
2. **You pay for the judge API calls out of your own budget.** For the paper-faithful three-judge protocol, judge calls typically cost more than your model's own inference.

### Opting in

To run L4, pass the `--judges` flag with one of:

```bash
# Paper-faithful protocol: 3-judge ensemble (GPT-5.1, Claude Sonnet 4.6, DeepSeek V3.2)
factorybench evaluate --model my-model --level L4 --judges paper-default

# Custom ensemble
factorybench evaluate --model my-model --level L4 --judges gpt-5.1,claude-sonnet-4.6

# Single-judge mode (cheaper but not paper-comparable; clearly flagged in output)
factorybench evaluate --model my-model --level L4 --judges deepseek-v3.2
```

If you omit `--judges` and L4 is in scope, the CLI exits with a clear error rather than proceeding with no scoring.

### Required API keys

The library checks at startup that every judge has a credential available. If any are missing, you get:

```
Error: L4 evaluation requested with --judges paper-default, but the following
credentials are missing:
  - ANTHROPIC_API_KEY (required for claude-sonnet-4.6)
  - DEEPSEEK_API_KEY  (required for deepseek-v3.2)

Set the missing variables, or choose a smaller ensemble with --judges.
```

No judge API call is ever made under FactoryBench's credentials. The keys come from your environment.

### Cost transparency

When L4 is selected, the cost preview separates model cost from judge cost:

```
This will evaluate my-model on 12,827 L4 items.
Estimated cost:
  Model calls (my-model):    $18.90
  Judge calls (3 judges):   $124.50
    GPT-5.1:                  $48.20
    Claude Sonnet 4.6:        $69.30
    DeepSeek V3.2:             $7.00
  Total:                    $143.40

Continue? [y/N]:
```

The full L4 split with the paper-default ensemble typically costs $100 to $160 per model evaluated. For development, the `mini` split with one judge brings this down to a couple of dollars.

### Ways to manage L4 cost

- **Iterate on `--split mini` first.** Two hundred items, three judges, costs about $2.
- **Use a single judge** with `--judges deepseek-v3.2`. Drops the full L4 cost to roughly $7. Results are flagged `"judge_mode": "single-judge"` and are not directly comparable to the paper.
- **Cache aggressively.** Re-runs only pay for items whose predictions or judge ensemble have changed. The cache is keyed on `(item_id, prediction_hash, judge_model, rubric_version)`.
- **Score offline against a saved cache** with `factorybench score --predictions ... --judges paper-default --judge-cache-only`. Lets you re-aggregate without making new judge calls.

### What's in the L4 results

Each L4 row in `predictions.jsonl` keeps the individual judge votes alongside the aggregate, so any score is auditable:

```jsonl
{"item_id": "L4_0042", "prediction": "...", "judge_votes": {"gpt-5.1": 0.5, "claude-sonnet-4.6": 0.5, "deepseek-v3.2": 0.0}, "score": 0.5}
```

The summary additionally reports inter-judge agreement (Fleiss' kappa) so you can tell when the median is hiding genuine disagreement among judges.

---

## Submitting to the leaderboard

The public leaderboard scores your model against a private 15% held-out split, which guards against contamination. Submit a saved results file:

```bash
factorybench submit \
    --predictions ./results/my-model_all.json \
    --model-name "MyLab Qwen3-7B-FT-v2" \
    --paper "https://arxiv.org/abs/..."
```

The command returns your scores on each level, your current leaderboard rank, and a citation block ready to paste into your paper.

---

## Useful flags and shortcuts

```bash
# 200-item smoke test, ~1 minute, ~$0.05
factorybench evaluate --model my-model --split mini

# evaluate one template only
factorybench evaluate --model my-model --template L2.7

# limit items for debugging
factorybench evaluate --model my-model --level L2 --max-items 20

# resume an interrupted run
factorybench evaluate --model my-model --level all --resume

# verify the pipeline without calling the model
factorybench evaluate --model my-model --level L2 --dry-run

# list what's available
factorybench list models
factorybench list levels
factorybench list templates --level L2

# diagnostics
factorybench info
```

`--resume` is worth knowing about: a `--level all` run can take hours, and resume-from-checkpoint means you do not pay twice for a crashed run.

---

## Troubleshooting

**`factorybench info` shows a provider as not configured.**
Set the corresponding environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) and try again. The CLI looks them up at runtime, not install time.

**L4 evaluation refuses to start.**
L4 requires `--judges` to be set explicitly and the corresponding API keys to be present. See the [Level 4 section](#level-4-and-the-llm-as-judge-ensemble) above.

**High parse-failure rate.**
Check `results.parse_failures()` for the offending outputs. The parser already extracts answers from prose preambles like *"Sure! The answer is C"* — it searches for letters / TF strings / numbers anywhere in the output, not just bare tokens. Persistent failures usually mean the model is returning the wrong *kind* of answer (e.g., a word like `"high"` for a scalar-numeric question, or a multi-paragraph essay for a single-letter MCQ). Adjust the system prompt to return only the format the question asks for.

**Cost surprised me.**
Always run `--dry-run` first on full evaluations. It prints token-level cost estimates without calling the model. The `mini` split is the recommended way to validate a setup before paying for the full corpus.

**Reproducing a paper number.**
Pin the dataset version: `factorybench evaluate --model gpt-5.1 --level L2 --dataset-version v1.0`. The library refuses silent dataset upgrades; reported numbers in the paper correspond to specific tagged releases listed in `CHANGELOG.md`.

**Reporting a bug.**
Include the output of `factorybench info` in your issue. It captures the package version, the pinned dataset version, the cache state, and the configured providers, which is what we ask for first anyway.

---

## Where to go next

- **Add your model to the built-in adapters**: see `docs/add-your-model.md`
- **Add a new robot or task**: see `docs/advanced/add-a-robot.md`
- **Add a new question template**: see `docs/advanced/add-a-template.md`
- **API reference**: see `docs/api-reference.md`
- **Leaderboard**: https://factorybench.org/leaderboard
