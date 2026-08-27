# FactoryBench: Evaluating Industrial Machine Understanding

[![License](https://img.shields.io/badge/License-Non--Commercial-lightgrey)](LICENSE.txt)

> **Anonymised mirror for double-blind review.**
> Author names, affiliations, contact details, the project website, the preprint
> link and all cloud project identifiers have been removed from this branch.
> Please do not attempt to deanonymise the authors.

FactoryBench is a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present **FactoryWave** (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as 69,691 released Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD.

Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% (chance-corrected) on the structured levels, and that the panel ordering reshuffles on the decision-making level: the model leading Levels 1 to 3 falls to fourth of six on Level 4. Both findings reveal a wide gap between current models and operational machine understanding.

<p align="center">
  <img src="assets/factorybench_pipeline.png" width="100%">
</p>

## Four levels of machine understanding

| Level | Task | Example |
|-------|------|---------|
| **L1 State** | Interpret what the machine is doing now | "What is the current of joint 3 right now?" |
| **L2 Intervention** | Reason about what an action now would change | "If force in joint 3 increases to X now, what happens?" |
| **L3 Counterfactual** | Reason about what a different past would have produced | "If force had increased to X at t=20 ms, what would have happened?" |
| **L4 Decision** | Generate a remediation plan from the trace | "Robot stopped with error C203A. What to do?" |

Each level builds on the previous one, and the four are scored separately so a
failure can be attributed to reading the signal, to reasoning about it, or to
acting on it.

## Headline results

Signed chance-corrected accuracy on Levels 1 to 3, so 0% is chance and a
non-LLM baseline is a real reference rather than a formality. Level 4 is an
uncorrected 0/0.5/1 rubric and is not directly comparable to the other three.

| Model | L1 State | L2 Intervention | L3 Counterfactual | L4 Decision |
|---|---|---|---|---|
| Non-LLM baseline | +4.3% | -0.2% | +6.2% | n/a |
| Claude Sonnet 4.6 | **+11.8%** | **+14.6%** | **+45.7%** | 15.9% |
| Mistral Large 3 | +5.9% | +5.3% | +19.7% | |
| Qwen3-235B | +3.6% | | +20.6% | 19.9% |
| GPT-5.1 | +3.2% | +6.9% | +27.5% | **21.5%** |
| DeepSeek V3.2 | -0.3% | +5.8% | +17.1% | 17.8% |
| Qwen3-4B | -1.8% | | | |

Two results are worth stating plainly. **Four of six models fail to clear the
non-LLM baseline on Level 1**, and two score below chance outright, so raw
scale does not reliably confer the ability to read precise state out of dense
industrial signals. And **the ranking reverses on Level 4**: the model that
leads the structured levels, more than doubling the next model on L3, drops to
fourth when asked to name a root cause and produce a remediation grounded in
manufacturer documentation. Signal comprehension and protocol retrieval are
separable abilities, and optimizing one does not deliver the other.

## FactoryWave

<p align="center">
  <img src="assets/factorybench_collage.png" width="100%">
</p>

| | |
|---|---|
| **Q&A items** | 69,691 released |
| **Episodes** | ~15,000 normalized |
| **Question templates** | 21 structured templates |
| **Answer formats** | Single-select, multi-select, ranking, numerical, tensor, free-form |
| **Robots** | UR3 cobot (125 Hz) + KUKA KR10 industrial arm (83 Hz) |
| **Faults** | 27 types across 3 tasks (pick-and-place, peg-in-hole, screwdriving) |
| **Counterfactuals** | Recorded on hardware, not simulated |

Level 3 ground truth is unusual and worth explaining: rather than simulating a
counterfactual, each baseline episode is re-executed 3 to 5 times on the
physical robot with every controllable condition held fixed and the target
fault injected at a fixed timestep. The run whose pre-injection segment
minimizes the signature-kernel MMD against the baseline is kept as the
empirical proxy for the do-operation.

## FactoryBench-Lite

Evaluating on the full pool is expensive, and its level aggregates are weighted
by how many items each template happens to contribute. **FactoryBench-Lite** is
a balanced 3,000-item subset: balanced across all 21 templates, and within each
template on the dimension that actually determines its answer. It is the
recommended entry point for cheap evaluation and the pool the reported panel
results are measured on.

## Quick start

```bash
pip install datasets
```

```python
from datasets import load_dataset

# The balanced subset, one file per level
lite = load_dataset("FactoryBench/FactoryBench", data_files="factorybench_lite/level_*.jsonl")

# Or the full released pool
full = load_dataset("FactoryBench/FactoryBench", data_files="factorybench_qa/level_*.jsonl")
```

Each item carries the question, the rendered time-series context, the answer,
its acceptance bounds, and provenance back to the source episode.

## Generating and evaluating

The generation, prompting, inference and scoring stages are separate:

```bash
pip install -e .
python -m src.pipeline.run_pipeline --help
```

See [`src/evaluation/gcp-setup.md`](src/evaluation/gcp-setup.md) for wiring up
an inference provider.

## For reviewers

This branch is the code accompanying a paper under double-blind review. Paper
sources are deliberately not included; the submitted PDF is on the review site.

Everything the paper's experimental claims rest on is here:

| Claim in the paper | Code |
|---|---|
| Q&A generation across the four levels | [`src/question_generation/`](src/question_generation/) |
| Scoring, parsers, LLM-as-judge protocol | [`src/scoring/`](src/scoring/) |
| Zero-shot panel evaluation | [`src/evaluation/`](src/evaluation/) |
| Time-series foundation model baselines (Chronos-Bolt, TimesFM-2.5) | [`experiments/v1/`](experiments/v1/) |
| Tool-augmented ReAct agent baseline | [`src/agentic/`](src/agentic/) |
| Linear probing of frozen activations | [`scripts/probing/`](scripts/probing/) |
| **Human expert baseline, 102 items released per item** | [`human_baseline/`](human_baseline/) |

The two foundation-model baselines ship their per-item results, so
[`experiments/v1/`](experiments/v1/) reproduces the appendix tables without a
GPU: `python experiments/v1/analyze.py --results experiments/v1/results_full.json`.

The human baseline ships the question, the expert's answer, a prose derivation,
runnable derivation code and its captured output for all 102 items; see
[`human_baseline/README.md`](human_baseline/README.md) for what it does and does
not establish.

**Dataset access.** The benchmark and the FactoryWave dataset are available
from an anonymous mirror:
[`huggingface.co/datasets/FactoryBench/FactoryBench`](https://huggingface.co/datasets/FactoryBench/FactoryBench).
Scripts read the identifier from configuration (`src/config.py`), so it can
equally be pointed at a local copy. The raw telemetry is multi-GB and is not
committed to this branch.

## Citation

```bibtex
@article{anonymous2026factorybench,
  title   = {FactoryBench: Evaluating Industrial Machine Understanding},
  author  = {Anonymous Authors},
  note    = {Under review. Preprint reference withheld for double-blind review.},
  year    = {2026}
}
```

## License

Copyright (c) 2026, the authors. All rights reserved. Code is licensed under
the [Non-Commercial Source Code License (anonymised for double-blind review)](LICENSE.txt); the data and
benchmark artefacts are released under CC BY-NC 4.0. Commercial use of either
requires a separate licence.
