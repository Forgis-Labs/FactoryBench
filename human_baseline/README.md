# Human expert baseline

The expert baseline reported in the paper's appendix, released per item.

## What is here

`submission/sampled_dataset.json` — **102 items**, and for each one:

| Field | |
|---|---|
| `question`, `options`, `context` | the item exactly as a model receives it |
| `answer` | the expert's answer |
| `answer_derivation` | prose: how the answer follows from the telemetry |
| `solve_code` | runnable code deriving the answer from the raw signals |
| `solve_output` | that code's captured output |
| `level`, `template_id`, `original_question_id` | provenance |

Coverage: **L1 25, L2 42, L3 23, L4 12**.

`submission/real_analysis_final.ipynb` renders all of it as a read-through
notebook with outputs captured, so the derivations can be read without running
anything.

`submission/template_reviews/` — per-level construct-validity reviews of the
question templates, tentative and final. Several document shortcuts a model
could otherwise have exploited, and the fixes applied.

`submission/run_solve_code.py` re-executes the stored `solve_code` against a
local copy of the release.

## What this establishes, and what it does not

It establishes that the items are **solvable from the released signals**, and it
makes every judgement auditable: a reader who doubts an answer can read the
derivation and run the code.

It is **not** a measurement of human accuracy. The expert worked with a
purpose-built analysis toolkit and no time limit, and these are verified ground
truth for the items rather than an attempt scored against a separate key. No
human score should be inferred by comparing these items to the model panel.

Items were admitted only after clearing two bars: **real provenance** (question
text, digit-normalised, matching the official generator output byte-for-byte;
hand-authored items excluded regardless of quality) and **no known unfixed
shortcut** (defects fixed, then re-verified blind by an independent solver who
could not see the stored answer). Blind re-verification covers a subset, not the
whole set.

## Not included

The grader-side material — the derivation toolkit with its per-skill heuristics
cheatsheet, and the stored ground-truth key — is deliberately withheld. It leaks
answers and solving shortcuts directly, and nothing in the paper depends on it.

## Raw telemetry

Not included: the parquet telemetry and official splits are multi-GB. Every item
carries the provenance needed to re-derive it from a fresh pull of the release.
Scripts read the data root from `FACTORYBENCH_DATA_ROOT`.
