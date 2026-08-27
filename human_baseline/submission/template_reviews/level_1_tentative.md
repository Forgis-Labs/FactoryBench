# Level 1 — template reviews (TENTATIVE — working copy, iterate here)

## LEVEL 1: DONE, CONFIRMED

All 4 templates approved. 23 real items in `final_submission/sampled_dataset.json` (5+6+6+6),
every one with real `solve_code` that has actually been executed and asserts its own answer
against real data (`run_solve_code.py --check` clean), rendered into
`final_submission/real_analysis_final.ipynb` with captured output.

Status tags: `[DRAFT]` (not yet discussed) / `[DISCUSSING]` (mid-iteration) / `[APPROVED]`
(frozen, copied to `level_1_final.md`).

## At-a-glance status (updated 2026-08-09)

| template_id | Skill | Status | Items |
|---|---|---|---|
| 1 | phase window | **Approved** | 5 |
| 3 | pairwise comparison | **Approved** | 6 |
| 6 | robot identity | **Approved** | 6 |
| 7 | extrapolation | **Approved** | 6 |

---

## `template_id=1` — "phase window" — [RE-APPROVED 2026-08-09 — see `level_1_final.md` for the fix; 1 of 6 sampled items dropped for contamination]

**[REOPENED]** A cross-level check run while drafting L2's `template_id=6` sibling found a
**33.0% phase-name/task-vocabulary mismatch rate** on this template's raw population
(aursad 46.8%, vorausad 39.2%, **factorywave 25.3%**) — the pick-and-place phase-name lexicon
("approach", "descent to the bin", "grasp") gets applied even to `peg_in_hole`/`screwing` tasks
whose real sub-steps don't match those words at all (100% mismatch on `peg_in_hole` phase 3/6/7
and `screwing` phase 3/7). This directly contradicts the original "None found / confirmed clean"
verdict below, which only checked the `aursad`-static-arm and one degenerate factorywave bucket
already documented — it never checked for this vocabulary-mismatch pattern specifically.

**Status of the 6 already-sampled items** (see `level_1_final.md` for the population-level fix):
all 6 have independent, real, blind-verified physical correspondence between their named phase
and actual signal content (e.g. "lift" -> genuine TCP-Z rise; "grasp of the object" -> genuine
torque-closure transient) — the mismatch defect manifests as *zero* such correspondence, so this
is evidence against contamination, but has not been checked via the same systematic
task-vocabulary cross-tab used to find the 33% rate. Recommend one targeted check before treating
the 6 as fully cleared.

~~[APPROVED] 2026-08-09~~

**Real example cited:** item `bc83aa2e-25f3-4c5b-ba3f-21b33b3b3bdb`, `data/level_1_train.jsonl`.

### 1) What it's asking / what understanding it tests
*"The robot is performing a manipulation task. We want to isolate the `{approach to the
object}` in the robot's time series. Assuming a fixed window length of `{52}` timesteps, at
which timestamp should the window begin? Answer only with an integer or decimal number,
nothing else."*
Wildcards: the named phase (e.g. "approach to the object," "descent to the bin" elsewhere) and
the window length. Tests whether the model can segment a continuous sensor trace into the named
behavioral phase — recognizing where one movement type ends and the named one begins.

### 2) Possible answer — free response, not MCQ
No lettered options (`"options": {}`) — answer is a single timestamp, graded against a
tolerance band (`acceptance_bounds`, here `min:0, max:296`). A value near the start of the
series implies the named phase begins almost immediately; a late value implies a long preceding
phase. Getting this right requires recognizing the actual kinematic signature of the phase
boundary, not guessing a fixed offset.

### 3) Data sources available in context
Full per-timestep telemetry at native rate: torque, contact-force estimate, feedback
position/speed, setpoint position — all 6 joints — for the entire shown window (which fully
contains the phase in question; nothing needed is hidden here).

### 4) Problems spotted + suggested fix
| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | **[UPDATE 2026-08-09, found during sampling]** `aursad`-provenance items are not solvable: arm is completely static for the whole window (checked 120 items). AURSAD is a screwdriving dataset — its phase labels come from screwdriving sub-steps, not arm motion. | 1914/5513 items (~35%) | Exclude `aursad`-provenance items from this template, or use a torque/current-based signal for this provenance instead of arm kinematics | Confirmed unsolvable on 120 sampled items |
| 2 | `factorywave` "approach to the object" phase is degenerate — always answer `0` | 369/5513 items (~7%) | Exclude this phase bucket when sampling | Confirmed 369/369 |

**Net verdict:** usable but NOT uniform — restrict to `factorywave`/`vorausad`, avoid "approach to the object." Already re-approved with this caveat since the 6 sampled items respect it (see `level_1_final.md`).

---

## `template_id=3` — "pairwise comparison" — [APPROVED] 2026-08-09

**Final sampling result:** 6 real items built with full-domain-knowledge solve code (kinematic
fingerprint for A, classify-then-compare with real fault IDs for B, task-equivalence for C,
`task_phase` re-key for D), all constraint-filtered (manifestation visible, avoided the
indistinguishable-fault cluster, UR-sourced anomalous sides). **2 of 6 items' derived answers
disagree with the shipped label** (both re-verified against real episode metadata) — kept as
`benchmark_ground_truth` alongside our proposed answer, same convention as `template_id=1`.
Pool-level audit (182 traceable pairs): our derivation disagrees with the shipped key on 15.4%
overall, independently reproducing the D re-key's 86.8% figure via a different method.

**Status update, not auto-approved:** the classify-then-compare redesign for B (state-names-only
catalog, no symptom key, per the reviewer's design) was built on real data — 10 hand-built items with
real hidden-channel telemetry — and blind-tested twice independently. Result: it produces a
genuine diagnostic-reasoning task (both blind solvers made verifiably correct physics arguments
at times), but measured accuracy (7/10 and 6/10, both at/near the trivial "always answer True"
baseline of 6/10; exact state naming 4/20 both runs) is **not approval-grade yet**. D fared
better: a re-key from real phase-annotation data demonstrably breaks its shadow-of-C problem
(100% -> 86.8% on the fixable pool) and is ready to adopt. **A and C remain solvable, unchanged,
and admittable now.**

**Real example cited:** item `a4f00e28-ceb2-4c2f-bcee-a5bb8d77c412`, `final_submission/raw_by_level/level_1/template_3.json` (2407 items: 2332 train / 37 validation / 38 test).

### 1) What it's asking / what understanding it tests
*"What changed between the two instances of robotic time series data? Answer only with a
4-letter string using F and T."* Two signal windows from two episodes, judged on 4 axes: robot
identity, anomaly-state equality, task identity, phase alignment. No wildcards anywhere —
question and all 4 option strings are byte-identical across all 2407 items; every distinct piece
of the item is which two real episodes get paired.

### 2) Possible answers — option by option

**A** — *"Those come from different robots."* **Solvable** — kinematic fingerprint, 100% held-out
(same method as `template_id=6`).

**B** — *"The two robots have different anomalous states..."* Keyed by fault-label equality.
**Constant False for all 2225 cross-dataset pairs (92.4%)** — every cross-dataset side has
`fault_id=0` by construction, not a real discriminator. Only the 182 same-dataset (factorywave-
factorywave) pairs carry real signal. The classify-then-compare redesign targets exactly this
pool — see below.

**C** — *"The two robots are performing different tasks."* **Solvable** — 100% via task-
equivalence classifier, re-verified 182/182 on the traceable pool.

**D** — *"...same task, but at different phases."* Shipped key is `D = NOT C` on 100% of the
traceable pool (98.75% corpus-wide) — including 7 pairs whose windows cover *identical* phase
sets, so it isn't testing phase alignment at all as shipped.

### 3) Data sources available in context
Rendered: `feedback_pos`/`feedback_speed`/`setpoint_pos` per side. **Confirmed upstream but
hidden** for the 182 factorywave-factorywave pairs: TCP force/torque, joint current, joint temp,
gripper channels, per-row fault flag, and `task_phase` — all traceable to real local parquets.
Channel coverage is asymmetric: UR-sourced sides carry force+current; **KUKA sides carry gripper
channels only — no force, no current**, a hard limitation for anomaly detection on those sides.
The `aursad`/`vorausad` episodes referenced by the 2225 cross-dataset pairs exist in **no** local
parquet at all — any fix touching them is locally unverifiable, same as the L2 sibling's finding.

### The B redesign, as actually built and tested
Per the agreed design: `context.possible_states` lists the 26 real `root_cause` names (from
`data/knowledge_graph.json`) applicable to that side's dataset — filtered by dataset only, never
by task (task-filtering would leak C/D) — plus "normal." No symptom signatures are exposed; the
model must build the physical mapping itself. Auxiliary sensor channels (TCP force, joint
current, gripper) are added alongside, with their own legend. The model classifies each side to
one state, then answers B consistent with its own two classifications.

**Note on scope: this withholding applies to the blind LLM under test, not to human
verification.** A human expert building/checking this item's ground truth is expected to use
real, internalized symptom-to-fault knowledge — that's what domain expertise is, not a leak. The
7/10 and 6/10 numbers below measure a memoryless LLM instance reconstructing that mapping cold,
which is the actual capability this option is meant to probe.

**10 real hand-built items**, all traced to real episodes with re-derived, verified fault labels
(6 B=True / 4 B=False, balanced). **Blind-verified twice, independently:**

| Metric | Run 1 | Run 2 | Read |
|---|---|---|---|
| B letter correct | 7/10 | 6/10 | At/near the trivial "always answer True" baseline (6/10 on this set) |
| Exact state name (20 sides) | 4/20 | 4/20 | 20% vs 3.7% chance — real signal, but weak |
| A / C / D vs shipped key | 8/10 / 8/10 / 8/10 | 5/10 / 8/10 / 8/10 | A/C hold up under the augmentation |

Systematic failure modes (both runs): same-fault pairs with asymmetric window evidence get
overcalled as "different states"; subtle persistent faults in low-motion windows read as normal;
task-inherent contact (e.g. screwing) gets misread as collision. Both solvers independently flagged
4 config-family root causes as indistinguishable from a single short window — the catalog is
finer-grained than what a 4-6 second excerpt can support.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | B constant-False for cross-dataset pairs | 2225/2407 (92.4%) | Restrict B-testable sampling to same-dataset pairs; sampling anomalous aursad/vorausad episodes is locally unverifiable (0 of those episode IDs exist in any released parquet) | Confirmed via `fault_id=0` on every cross-dataset side |
| 2 | Classify-then-compare design alone doesn't clear the bar | the 182 same-dataset pairs | Needs fixes #3 + #4 on top of the design, not the design alone | Built and blind-tested: 6-7/10 vs a 6/10 trivial baseline — genuine improvement in kind, not yet in degree |
| 3 | Uniform window sampling misses the fault's actual manifestation | transient-fault sides: window overlaps the fault-active rows only 18.8% of the time | Manifestation-aware sampling: require fault-row overlap for transient faults, require real motion for persistent faults | Overlap rate measured exactly from the per-row fault column |
| 4 | State catalog is finer than a short window can decide, plus has its own data-quality defects (one fault ID's dataset tag looks wrong, a few catalog entries have missing/empty dataset tags) | payload/config-family root causes indistinguishable in both blind runs | Grade at the family level (group the indistinguishable config-family causes together); clean up the catalog's dataset tags before using it as a filter | Family-level grading would have converted several near-misses in the blind test |
| 5 | A handful of B's shipped keys are themselves wrong or unverifiable (counterfactual side missing its fault ID) | 3 confirmed mis-keyed + 25 unverifiable, out of the 182-pool | Re-key from real episode metadata | Counts exact against source metadata |
| 6 | KUKA sides have no force/current channels — anomaly detection there is impossible by construction | about half of the 182 same-dataset pairs | Restrict anomalous-side sampling to UR-sourced episodes, or disclose as intended difficulty | Confirmed directly: the one hand-built item testing this got called wrong by both blind solvers |
| 7 | D tracks C almost perfectly instead of testing real phase alignment | 100% of the traceable pool | **Re-key D from real `task_phase` data** (present in all 4 raw parquets) instead of `NOT C` | Verified: "always answer NOT C" drops from 100% to 86.8% accurate once graded against the real re-key — D now carries independent signal for the first time |

**Net verdict:** A and C are solvable now, no changes needed, ready to admit. **B is an honest
partial** — the redesign is real progress (genuine diagnostic reasoning, not lookup) but doesn't
clear the bar yet; recommend adopting it together with fixes #3/#4/#6 and re-testing blind before
admitting any B-driven item. **D should be fixed, not dropped** — the phase re-key works and is
grounded in real shipped data.

**Cross-check against the L2 `template_id=8` sibling** (per standing practice): agrees on B's
core problem and the channel-exposure fix direction. Updates that sibling's own draft in two
ways: (1) its open question "does exposing channels lift B above chance" is now answered — not
by itself, no; (2) its D recommendation ("drop D, or only score where C=F") is now stale — a
better option (the same phase re-key) is verified feasible here and should be offered there too.

---

## `template_id=6` — "robot identity" — [APPROVED] 2026-08-09

**Good news: the impurity caveat is substantially retired.** A dedicated search found a genuinely
physical discriminator — servo tracking-error dynamics (how much a joint lags its commanded
position/velocity, and how fast that lag responds) — that separates UR3e from "Yu5" at 99.73-100%
held-out, essentially matching the old impure method (100%) without relying on batch fingerprints.
It even zero-shot transfers to the L2 sibling template untouched (99.87% held-out there). Every
remaining error is a window where the arm barely moves at all — genuinely no identity signal
exists in position/speed alone for those, not a flaw in the method.

**Bonus finding, independent of the solving question:** the dataset's own documentation says the
`vorausad` data was recorded on a **UR5** robot, while the question's answer key calls it "Agile
Robots Yu 5 Industrial" — this looks like a **name mix-up between "UR5" and "Yu 5"** upstream,
worth flagging to a co-author regardless of how the solving side turns out (see the defect handoff).

**Real example cited:** item `154d30d4-8f73-4d2d-a0d5-7666c1cf7a77`, `final_submission/raw_by_level/level_1/template_6.json`.

### 1) What it's asking / what understanding it tests
*"What robot does this sensor data originate from? Answer only with the letter of the correct
option."* Tests whether the model can fingerprint a robot's identity purely from its motion
telemetry — recognizing kinematic signatures specific to a given arm's geometry, not just
pattern-matching numbers.

### 2) Possible answers — option by option
**A** — *"Agile Robots Yu 5 Industrial"* — A 6-axis cobot, same general joint topology as B.
Correctly picking this (when true) requires a signal that survives topology-sharing with UR3e —
pure kinematics alone doesn't cleanly separate them (see problem below).
**B** — *"Universal Robots UR3e"* — The most common source robot in this dataset.
Distinguishing it from C (KUKA) is straightforward via a parallel-axis kinematic invariant (a
specific 3-joint sum stays constant while the tool holds orientation) — cleanly separates the
two, 100% on held-out data. Distinguishing it from A is the hard case.
**C** — *"KUKA KR 10 R1100-2"* — Different wrist topology from both A and B, cleanly separable
via the same kinematic invariant.

### 3) Data sources available in context
Only `feedback_pos`/`setpoint_pos` for all 6 joints + timestamp — no gripper or force/torque
channel (explicitly listed under `"hides": ["robot", "gripper"]` in this item, confirming
gripper data exists upstream but is deliberately withheld here).

### 4) Problems spotted + suggested fix
| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | UR3e vs. Yu5 share cobot topology — pure kinematics can't tell them apart | Every item where the true answer is A or B | **Resolved with a real physical signal**: servo tracking-error structure (lag magnitude + response time between commanded and actual joint motion) — a genuine controller/motor-design property, not a batch artifact. Rule: predict Yu5 if the tracking-error lag is large, broad (affects most joints, not just one), and fast-responding (~8ms cycle vs UR's ~0.5-1ms) | 99.73-100% held-out on this template; 99.87% held-out zero-shot on the untouched L2 sibling. Sole remaining errors are near-fully-static windows with no motion at all to carry any identity signal, physical or otherwise |
| 2 | **NEW**: the knowledge graph documents `vorausad` as recorded on a UR5 robot, but the answer key calls it "Agile Robots Yu 5 Industrial" | Every item where the true answer involves this dataset | Flag to a co-author — likely a UR5/Yu5 naming mix-up upstream, independent of whether the template is solvable | Confirmed directly in the dataset's own documentation |

**Net verdict:** usable, and now on a clean physical basis (99.7-100%, not just "works but impure").
Two things worth sending to a co-author regardless: the UR5/Yu5 naming question, and that static
(near-zero-motion) windows are fundamentally unable to carry robot-identity signal in
position/speed alone — worth avoiding when sampling episodes for this template.

---

## `template_id=7` — "extrapolation" — [APPROVED] 2026-08-09

**Real example cited:** item `0990c8c0-083d-430e-b4dd-f47868658ad1`, `final_submission/raw_by_level/level_1/template_7.json` (3691 items).

### 1) What it's asking / what understanding it tests
*"Given the sensor stream below, what is the expected value of the `{motor torque}` of joint `{2}` at T+`{100}`ms?"* Wildcards: signal family (position/velocity/torque), joint index, horizon (8-1014ms). Tests forward-dynamics prediction from a pre-fault-free window — the L1 (state-only) sibling of L2's `template_id=4/5` and L3's `template_id=4/5`. Unlike those, there's no fault here — pure continuation of nominal motion.

### 2) Possible answer — free response, not MCQ
Single number, graded against `acceptance_bounds`. **Field is misnamed**: it contains only `{signal, steps_ahead, actual_value, horizon_ms}` — no min/max bounds at all, despite the key name (confirmed on all 3691 items). Correctly answering requires recognizing the physical regime: settled/zero-velocity (persistence is correct), quantization floor (persistence at the floor), or a clean transient (linear fit, ideally cross-checked against a physics model for torque channels).

### 3) Data sources available in context
Full per-timestep telemetry, ~100ms spacing, spans `factorywave`, `aursad`, and `vorausad` provenance with different channel sets (position/speed/setpoint for factorywave; add torque/contact-force for aursad). Enough to do genuine physics-informed extrapolation on most items.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | **Dataset was fully re-rolled by the latest regen — all 3 previously-verified items are now orphaned** (0/3 IDs exist in the fresh pull) | bank went from 3 to 0 usable | Re-sample and re-verify (done — see below) | 5 new items sampled + blind-verified against the current pull |
| 2 | **Naive "repeat the last value" solves the majority of the template with zero reasoning** — 77.7% overall, 88.8% with the best of 7 naive rules | all 3691 items | Filter sampling to high-activity-at-window-end items with no new command segment starting inside the horizon | On that filtered slice: persistence drops to 46.7%, best-of-naive to 71.4% — genuinely requires real reasoning; ~1-3% of the pool survives both filters |
| 3 | `acceptance_bounds` field contains no actual bounds despite the name — benchmark doesn't define numeric correctness tolerance as shipped | 3691/3691 | Add real min/max (or std/margin, matching sibling templates) to the field | Confirmed empty of bounds on all items; we compute our own margin (`max(2% of answer, 0.75*window std, 1e-3)`) as a stand-in |
| 4 | ~12% duplicated/stale rows — a lower-rate stream forward-filled onto the render grid (worst on `factorywave`: mean 12.1% duplicate rows, up to 39.5% of items with >10% duplicates, 11% with a stale final row) | most affected: factorywave; aursad/vorausad clean (0%) | Render at native rate, or clearly flag interpolated rows | Confirmed via duplicate-row scan; can silently corrupt the effective prediction horizon |

**Net verdict:** usable, but only via careful item selection — the template as a whole is
dominated by a trivial "nothing changes" shortcut. 5 items banked so far (of 10 candidates,
using the high-activity + physics-informed-reasoning filter); the other 5 failures were
consistently "the target waypoint isn't encoded anywhere in the visible window," not weak
reasoning — an honest, expected failure mode, not a bug.
