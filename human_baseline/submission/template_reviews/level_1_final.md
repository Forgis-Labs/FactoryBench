# Level 1 — template reviews (FINAL — approved only)

## LEVEL 1: DONE, CONFIRMED

All 4 templates (1, 3, 6, 7) approved. 23 real items in `final_submission/sampled_dataset.json`,
every one with real, executed `solve_code` (`run_solve_code.py --check` clean) rendered into
`final_submission/real_analysis_final.ipynb` with captured output.

---

## `template_id=1` — "phase window" — RE-APPROVED 2026-08-09 (fixed, self-approved per standing delegation)

**Resolution of the 2026-08-09 reopening:** the phase-vocabulary mismatch is real (33.0%
overall). Root cause, confirmed by checking `provenance.task` directly: the pick-and-place
phase lexicon ("approach", "lift", "descent to the bin", "release", etc.) gets applied even to
`peg_in_hole`/`screwing`-task episodes it doesn't describe. **Exclusion policy adopted** (extends
problems 1-2 below): also exclude `factorywave` items where `task=peg_in_hole` and
`phase_name in {3,6,7}`, or `task=screwing` and `phase_name in {3,7}` (100% mismatched per
direct check); and `vorausad` `phase_name in {0,2}` (inherited from the confirmed L2-sibling
rate — not independently re-verified at L1 specifically, flag if it matters later).

**Impact on the 6 already-sampled items:** checked all 6 directly against `provenance.task` +
`phase_name`. **5 of 6 are clean** (3x `pick_and_place`, all phase names outside any flagged
bucket; 1x `vorausad` phase 3, outside the {0,2} exclusion). **1 of 6 is contaminated and
dropped**: `0071024b-7cfd-44a7-b541-ffdce85b9b80` (task=`peg_in_hole`, phase_name=7 — exactly
the flagged 100%-mismatch bucket). Its blind-pass result does not save it; the concern is
whether "release of the object" is even the right description of what's being isolated for a
peg-in-hole task, not whether a real phase boundary exists somewhere nearby.

**Current usable bank: 5 items** (was 6). More can be sampled later against the exclusion policy
above to reach quota.

**Our-side mitigation (in addition to the upstream ask below):** every question we ship in this
submission has its phase name cross-checked against the source episode's real `provenance.task`
before inclusion, and is excluded/renamed accordingly if they don't match — we do not rely on
the generator's phase-name lexicon being correct. This is a per-item verification step on our
side, distinct from asking a co-author to fix the generator itself (see `the defect handoff`).

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
| 1 | **[UPDATE 2026-08-09]** `aursad`-provenance items are not solvable: the arm is completely static for the whole shown window (checked 120 items — none exceed 10 deg/s, ~half under 1 deg/s). AURSAD is a screwdriving dataset; its "approach/lift/pre-grasp" phase labels come from screwdriving sub-steps, not arm motion — there's no kinematic signature to find. | 1914/5513 items (~35% of this template) | **Exclude `aursad`-provenance items from this template**, or (bigger ask) have the generator use a screwdriver-relevant signal (e.g. torque/current) for phase boundaries on this provenance instead of arm kinematics | Confirmed unsolvable on 120 sampled items; not yet tried against torque-based reasoning |
| 2 | `factorywave` "approach to the object" phase is degenerate — every one of 369 items has answer `0` | 369/5513 items (~7%) | Exclude this specific phase-name bucket when sampling, or investigate why the generator always places this phase at the window start | Confirmed on full population (369/369) |

**Net verdict:** usable, but NOT uniformly — restrict sampling to `factorywave`/`vorausad` provenance and avoid the "approach to the object" phase bucket. ~58% of the template (non-aursad, non-degenerate-phase) remains cleanly solvable, verified via 6/6 blind-passed real examples.

---

## `template_id=7` — "extrapolation" — APPROVED 2026-08-09

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
| 1 | Dataset was fully re-rolled by the latest regen — all 3 previously-verified items are now orphaned | bank went from 3 to 0 usable | Re-sample and re-verify (done) | 5 new items sampled + blind-verified against the current pull |
| 2 | **Naive "repeat the last value" solves the majority of the template with zero reasoning** — 77.7% overall, 88.8% with the best of 7 naive rules | all 3691 items | Filter sampling to high-activity-at-window-end items with no new command segment starting inside the horizon | On that filtered slice: persistence drops to 46.7%, best-of-naive to 71.4% — genuinely requires real reasoning; ~1-3% of the pool survives both filters |
| 3 | `acceptance_bounds` field contains no actual bounds despite the name — benchmark doesn't define numeric correctness tolerance as shipped | 3691/3691 | Add real min/max (or std/margin, matching sibling templates) to the field | Confirmed empty of bounds on all items; we compute our own margin (`max(2% of answer, 0.75*window std, 1e-3)`) as a stand-in |
| 4 | ~12% duplicated/stale rows — a lower-rate stream forward-filled onto the render grid (worst on `factorywave`) | most affected: factorywave; aursad/vorausad clean (0%) | Render at native rate, or clearly flag interpolated rows | Confirmed via duplicate-row scan; can silently corrupt the effective prediction horizon |

**Net verdict:** usable, but only via careful item selection — the template as a whole is
dominated by a trivial "nothing changes" shortcut. Sampling policy for this submission: only
high-activity-quartile items with no new command segment inside the horizon, blind-verified with
genuine physics-informed reasoning (settle/quantization-floor/transient regime recognition).

---

## `template_id=6` — "robot identity" — APPROVED 2026-08-09

**Real example cited:** item `154d30d4-8f73-4d2d-a0d5-7666c1cf7a77`, `final_submission/raw_by_level/level_1/template_6.json`.

### 1) What it's asking / what understanding it tests
*"What robot does this sensor data originate from? Answer only with the letter of the correct option."* Tests whether the model can fingerprint a robot's identity purely from its motion telemetry — recognizing kinematic signatures specific to a given arm's geometry, not just pattern-matching numbers.

### 2) Possible answers — option by option
**A** — *"Agile Robots Yu 5 Industrial"* — A 6-axis cobot, same general joint topology as B. Correctly picking this (when true) requires a signal that survives topology-sharing with UR3e — now resolved via servo tracking-error dynamics rather than pure kinematics.
**B** — *"Universal Robots UR3e"* — The most common source robot in this dataset. Distinguishing it from C (KUKA) is straightforward via a parallel-axis kinematic invariant, 100% held-out. Distinguishing it from A used to be the hard case — now resolved.
**C** — *"KUKA KR 10 R1100-2"* — Different wrist topology from both A and B, cleanly separable via the parallel-axis invariant.

### 3) Data sources available in context
Only `feedback_pos`/`setpoint_pos` for all 6 joints + timestamp — no gripper or force/torque channel (`"hides": ["robot", "gripper"]`).

### 4) Problems spotted + suggested fix
| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | UR3e vs. Yu5 share cobot topology — pure kinematics can't tell them apart | Every item where the true answer is A or B | **Resolved with a real physical signal**: servo tracking-error structure (lag magnitude + response time between commanded and actual joint motion) — a genuine controller/motor-design property. Rule: predict Yu5 if the tracking-error lag is large, broad (affects most joints), and fast-responding (~8ms cycle vs UR's ~0.5-1ms) | 99.73-100% held-out on this template; 99.87% held-out zero-shot on the untouched L2 sibling. Sole remaining errors are near-fully-static windows with no motion at all |
| 2 | The knowledge graph documents `vorausad` as recorded on a UR5 robot, but the answer key calls it "Agile Robots Yu 5 Industrial" | Every item where the true answer involves this dataset | Flag to a co-author — likely a UR5/Yu5 naming mix-up upstream | Confirmed directly in the dataset's own documentation |

**Net verdict:** usable, on a clean physical basis (99.7-100%, not just "works but impure"). Two
things sent to a co-author regardless: the UR5/Yu5 naming question, and that static (near-zero-motion)
windows are fundamentally unable to carry robot-identity signal in position/speed alone.

---

## `template_id=3` — "pairwise comparison" — APPROVED 2026-08-09

**Real example cited:** item `a4f00e28-ceb2-4c2f-bcee-a5bb8d77c412`, `final_submission/raw_by_level/level_1/template_3.json` (2407 items: 2332 train / 37 validation / 38 test).

### 1) What it's asking / what understanding it tests
*"What changed between the two instances of robotic time series data? Answer only with a 4-letter string using F and T."* Two signal windows from two episodes, judged on 4 axes: robot identity, anomaly-state equality, task identity, phase alignment. No wildcards — every distinct piece of the item is which two real episodes get paired.

### 2) Possible answers — option by option
**A** — *"Those come from different robots."* Solvable — kinematic fingerprint, 100% held-out.
**B** — *"The two robots have different anomalous states..."* Constant False for 92.4% of items (cross-dataset pairs, no real fault). For the 182 same-dataset pairs where it carries signal, solved via classify-then-compare using full domain knowledge (real fault IDs + fault catalog), constraint-filtered to items where the fault's manifestation is genuinely visible.
**C** — *"...different tasks."* Solvable — 100% via task-equivalence classifier.
**D** — *"...same task, but different phases."* Re-keyed from real `task_phase` data instead of the shipped `D = NOT C` rule, which was wrong 13.2-14.4% of the time.

### 3) Data sources available in context
`feedback_pos`/`feedback_speed`/`setpoint_pos` per side. Confirmed upstream but hidden for the 182 same-dataset pairs: TCP force/torque, joint current, gripper channels, per-row fault flag, `task_phase`. KUKA sides carry gripper channels only — no force/current, a hard limitation for anomaly detection there.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | B constant-False for cross-dataset pairs | 2225/2407 (92.4%) | Restrict B-testable sampling to same-dataset pairs | Confirmed via `fault_id=0` on every cross-dataset side |
| 2 | Uniform window sampling misses the fault's actual manifestation | transient faults: window overlaps fault-active rows only 18.8% of the time | Manifestation-aware sampling; require fault-row overlap for transient faults, real motion for persistent faults | Measured exactly from the per-row fault column |
| 3 | State catalog finer than a short window can decide, plus its own data-quality defects (a mismatched dataset tag, a few empty ones) | payload/config-family causes indistinguishable in blind testing | Grade at family level; clean up catalog dataset tags | Family-level grading converts several near-misses |
| 4 | A handful of B's shipped keys are themselves wrong or unverifiable | 3 confirmed mis-keyed + 25 unverifiable, of 182 | Re-key from real episode metadata | Exact against source metadata |
| 5 | KUKA sides have no force/current — anomaly detection impossible by construction | ~half of the 182 same-dataset pairs | Restrict anomalous-side sampling to UR-sourced episodes | Confirmed directly |
| 6 | D tracks C almost perfectly instead of testing real phase alignment | 100% of traceable pool | Re-key D from real `task_phase` instead of `NOT C` | "Always answer NOT C" drops from 100% to 86.8% accurate against the real re-key |
| 7 | `robot_model` field in episode metadata is `None` for a sizeable fraction of factorywave episodes | not yet quantified | Flag to a co-author — genuine upstream metadata gap | Confirmed on several sampled episodes; worked around via the kinematic invariant instead |
| 8 | Undocumented ±300N TCP-force saturation sentinel inflates naive percentile stats | 2687 rows in `ur_signals_10hz` | Document the sentinel value, or clip/flag it in the release | Confirmed: masking it moves p99 force from 419N to 67N |

**Net verdict:** admitted. A/C solvable unchanged. B admitted only via a constraint-filtered
hand-picked subset (manifestation visible, avoiding the indistinguishable-fault cluster,
UR-sourced anomalous sides) using full domain knowledge for our own answer key — not the
blind-LLM-only standard, which stays a harder, separate, disclosed difficulty measurement. D
uses the verified phase re-key. **6 items sampled, solve-code-verified; 2 of the 6 have a
derived answer that disagrees with the shipped label (kept as `benchmark_ground_truth`
alongside our proposed answer, same convention as `template_id=1`).** Pool-level audit (182
traceable pairs): our derivation disagrees with the shipped key on 15.4% overall.
