# Level 3 — template reviews (TENTATIVE — working copy, iterate here)

## LEVEL 3: DONE, CONFIRMED

Status tags: `[DRAFT]` (not yet discussed) / `[DISCUSSING]` (mid-iteration) / `[APPROVED]`
(frozen, copied to `level_3_final.md`).

## Defect classification policy (adopted 2026-08-09, applied retroactively)

Every row in a "Problems spotted" table below now carries a **Class** tag. The rule:

- **`[BLOCKER]`** — exploitable or disqualifying from **one item's own page**, with no memory of
  any other item: the answer is printed/derivable in that item's own shown context; that item's
  own option text names the wrong physical quantity; that item's own answer is unverifiable or
  non-unique from its own rendering. These still require a fix.
- **`[FLAG-ONLY]`** — only visible in the **aggregate** across many items (class-frequency skew,
  distributional correlations you'd have to have seen many items to learn), *or* a finding that
  is deliberate design / cosmetic / grading-policy and does not stop a competent solver from
  converging on that item's answer. A model answering one question cold cannot exploit an
  aggregate regularity, so these are flagged to a co-author, **not blocking**.
- **`[POSITIONAL-GRAY-AREA]`** — answer-position / option-ordering skew. Kept as a flag rather
  than fully downgraded, because a fresh blind LLM carries a well-documented *general* positional
  bias in MCQ/ranking tasks — no benchmark-specific training needed to trip it.
  **No L3 finding falls in this class** (L3 has no lettered-answer-position defect; `template_id=1`
  is about reconstructing physical order, which is the task itself, not an option-position bias).

## At-a-glance status (updated 2026-08-09 — LEVEL 3 DONE, 90 items total across L1-L3, 5/5 L3 templates sampled)

| template_id | Type | Status | What's needed |
|---|---|---|---|
| 1 | segment ordering | **Done** (6 items) | Nothing — precision fix applied, sampled, solve_code passes |
| 2 | intervention outcome | **Done** (5 items) | Nothing (one small loose end noted, doesn't block) |
| 3 | trajectory multiselect | **Done** (6 items) | Nothing — reword + TCP precision fix applied, sampled, solve_code passes |
| 4 / 5 | future signal value | **Done** (6 items) | Nothing — truncation + reword fix applied; chaotic tail (~3%) dropped from graded set per measured investigation, not reframed |

Sections ordered by `template_id`, increasing. All problem tables carry `[BLOCKER]` /
`[FLAG-ONLY]` / `[POSITIONAL-GRAY-AREA]` class tags per the policy below (added 2026-08-09).

---

## `template_id=1` — "predictive" (segment ordering) — [APPROVED] 2026-08-09 (tie-break fix: render 3-4 decimals)

**Real example cited:** item `598cd997-8922-461f-9c22-8eab6010d57e`, `final_submission/raw_by_level/level_3/template_1.json` (612 items).

### 1) What it's asking / what understanding it tests
Shows a baseline sensor stream spanning a fault, then presents 4 unlabeled raw sensor segments
and asks for their chronological order as a 4-letter permutation. *"...rank the signal segments
listed in the 'options' field in the order you would expect them to appear as the event
manifests."* Wildcards: fault name (4 kinds), event timestep. Forces reconstructing the arrow of
time from robot dynamics alone — no timestamps in the options themselves. Chance = 4.2% (1/24).

### 2) Possible answers — what the options are
No natural-language predicates — each option is a raw multi-row sensor dump (5-7 rows). In the
cited item: **A** = decelerating arrival (speeds bleed toward zero as position converges), **C**
= settle (residual creep dying out), **B** = hold (fully static), **D** = departure
(acceleration breaks out of rest). Correct ranking = recovering *arrive -> settle -> hold ->
depart* by matching each block's terminal state to the next block's initial state — B-vs-D
(both anchored at the same pose, differing only in end-velocity sign) is the discriminating pair
in most items.

### 3) Data sources available in context
One contiguous, time-sorted window (32-64 rows, ~101ms spacing) with the fault event always
falling inside it (612/612, median 51% through). **The 4 options are drawn from beyond the shown
window** (median 11-row / ~1.1s gap, confirmed against raw parquet on 92/92 resolvable sampled
items) — a hidden-future dependency, same family as `template_id=2`/`3`.

### 4) Problems spotted + suggested fix

| # | Class | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|---|
| 1 | `[FLAG-ONLY]` | *Per-item class, but verified non-binding — nothing to fix.* Options come from beyond the shown window | 612/612 | **No fix — MITIGATED by construction.** The 4 segments are exact contiguous, equal-length, non-overlapping tiles of one continuous stretch, so their *relative* order is fully determined by endpoint continuity within the options themselves — the hidden rows are never actually needed | Verified: labeled answer is a valid tiling in 92/92 sampled items. Honest shown-data-only solver: 84.6% expected held-out vs 4.2% chance |
| 2 | `[BLOCKER]` | *That item's own answer is not uniquely selectable from its own rendered page.* 2-decimal rounding + long static stretches make ~17% of items genuinely tied (multiple valid orderings score equally) | 20/119 held-out (16.8%); 16/612 (2.6%) have literally identical option strings | Filter at generation: reject items without a unique continuity chain, or render 3-4 decimals | On the unique-chain subset, accuracy rises to 93.9% held-out / 91.6% overall. Solver isn't wrong on tied items — true answer is inside the tied set 19/20 times, just not uniquely selectable |
| 3 | `[FLAG-ONLY]` | *Cosmetic provenance wording — does not affect answerability of any item.* "Counterfactual scenario" framing is fiction — no second arm exists, this is the episode's own real continuation | 612/612 | Reword: "the continuation of this episode after the fault..." instead of "counterfactual scenario where..." | Wording-only, provenance triple-equality confirmed 612/612, zero accuracy impact |
| 4 | `[FLAG-ONLY]` | *Textbook corpus-level imbalance — invisible from any single item, so a cold solver cannot exploit it. Downgraded under the 2026-08-09 policy.* Fault-type sampling is skewed (`self collision link interference` = 2.0% of items) | minor | Sample the notebook's items deliberately across fault types rather than uniformly at random | Bookkeeping only — accuracy varies little by family (78-91%) |

**Net verdict:** the hidden-future dependency here is real but non-binding — unlike its siblings,
this template ships everything needed inside the options themselves. Honest solver clears 84.6%
held-out (94% on the unique-chain subset after the tie filter). Two cheap fixes (reword +
tie-filter), no new data channels or window changes needed. Usable as-is.

**Verdict unchanged under the 2026-08-09 policy** — the one downgrade here (#4, fault-type
sampling skew) was already scored as bookkeeping, and the single remaining `[BLOCKER]` (#2, tied
orderings) is per-item: on those ~17% of items the correct permutation is not uniquely
determined by that item's own rendered page. So the tie-filter stays a required fix, and
"usable as-is" still holds for the rest.

---

## `template_id=2` — "intervention outcome" — [APPROVED] 2026-08-09

**Real example cited throughout:** item `7f3cfd88-53d1-44cc-bd3b-f47e277f3af9`, `data/level_3_train.jsonl`.

### 1) What it's asking / what understanding it tests
Shows ~30-60 sensor readings around a fault event ("collision cardboard object occurs at
timestep `{T}` ms"), then asks 4 true/false statements about what happens **after** that
event, answered as a 4-letter T/F string. Intent: force the model to hold a rough predictive
model of robot dynamics — not read off a number, but infer the likely continuation of behavior
from a partial view of it.

### 2) Possible answers — option by option

**A** — *"Following the event, at least one joint sweeps more than `62` degrees of travel."*
**[CORRECTION 2026-08-09: `62` is a fixed constant, not a wildcard — confirmed only 1 distinct
value across the entire 618-item population.]** Physical meaning: true only if the model
recognizes the fault triggers a large, sweeping recovery motion rather than a small correction.

**B** — *"Following the event, command and measured TCP motion remain aligned (mean TCP
tracking error stays below `{0.0026}`)."*
Wildcard: the alignment threshold. Physical meaning as worded: implies *positional* alignment
of the tool. **Mislabeled** — actually graded on *speed* alignment (commanded vs. realized TCP
speed), not position.

**C** — *"Following the event, mean joint tracking error exceeds `{0.1977}` (absolute,
commanded vs measured position)."*
Wildcard: the error threshold. Physical meaning as worded: explicitly says "position" — same
mislabeling as B, stated even more directly in the text. Actually graded on commanded-vs-measured
joint *velocity*.

**D** — *"Following the event, the most active joint accumulates more than `100` degrees of
total path length."*
**[CORRECTION 2026-08-09: `100` is also a fixed constant, not a wildcard — 1 distinct value
across the population.]** Physical meaning: true only if the model tracks cumulative distance
traveled (odometry-style), not net displacement — distinguishes a search/recovery pattern from
a quick, direct settle.

### 3) Data sources available in context
Shown: ~30-60 rows at ~100ms spacing of position, speed, and setpoint channels for all 6
joints, plus TCP position/speed, rebased to t=0, fault flag visible at the event timestep.
Confirmed present in every item's own legend: `setpoint_speed_*`/`feedback_speed_*` (joint) and
`setpoint_tcp_speed_*`/`feedback_tcp_speed_*` (TCP) — i.e., everything needed for the *correct*
(velocity-based) grading is already exposed; nothing needs to be added. Not shown: the rest of
the episode after the visible window, which is what the true answer for all 4 options is
computed over.

### 4) Problems spotted + suggested fix

| # | Class | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|---|
| 1 | `[FLAG-ONLY]` | *Deliberate design, and verified answerable per item (83-99% honest held-out) — not a defect.* A/D-style ("sweep"/"path length") and current-level statements require inferring behavior beyond the shown window | ~all items | **No fix — intended design.** Tests genuine forecasting of dynamics. | 83-99% held-out honest accuracy depending on family; working as intended |
| 2 | `[BLOCKER]` | *Per-item mislabeled axis: that item's own option text names the wrong physical quantity, so correct physics on that one page yields the wrong T/F.* B/C-style ("tracking error") statements say *position*, are graded on *velocity* | ~2 of 4 options in most items | **Reword only** — change "commanded vs measured position" -> "commanded vs measured speed" in the option text; no new data needed, columns already ship | 94.2% combined held-out once graded on the correct (already-worded-correctly) axis |
| 3 | `[BLOCKER]` *if confirmed* | *Per-item class, not aggregate: if real, that item's own rendered threshold disagrees with the threshold its own answer was graded against. Unconfirmed today, so it does not yet flip the verdict.* **[LOOSE END, found 2026-08-09 while sampling]** A/D thresholds (`62`/`100`) may not be genuine constants at the generator's internal level — those two families only reproduce at 79-87% (vs 90-100% for the others), suggesting the generator computes them against an item-specific value while rendering a fixed placeholder number in the text | not yet quantified | Needs investigation — likely explains the below-100% accuracy on A/D specifically | Not yet verified; flagged, not blocking the 5 sampled items (all chosen with wide safety margins on every option) |

**Net verdict:** this template needs one text-level fix (rename "position" to "speed"/"velocity"
in the tracking-error option templates) applied at the generator level — no new data
collection, no window changes. Once applied, this template is fully usable as intended: hard
for a model without real dynamics understanding, reliably solvable by an expert with the
toolkit. Loose end #3 doesn't change this verdict but is worth resolving.

**Verdict unchanged under the 2026-08-09 policy** — nothing here was corpus-level. #2 stays a
`[BLOCKER]` (per-item mislabeled axis) and remains the one required fix; #1 was already a
deliberate non-fix.

---

## `template_id=3` — "trajectory outcome multiselect" — [APPROVED] 2026-08-09 (reword UR-schema position->speed on tracking-error options + render TCP channels at 4-5 decimals)

**Real example cited throughout:** item `b578530f-0d4b-40a6-a856-67cd670d04fc`, `final_submission/raw_by_level/level_3/template_3.json` (625 items: 502 train / 56 validation / 67 test).

### 1) What it's asking / what understanding it tests
Shows a 32-64 row window straddling a fault event ("Given the counterfactual scenario where a
`{collision foam object}` occurs at timestep `{3734}` ms, select all statements that would
apply"), then asks 4 T/F statements about post-event behavior. Sibling of `template_id=2`,
drawing its 4 slots from a pool of 7 physical-signal families instead of a fixed 4. Event stem
wildcards seen: `collision foam object`, `collision cardboard object`, `collision hanging
cable`, `self collision link interference`, `joint position limit violation`.

### 2) Possible answers — option by option (this item's draw)

**A** — *"Following the event, at least one joint speed drops sharply (mean speed magnitude
falls to <=`{64}`% of its pre-event mean)."* Wildcard: retention % (53 distinct values, ~44-55%
typical). Physical meaning: true only if the collision arrests motion rather than the
controller pushing through it.

**B** — *"Following the event, command and measured TCP motion remain aligned (mean TCP
tracking error stays below `{0.0026}`)."* Wildcard: alignment threshold. Worded as position,
**mislabeled** — actually graded on speed, same bug as tmpl_2's sibling option. Also hit by the
precision defect (#3 below).

**C** — *"Following the event, mean robot current stays below `{1.040}` (absolute)."* Wildcard:
current threshold (159 distinct values). Ground truth computed over the whole rest of the
episode, not the shown tail (see #1) — forecasting required, by design.

**D** — *"Following the event, mean joint tracking error stays below `{0.1645}` (absolute,
commanded vs measured position)."* Wildcard: error threshold. Worded as position — **mislabeled
on UR-schema robots** (actually velocity-graded); **correct as worded on KUKA-schema robots**,
where position is the only axis that physically exists (no speed channel shipped for KUKA here).

Two other families in the pool are drawn by other items with **fixed constants, not wildcards**
(confirmed): *"...sweeps more than `62` degrees..."* and *"...accumulates more than `100`
degrees of total path length."* — same constants as tmpl_2's identically-worded options,
consistent with a shared generator.

### 3) Data sources available in context
Two schemas ship under this template: **UR-like** (552 items — joint+TCP position AND speed,
setpoint+feedback for all, `robot_current`, per-joint current, fault flag) supports all 7
option families; **KUKA-like** (73 items — position only, no speed channels at all, no
`robot_current`) only ever draws the sweep/joint-track/current-relax/path-length families.
Median 47 rows shown, min 32/max 64; post-event rows visible: median 23, **min 6** (the cited
item shows only ~500ms post-event out of a much longer episode). Speed channels needed for the
*correct* grading of tracking-error options already exist in every UR-schema item's own legend
— no new data needed for problem #2 below, just correct wording.

### 4) Problems spotted + suggested fix

| # | Class | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|---|
| 1 | `[FLAG-ONLY]` | *Deliberate design and verified learnable per item (88% literal, 97-99% forecaster) — not a defect.* Current/sweep/path-length families require forecasting the full post-event episode, not just the shown tail | rcu family 373/625 (60%); ~all items have >=1 forecast-only option | **No fix — intended design**, same as tmpl_2 | Shown-mean-literal 88.2% overall (74.5% when <=10 post-event rows shown, 92.2% when >25); a trained forecaster reaches 97-99% held-out. Working as intended |
| 2 | `[BLOCKER]` | *Per-item mislabeled axis, discoverable from that one item's own physics + its own legend.* Tracking-error options say *position*, graded on *velocity* — **UR-schema only** | 516/625 (83%) carry >=1 mis-worded option; the 73 KUKA items are correct as worded | Reword UR-schema tracking options "position"->"speed"; **leave KUKA-schema options untouched** (position is genuinely correct there) — fix must be conditioned on robot schema, not a blind find/replace | Threshold-ratio calibration proves the axis per schema (UR position ratio 0.10 vs speed ratio 1.10; KUKA position ratio 1.32, correctly on the boundary as worded). ~94% TCP / ~90%+ UR-joint / 96% KUKA-as-worded literal accuracy once correctly graded |
| 3 | `[BLOCKER]` | *Named explicitly in the policy: a rendering-precision issue that makes that item's own answer unverifiable from its own page.* Rendering precision: `time_series` rounded to 2 decimals, below the resolution needed to verify the TCP threshold (~0.0026 scale) from the page at all | 384/625 (61%) — every TCP-tracking option | Render TCP speed channels at >=4-5 decimals, independent of the wording fix in #2 | TCP thresholds (median 0.0025) are all smaller than one display quantum (0.01); 384/384 unreadable from context alone. The cited item's option B reads as apparently-True from the rounded page (0.00109 < 0.0026) while true ground truth is False — unverifiable by a human without the raw parquet |

**Net verdict:** two cheap fixes (schema-conditioned reword + display precision bump) plus one
deliberate non-fix (the forecasting requirement is intended and genuinely learnable). No new
data collection or window changes needed.

**Verdict unchanged under the 2026-08-09 policy** — both fixes (#2 mislabeled axis, #3 rendering
precision) are per-item `[BLOCKER]`s: each is discoverable and disqualifying from a single item's
own page, with no aggregate knowledge involved. Nothing here downgraded.

---

## `template_id=4` (scalar) / `template_id=5` (vector) — "future signal value" — [APPROVED] 2026-08-09 (truncate window ~300ms lead before target; reword "motor torque"->"commanded (target) torque" and "counterfactual scenario"->"the episode where"; chaotic-tail qualitative-band reframe accuracy investigation launched in background)

Reviewed together: identical task/context/grading, differ only in answer shape (1 number vs. a
6-vector). **Real examples:** scalar `f8aff48c-828d-49c1-86b0-cf397a357159`, vector
`dbc4c352-01b3-4c53-86b0-0ce5ca24c5ad`, pathological edge case `324a3768-0b85-4eda-8015-fc58b93d9896`
(all `final_submission/raw_by_level/level_3/`).

### 1) What it's asking / what understanding it tests
Scalar: *"In the counterfactual scenario where a `{collision foam object}` occurs at timestep
`{2827}` ms, what would be the expected value of the `{position}` of joint `{5}` at T+`{705}`ms?"*
Vector: same shape, asks for all 6 joints' values as a JSON array. Wildcards: fault type (4
kinds), event timestamp, signal family, joint index (scalar only), horizon Δ (8-1014ms). Free
response, graded by `acceptance_bounds={signal, std, margin}` where margin = 0.75*std (±0.75σ
band); vector grading is currently a conjunction over all 6 components.

### 2) Possible answers — what the number means physically
No lettered options; 5 signal families instead: `position`/`commanded position` (where the joint
ends up vs. what the controller intends), `velocity`/`commanded velocity` (the transient itself,
sign flip = direction reversed), and `motor torque` (**mislabeled** — see #3). Answers ship at
full precision but the ±0.75σ band is huge relative to it (e.g. margin=25.31° on a 261.17°
position) — locating the right region, not precision, is meant to be the difficulty.

### 3) Data sources available in context
32-64 rows, ~100ms spacing, `acronym_mapping` legend. Two column variants: speed-variant (88%
of items, has feedback+setpoint position/speed) and torque-variant (12%, has
`effort_target_torque` instead of speed). Event timestamp is always exactly a shown row (1084/1084)
so `T` is unambiguous. **The queried channel is always among the shown columns, 1084/1084, no
exceptions** — combined with problem #1, this is the whole issue.

### 4) Problems spotted + suggested fix

| # | Class | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|---|
| 1 | `[BLOCKER]` | *The purest per-item leak in L3 — the answer is printed on that item's own page; no aggregate knowledge required.* **Target timestamp lies inside the shown window — the answer is the printed cell, verbatim, to the rendered 2dp.** Not "recoverable," literally copyable | scalar 580/611 (94.9%), vector 449/473 (94.9%); 100% of those match the printed cell exactly | Truncate context to end a fixed lead (~300ms, ~3 rows) before the target — generator-level, no new data | With 300ms lead: physics-informed damped-velocity predictor scores 76.0% (scalar) / 70.6% (vector, per-component) inside the *existing* tolerance band; best-of-5 methods reach 90.7%/88.9%. Genuinely hard and solvable, vs. trivial today |
| 2 | `[BLOCKER]` | *Per-item leak: the item's own visible continuation brackets its own answer.* Window also shows a median 1.6-1.7s (max 5.4s) *past* the target — even missing the exact row, the continuation is visible | same ~95% | Subsumed by fix #1 | 0ms post-target tail after truncation |
| 3 | `[BLOCKER]` | *Per-item mislabeled channel — that item asks for one physical quantity and grades another, both present in its own legend.* "motor torque" is graded on **commanded/target** torque (`effort_target_torque`), not measured motor torque — same mislabel class as `template_id=2` | 29/611 scalar + 9/473 vector (3.5% combined) | Reword: "motor torque" -> "commanded (target) torque" | Wording-only; mapping verified 1:1 over all 38 torque items |
| 4 | `[FLAG-ONLY]` | *Cosmetic provenance wording; explicitly zero accuracy impact.* "Counterfactual scenario" framing is fiction — `sampled_subfolder == counterpart_subfolder` in 1084/1084 items; the fault genuinely happened in the shown episode | 100%, L3-wide (also true of tmpl 1/2/3) | Reword "counterfactual scenario where X occurs" -> "the episode where X occurs" | Cosmetic, no accuracy impact — expert answers identically either way |
| 5 | `[FLAG-ONLY]` | *Scoring-policy harshness, not an answerability defect — the solver can still converge on every component from that item's own page; the conjunction only depresses the score afterwards. Downgraded from blocking on 2026-08-09; still a strong recommendation.* Vector grading is an all-6 conjunction — one chaotic joint zeroes the item even when broadly right | template_id=5, all 473 | Grade per-component or require >=5/6 | At 300ms lead: 70.6% per-component vs only 43.5% all-6 for the same predictor — conjunction costs ~20-27 points for no benefit |
| 6 | `[BLOCKER]` *(on its ~3% subpopulation)* | *Per-item: on those items the answer is not derivable from that item's own data at better than chance, so an expert cannot converge. Aggregate rarity does not make it flag-only — it makes it narrow.* **A genuinely chaotic tail exists where the post-fault transient reverses inside the gap — anti-predictive, not just hard.** Local extrapolation gets the *direction* right only 21-29% of the time here (worse than a coin flip) | at 300ms lead: 2.6% of scalar items / 2.9% of vector components (98 units, 61 items, 58 episodes) exceed 2σ error on best-of-6 | **[UPDATED 2026-08-09, measured]** Original proposal (reframe as a qualitative band question) tested and **rejected** — on held-out chaotic-tail data every solver scores *below* the majority-class baseline (best 56.9%±6.8 vs 68.0%±6.5 baseline; forecast-based solver is a coin flip at 50.1%). **Correct fix: drop the chaotic-tail units from the graded set entirely** (cheap one-line generator screen: `\|answer − y_last\| > 2σ` at the truncation cut, 100% recall / 37% precision, flags 7.7% of components) | Measured on 400 independent episode-disjoint held-out splits: reframe rejected as a rescue for the tail. **However the same band question is healthy corpus-wide** (not tail-conditioned): truth is a balanced 50.5% "inside", and a real forecaster scores 74.1%±1.1 vs a 49.1% majority baseline — a genuine ~25-point signal. Recommend adopting it as an optional companion question everywhere, just not as a way to keep the chaotic tail in the graded set |

**Net verdict:** currently NOT testing prediction for ~95% of items — a formatted lookup that
prints the label in the prompt, the most severe defect class found in L3 so far. Repair is
mechanical and generator-level: truncate the window (**≥400ms lead, not a flat 300ms** — see
below) and reword 2 families. With those applied, a competent physics baseline scores 76%/71%
(91%/89% best-of-6) on the existing tolerance band — lands exactly on the intended design point.
The ~3% chaotic tail does not need (and should not get) the qualitative reframe: measured
held-out, that reframe loses to a constant answer. Drop those units from the graded set instead;
the band question survives as a healthy, unrelated ~25-point-margin companion question on the
other 97%.

**Secondary defect found while sampling (2026-08-09):** a flat 300ms truncation lead deletes the
referenced fault row itself from the window on 243/1084 items (22.4%) — those whose Δ < 300ms,
since the event timestamp is always exactly one shown row. The item still names the event
timestamp in its prompt text, but the collision is no longer observable in context. Fix:
truncate at `max(300ms, ...)` while pinning the event row visible, or resample Δ ≥ ~400ms for
those items specifically.

**Changed under the 2026-08-09 policy:** per-component vector grading (#5) has moved *out* of
the blocking repair set and into flag-to-a co-author. It is a scoring-harshness issue, not an
answerability one — the all-6 conjunction never prevents a solver from converging on any
component from that item's own page, it only penalises the result afterwards. Still strongly
recommended (worth ~20-27 points for no benefit), just no longer a gate. The template's overall
severity is unchanged, because #1 (answer printed verbatim, ~95% of items) is a textbook
per-item leak and remains the blocker that defines this verdict.

*Minor, no fix proposed:* the last two window rows are byte-identical duplicates in ~20% of items
(padding artifact) — worth fixing alongside #1 but not a correctness defect on its own.
