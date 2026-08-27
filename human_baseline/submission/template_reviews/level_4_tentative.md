# Level 4 — template reviews (TENTATIVE — working copy, iterate here)

## LEVEL 4: DONE, CONFIRMED

Status tags: `[DRAFT]` (not yet discussed) / `[DISCUSSING]` (mid-iteration) / `[APPROVED]`
(frozen, copied to `level_4_final.md`).

### Defect-classification policy (adopted 2026-08-09, applied retroactively below)

Every row in a "Problems spotted" table now carries one of three tags, decided by **what a blind
solver could actually do with the ONE item in front of it**, with no memory of other items and no
prior training on this benchmark:

- **`[BLOCKER]`** — per-item. The defect is readable from, or acts on, that single item: gold text
  factually wrong for its own labeled category, a copy-paste/lettering glitch in that item's own
  gold text, or a channel genuinely absent for that item's provenance making the diagnosis
  *physically* impossible for that item. Aggregate accuracy numbers may be how we *measured* it —
  what matters is that the mechanism is per-item.
- **`[FLAG-ONLY]`** — corpus-level. Only exploitable by knowing the aggregate distribution across
  many items (majority-class guessing, a task-identity-to-answer-class correlation you must see many
  examples to learn). Real dataset-quality issues, worth reporting to a co-author, but they do not block a
  single-item solver. Not shipping blockers.
- **`[POSITIONAL-GRAY-AREA]`** — positional/ordering bias; kept as a flag either way by default.
  **Checked and not applicable anywhere in level 4:** `options={}` for 100% of items in both
  templates (8668/8668 and 1279/1279 in train), so both are genuine free-text with no lettered
  options and no answer ordering to bias. No row carries this tag.

## At-a-glance status (updated 2026-08-09)

| template_id | Type | Status | What's needed |
|---|---|---|---|
| 1 | troubleshooting | **Done** (6 items) | Nothing — MCQ conversion + 3 text fixes applied, sampled, solve_code passes; LLM-judge free-text module flagged to a co-author as a future idea, not built |
| 2 | optimization | **Done** (6 items) | Nothing — render fix (12+6 cols), drop KUKA, MCQ conversion applied, sampled, solve_code passes; 5 new findings (mislabeled channel, temperature leak, render scale error, TCP-fix scope, headline-number integrity concern) flagged to a co-author |

## `template_id=1` — "troubleshooting" — [APPROVED] 2026-08-09 (convert free-text to MCQ over the 27 categories; fix `external_arm_disturbance` gold text; merge the 3-way misconfiguration family; reword the `gripper_release_during_motion` glitch)

**Real example cited:** item `0d6ac7c7-9ed2-4490-85b2-0121a90179d9`, `data/level_4_train.jsonl` (aursad, `loosening_phase`). Population: 10,828 items (8,668 train / 1,067 validation / 1,093 test) across 4 provenance datasets: factorywave 47.8%, aursad 37.8%, factorywave_kuka 8.1%, vorausad 6.3%.

### 1) What it's asking / what understanding it tests
*"Given the sensor stream below, does the machine show signs of anomalous behavior? If yes, identify the most likely root cause and describe the steps you would take to fix it."* No wildcards — identical text for all 10,828 items. Two-stage reasoning: (a) detect whether the window deviates from nominal at all (40.8% are `normal`); (b) map the deviation's signature to 1 of 26 fault categories and prescribe a fix. Different provenance datasets carry genuinely different diagnostic signal (position-tracking vs. torque/force-based).

### 2) Possible answers — free response, not MCQ
`options={}`, `acceptance_bounds=null` — but despite the free-text framing, gold `answer` is one of exactly **27 canned paragraphs, one per root cause, byte-identical regardless of robot/task**. Grading reduces to category matching; the "steps to fix" half is unfalsifiable boilerplate. Sample gold texts: `normal` -> "No anomalous behavior detected..."; `loosening_phase` -> asks to verify task-phase intent, not a fault; `external_arm_disturbance` -> **wrong for its own category** (prescribes the fix for a *misconfiguration*, not an external push); `payload_cog_misconfiguration`/`tcp_frame_misconfiguration`/`payload_misconfiguration` -> near-duplicate texts for 3 nominally distinct categories; `gripper_release_during_motion` -> has a copy-paste glitch (duplicate/out-of-order lettering, a stray internal fragment referencing "policy").

### 3) Data sources available in context
Channel set varies meaningfully by provenance: factorywave (position+speed only, no torque — diagnosis must come from kinematic deviations) and factorywave_kuka (torque, no speed) vs. aursad/vorausad (torque + contact-force channels, where position tracking is near sensor-quantization noise — the real signal lives in torque/force).

### 4) Problems spotted + suggested fix

Held-out classifier (HistGradientBoosting, per-provenance, ~200 summary features of the shown window, model selected on validation, test scored once, split hygiene verified — no episode spans splits):

| Provenance | Majority baseline | Multiclass test acc | Read |
|---|---|---|---|
| aursad | 47.1% | **81.3%** | Genuinely answerable — torque/force signatures separate loosening/normal well |
| factorywave | 46.3% | **58.4%** | Carried by `normal` (93% recall) + a few kinematically loud faults; most fault classes <=25% recall |
| factorywave_kuka | 36.6% | **50.0%** | Modest attribution lift only |
| vorausad | 22.1% | **19.1%** | **Below its own majority baseline** — not answerable as labeled |
| **Weighted overall** | 44.4% | **64.0%** | Lower bound on expert answerability; shape is decisive per-provenance |

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | **[BLOCKER]** Free-text field is a fiction — gold is 27 byte-identical canned paragraphs. Per-item: on any single item, a correct free-text diagnosis is scored against one rigid paragraph; no aggregate knowledge is needed for the item to grade wrong | 100% | **[DECIDED 2026-08-09]** Convert to a real MCQ: render the 27 categories as lettered/shuffled options, drop the free-text framing entirely for our sampled items. (Considered and rejected for now: keep free-text with zero category leakage, grade via an LLM-judge that maps the response to a category — preserves the harder "generate, don't recognize" task, but adds a judge-validation step. Flagging this as a suggested future module for a co-author rather than building it ourselves now.) | Verified: exactly 27 distinct answer strings, 1 per category — clean 1:1 MCQ conversion, no ambiguity |
| 2 | **[BLOCKER]** `external_arm_disturbance` gold text prescribes the *misconfiguration* fix — factually wrong for its own labeled root cause; a correctly-reasoning expert fails text-grading. Textbook per-item defect: readable from that one item's own gold text | 375 (3.5%) | Rewrite to category-correct text (check for external interference/snagged cable, clear it, inspect links — only fall back to config-check if the workspace is confirmed clear) | Defect verified byte-level; **this is the one critical defect by the severity bar** — wrong gold blocks a correct expert |
| 3 | **[BLOCKER]** `tcp_frame_misconfiguration`/`payload_cog_misconfiguration`/`payload_misconfiguration` gold texts are near-duplicates of each other; categories are physically indistinguishable (known config-family wall) AND share one remediation. Per-item physical impossibility (same mechanism as `template_id=2` #2 — the discriminating model-side signal isn't in the item), on a small population | 33 (0.3%) | Merge into one category with one correct combined text | Classifier confirms the wall directly: exact accuracy 1/13 on this family. Blocking *mechanism*, but 0.3% population — fails the population half of the severity bar, so still not "critical" |
| 4 | **[BLOCKER]** Copy-paste glitch in `gripper_release_during_motion` gold text (duplicate lettering, stray internal "policy" fragment) — a defect in that item's own gold text | 144 (1.3%) | Reword the affected steps | Verified byte-level in all 144 items |
| 5 | **[FLAG-ONLY]** *(re-tagged 2026-08-09 under the new policy — was treated as a blocker)* vorausad slice looks unanswerable as labeled — held-out classification *below* majority baseline on both detection and attribution | 681 (6.3%) | Relabel to coarse categories, or drop the slice — but re-measure first (see right) | Confirmed with multiple model families/feature sets — not an underfit artifact. **But the mechanism is corpus-level, not per-item:** (a) no channel is missing that would make a vorausad item physically undiagnosable — vorausad renders `effort_target_torque` + `feedback_pos`/`feedback_speed`/`setpoint_pos`, and its sibling `aursad`, sharing most of the same fault taxonomy, reaches 81.3% off the same torque family (aursad's only extra channel is `est_contact_force`); (b) the evidence is an aggregate accuracy vs. a *corpus-wide majority baseline* — 11 classes over only 681 items (~545 train / ~68 test), where a 19.1%-vs-22.1% gap is inside sampling noise and is the classic signature of too few examples per class, not of a per-item information void. A solver shown one vorausad item is not provably blocked. Recommend re-measuring with balanced/repeated CV before deciding relabel-vs-drop |

**Net verdict — UPDATED 2026-08-09, changed by the new defect-classification policy.** The
template's core design is sound for its two biggest slices (aursad strongly answerable at 81%,
factorywave answerable for normal-vs-loud-fault). All required fixes are text/label-level, no new
data collection. **Shipping blockers (all per-item): switch to category-match grading (#1), fix the
`external_arm_disturbance` gold text (#2 — the one critical item), merge the 3-way misconfiguration
family (#3), fix the gripper-release lettering (#4).** *Change vs. the previous verdict:* the
vorausad slice (#5) is **no longer a blocker** — its evidence is a corpus-level statistic (below a
corpus-wide majority baseline on an ~68-item, 11-class test slice), not a demonstration that any
single vorausad item is undiagnosable, and vorausad ships the torque channels its sibling aursad
solves the same faults from. Downgraded to a flag for a co-author: worth re-measuring and possibly
relabelling to coarse categories, but it does not hold up the template. Net effect: 4 blockers
instead of 5, and the blocker set is now entirely text/label edits with no drop-the-slice decision
in the critical path.

## `template_id=2` — "optimization" — [APPROVED] 2026-08-09 (render 12+6 extra columns for factorywave — see #9 for the tool-pose addition; drop KUKA 426 items, confirmed genuinely unbreakable; rename the mislabeled `effort_target_torque` alias on KUKA items; convert free-text to MCQ — 7 valid options on factorywave, not 14, since 7 of the 14 shipped strings are KUKA-only and KUKA is dropped)

**Real examples cited:** `b6695746-0818-4b03-ad05-80e78fe72c65` (factorywave, payload mass), `a7e0920c-2a5e-4bcb-9e41-364a94d4d850` (factorywave, CoG), `742c8c29-6915-4e62-8d5d-95a322109db6` (factorywave, TCP), `dfdbd68e-efdb-49bf-bf31-ae0082312dda` / `7258d61e-9551-4f8f-b1f6-62e159f0f251` (KUKA mass/CoG). Population: 1601 items (factorywave 1175, factorywave_kuka 426).

### 1) What it's asking / what understanding it tests
*"An engineer wants to increase the effectiveness and accuracy of this machine. Based on the sensor stream below, what operational changes or parameter adjustments could help achieve this?"* No wildcards. From a ~5s window, diagnose which of 3 controller configuration errors is present (wrong payload mass / wrong payload center-of-gravity / wrong TCP frame offset) and prescribe the fix. The real payload is always 1.5kg — only the controller's internal *model* of it is wrong.

### 2) Possible answers — free response, not MCQ
14 canned answer strings total (category-match grading, same as the L4 sibling). **Payload mass misconfig** (1095 items): controller's gravity compensation over/under-predicts a vertical load. **CoG misconfig** (356 items): gravity magnitude is right but its lever arm is wrong — phantom lateral force/torque instead of vertical. **TCP frame misconfig** (150 items, factorywave only): a purely kinematic bookkeeping error with no dynamic consequence at all.

### 3) Data sources available in context
Rendered: position/speed/setpoint only (factorywave) or torque+position (KUKA) — all **actual-side** channels. **Available in the raw parquets but never rendered**: for factorywave, `joint_current`/`target_joint_current` (the controller's own *predicted* current) and model-based `tcp_force`/`tcp_torque` — these are **model-side** channels, and that distinction is the whole story (see #4). KUKA's raw parquet has no equivalent model-side channel at all.

### 4) Problems spotted + suggested fix

**Why the wall existed, now mechanistically understood:** a misconfiguration only changes the controller's internal model, never the real physics — actual-side telemetry (position, speed, total torque) must reflect the true 1.5kg payload identically regardless of what's misconfigured, because the position loop silently re-splits effort to compensate. The signal only exists in *model-side* channels (the controller's own predictions), whose bias against reality directly encodes the error. Confirmed with a physics check, not just accuracy: mean phantom TCP force scales linearly with the configured mass error exactly as gravity predicts (~9.8 N per kg of error).

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | **[BLOCKER]** Rendered channels are all actual-side — physically unanswerable as shipped. Per-item: for any single item, the model-side channel that carries the entire signal is simply not in that item's render | 100% | **[DECIDED 2026-08-09] Render 12 extra columns already sitting in the raw parquets** (joint current + target joint current, optionally TCP force/torque) — no new data collection | **0.9825 held-out vs 0.7325 majority** — wall broken. Single-channel-family ablations still strong (0.85-0.88) |
| 2 | **[BLOCKER]** *(re-investigated and confirmed 2026-08-09)* KUKA slice has no model-side channel. `motor_torque_0-5` exists in `data/kuka_signals.parquet` and was newly tested — decisively confirmed **actual-side**, not model-side: it is measured motor current times a *fixed*, payload-independent per-joint gain (episode-to-episode gain spread 0.22-0.65%). Direct test on fault 23 (mass configured to 0/0.5/1/7kg, true payload constant): pose-conditioned residual vs. configured mass is +0.021 Nm/kg (95% CI [-0.213, +0.243]) against a -5.07 Nm/kg model-side prediction — under 0.5% of what a real signal would show. Decisive confirmation: the 31 miskeyed CoG items (see #3) are statistically indistinguishable from the 178 genuinely-misconfigured ones on this channel (a model-side channel would show the 31 sitting at exactly zero error; it doesn't) | 426 (26.6%) | **Drop the KUKA items from this template** — genuinely unbreakable with what exists, now confirmed by direct intervention testing rather than accuracy alone | Exhaustive negative, now mechanistically explained: fitting torque/current features alone reaches only 0.557±0.055 (vs 0.509 majority) — noise. A naive full-episode classifier reaches 0.734, but this is a corpus-level leak (see #6 below), not a real signal — collapses back to 0.568 once the leak is projected out |
| 3 | **[BLOCKER]** 31 KUKA CoG items have the "misconfigured" value exactly equal to the correct value — no fault actually exists, but a fault answer is still expected. Per-item: that item's own gold is wrong for that item's own physics | 31 items (15% of KUKA CoG) | **Moot — folded into #2.** KUKA is dropped from the template entirely, so these 31 items are dropped along with the rest of the slice; no separate regeneration needed | Verified against real episode metadata; independently re-confirmed during the #2 investigation (exactly 31 found again via an unrelated statistical test) |
| 4 | **[BLOCKER]** Free-text field is a fiction (14 canned strings) — per-item, exactly as `template_id=1` #1: a correct free-text answer on a single item is graded against one rigid canned string | 100% | **[DECIDED 2026-08-09]** Same as `template_id=1`: convert to MCQ over the 14 categories for our sampled items; flag the free-text + LLM-judge-classifier module to a co-author as a future idea rather than building it now | 14 distinct answer strings verified; `options={}` on 1279/1279 items, so the free-text framing is real |
| 5 | **[FLAG-ONLY]** *(split out of old row 4 and re-tagged 2026-08-09)* Task identity leaks the answer category on factorywave (one task type = 100% mass-class) | 468 items | Report within-task numbers; the leak becomes harmless once #1 makes the task genuinely solvable | Within-task accuracy still 0.98+ after the fix. **Corpus-level by construction:** `provenance` ships no `task` field at all (keys are dataset/episode/subseries_start_index/subseries_length/relevance), so a solver must (a) infer the task from telemetry and (b) already know the corpus-wide task→class correlation — neither is available from the one item in front of it |
| 6 | **[BLOCKER, NEW 2026-08-09]** KUKA's rendered `motor_torque` channel is displayed under the acronym/legend name `effort_target_torque` — the same name used elsewhere in this dataset (UR/vorausad) for a genuine model-side/commanded-torque channel. Per-item: an expert reading that item's own legend is told they have the controller's commanded torque (the model-side quantity that would make the item solvable) and will confidently reason to a wrong answer from data that is really just relabeled measured current. Traced to a shared generation script (`archived_regen_data/level_2/predictive/t6_regen.py:54`), so the same mislabeling likely affects other templates drawing on `kuka_signals.parquet`, not just this one | 426 KUKA items in this template; broader blast radius across other KUKA-sourced templates not yet quantified | Rename the rendered acronym for KUKA items from `effort_target_torque`/`ett*` to something that doesn't imply a model-side quantity (e.g. `motor_torque`/`estimated_joint_torque`), independent of whether the KUKA slice is dropped here — the mislabeling is worse than simply omitting the channel, and needs fixing at the shared script level | Confirmed via the same proportionality test as #2: `motor_torque = K_j * motor_current`, fixed gain, no payload dependence — the "target" name is false regardless of what template renders it |
| 7 | **[FLAG-ONLY, NEW 2026-08-09]** A real corpus-level trap for future audits, not exploitable by a single blind item: the two KUKA misconfiguration classes (mass vs. CoG) were recorded in separate contiguous time blocks (3 class switches across 425 consecutive `created_at` transitions). Motor temperature (unrendered) drifts monotonically within a session and nearly perfectly separates the classes (0.960-1.000 held-out) for reasons having nothing to do with payload physics; motor current partially inherits this drift (~19% of variance), which is what produces the misleading 0.734 held-out figure in #2 before the leak is projected out | 426/426 KUKA items, corpus-structural | Randomize/interleave recording order for future KUKA regens; document the leak so nobody mistakes a temperature-drift classifier for a broken wall later | Mechanistically confirmed: projecting out the temperature component collapses the 0.734 figure back to 0.568 (noise-level), matching #2's direct intervention test |
| 8 | **[FLAG-ONLY, NEW 2026-08-09, found while sampling]** A uniform rendering scale error: for 707/1175 factorywave items (those backed by `ur_signals.parquet` or `ur_screwdriver_signals.parquet` rather than `ur_signals_10hz.parquet`), shipped `feedback_pos`/`setpoint_pos` values equal the raw parquet values times a constant 0.988557 (a 1.16% uniform scale error on every joint). Items backed by `ur_signals_10hz.parquet` (468/1175) match the raw parquet exactly. Not large enough to change any option's correctness on its own, but worth fixing at the rendering level | 707/1175 factorywave items (60.2%) | Apply the correction factor at render time, or re-render directly from `ur_signals_10hz.parquet`-equivalent resampling for the other two sources | Confirmed via direct parquet comparison; sampled items account for it explicitly in `solve_code` (0.02° tolerance on `ur_signals_10hz`-backed items, 1.2° tolerance on the two affected sources) |
| 9 | **[BLOCKER, NEW 2026-08-09, correction to fix #1's scope]** Fix #1's "render 12 extra columns" is **incomplete for the TCP-frame category specifically**. The 12 current/force/torque columns carry zero dynamic signature for a TCP-frame misconfiguration (confirmed: on a TCP item, the current-residual payload-model inversion returns the *correct* payload, i.e. no phantom signal at all) — TCP frame is a purely kinematic bookkeeping error, so it can only be identified from the controller-reported tool pose (`tcp_x/y/z/rx/ry/rz`), which fix #1 as originally scoped does not include | 150 TCP-frame items (all factorywave) | Add the 6 tool-pose columns to the render list alongside the original 12 (18 total), specifically to make TCP-frame items solvable | Confirmed identifiable via matched-pose differencing against reference episodes once tool-pose columns are included: derived offset `[+1.999, +0.004, +54.999]` mm against a true `[2,0,55]` mm configuration, ~500σ on the x-axis |
| 10 | **[FLAG-ONLY, NEW 2026-08-09, integrity concern on fix #1's headline number]** The originally reported 0.9825 held-out figure for fix #1 may not be measuring what it claims. A rigorous re-derivation using genuine payload physics (matched-pose inversion of the controller's feedforward model, calibrated against *reference* episodes with known configured payloads) only reaches ~0.67 accuracy from marginal window statistics alone — not enough to distinguish adjacent 0.5kg-spaced mass options. The classifier behind 0.9825 likely rides a corpus-level correlation between trajectory/task family and configured mass, rather than the intended physical signal. This does not invalidate the render fix itself (real payload signal genuinely exists and is usable — 100% accuracy achieved via matched-pose inversion on `pick_and_place`, >=10σ separation on `peg_in_hole`), but the headline number needs re-measurement with task-family-disjoint splits before being quoted again. Also newly discovered: the `screwing` task's windows are ill-conditioned for this method (pose diversity too low; mass estimates scatter -0.45 to 2.58 for true values 0.0-2.0) and were excluded from sampling entirely | Headline number affects all 1175 factorywave items; the `screwing` subset (347, 29.5%) may need separate treatment or exclusion | Re-measure fix #1's held-out accuracy with episode/task-family-disjoint cross-validation before re-quoting 0.9825; investigate whether `screwing`-task items are admissible under any method | Sampled items were solved via the rigorous matched-pose method, not the original classifier, so the 6 admitted items are unaffected by this concern — it's a concern about the template-wide headline number, not about any single admitted item |

**Net verdict:** the strongest fix-success story in level 4 — the "unanswerable" wall was real for
the channels currently shown, but for 73.4% of the template the breaking data was sitting in the
shipped raw files the whole time; the fix is a one-line render-list change. The KUKA 26.6% is a
genuine, well-understood negative, now confirmed by direct intervention testing (not just
accuracy) — dropped from the template rather than shipping a guessing-only quarter of it.
**Two new findings surfaced during the KUKA investigation, both added 2026-08-09:** a mislabeled
channel name (#6) that's arguably worse than a missing channel — it actively misleads an expert
reader — and a temperature/acquisition-order corpus leak (#7) that's a trap for future audits, not
a per-item defect. Neither changes the drop-KUKA verdict; #6 should be fixed at the generation
level regardless since it likely affects other KUKA-sourced templates too. **Unchanged by the new
defect-classification policy (2026-08-09):** #1, #2 and #3 are per-item physical/gold defects and
stay blockers — #2 in particular is measured in aggregate accuracy but caused per-item (the channel
does not carry payload information for any KUKA episode, period). The only downgrade is the
factorywave task-identity leak, split out as #5 and now `[FLAG-ONLY]`; it was already judged
harmless post-fix, so the verdict does not move.

**Three more findings surfaced while sampling factorywave items, added 2026-08-09 (#8-#10):**
a small (1.16%) uniform rendering scale error affecting 60% of factorywave items (#8,
flag-only, doesn't flip any option); fix #1 as originally scoped is **incomplete** for the
TCP-frame category, which has zero dynamic signature and needs 6 tool-pose columns added
alongside the original 12 (#9, blocker — now fixed in the sampled items, needs the same
correction applied at the generation level); and an integrity concern on fix #1's own headline
number — 0.9825 held-out may be riding a task/trajectory-family correlation with configured mass
rather than genuine payload physics, since marginal window statistics alone only reach ~0.67
(#10, flag-only — the underlying signal is real and usable via matched-pose inversion against
reference episodes, confirmed at 100% on `pick_and_place` and >=10σ on `peg_in_hole`, but the
headline number needs re-measurement with task-family-disjoint splits, and the `screwing` task
subset, 29.5% of factorywave, was found ill-conditioned for this method and excluded from
sampling). None of this changes the render-fix verdict — the wall is still broken — but the
*mechanism* and the *scope of what to render* both needed correction, and the template's own
reported accuracy figure should not be re-quoted until re-measured properly.
