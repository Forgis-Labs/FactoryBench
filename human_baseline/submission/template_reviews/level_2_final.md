# Level 2 — template reviews (FINAL — approved only)

## LEVEL 2: DONE, CONFIRMED

All 9 templates resolved: 8 approved with real items (42 total), `template_id=9` approved and
explicitly excluded by design. Every item has real, executed `solve_code`
(`run_solve_code.py --check` clean) rendered into `final_submission/real_analysis_final.ipynb`.

---

## `template_id=1` — "onset ranking" (skill 4) — APPROVED 2026-08-09

**Real examples cited:** `96834b9f-df4a-4b1f-a8fe-d281ee59ae3c` (typical), `85d23d58-35d5-4072-ac3d-beafef89074f` (leak exemplar) — both verified in `final_submission/raw_by_level/level_2/template_1.json`. Population: 722 items.

### 1) What it's asking / what understanding it tests
*"The sensor stream below is from a robot exhibiting `{a collision with a soft foam object}`. Rank the signal segments listed in the 'options' field in the order you would expect them to appear as the anomaly manifests."* One wildcard (fault phrase, 6 kinds). Reconstruct where 4 undated telemetry snippets sit in an anomaly's life cycle (approach -> arrest -> recovery) from physics alone. Ground-truth rule confirmed: strict chronological order within the episode (684/684 pairwise orderings verified).

### 2) Possible answers — option by option
Structurally MCQ (A-D) but graded as an exact 4-letter permutation string, chance = 4.2%, answer distribution confirmed uniform across all 24 permutations. Each option is 5-7 raw timesteps with no timestamps. Distinguishing "fast, force building" from "accelerating, force decaying" segments — both similar peak speeds — is the genuine discriminator; single-statistic solvers cap at 27.6% exact, confirming this is real difficulty, not noise.

### 3) Data sources available in context
35-161 rows, ~101ms spacing, `t=` printed on every row, 30 channels, 4 channel-set variants, all fully documented in a 31-entry legend.

### 4) Problems spotted + suggested fix

| # | Class | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|---|
| 1 | `[BLOCKER]` | New leak: option segments are byte-identical rows already visible in the printed stream — sorting by printed position alone reproduces the answer | 97/722 (13.4%) fully leaked | Sample only from items where segments are confirmed absent from the shown context (the ~87% majority already does this correctly) | String-matching-only scores 100% on the leaked subset vs 4.2% chance; fix verified — all 6 admitted items confirmed leak-free |
| 2 | `[FLAG-ONLY, CORRECTED]` "32/87 overlap" was a measurement artifact (row-level duplicate matching, not real span overlap) — re-measured, 0/456 traceable items genuinely overlap | 0/456 | No fix needed | Tie-break rule (order by first timestamp) implemented defensively in every solve script, never triggers on real data |
| 3 | `[FLAG-ONLY]` Hidden time-skip in the stream — every row's timestamp is printed, so not actually hidden | 96.0% of items | No fix — not a real defect once timestamps are accounted for | — |
| 4 | `[FLAG-ONLY]` Duplicate rows within a segment (resample artifact) | 8.4% of option rows | Cosmetic dedup at render | No item's answer depends on it |
| 5 | `[BLOCKER, found during sampling]` Adjacent segments sometimes share a byte-identical boundary row (10Hz resample duplication) — a zero-physics chaining shortcut | 128/789 seams (16.2%); 99/263 clean items (37.6%) | Dedupe at render, same root cause as #4 but at segment boundaries | Confirmed; not fatal — 2 of 6 admitted items have zero such seams and were still blind-solved correctly from pure physics |

**Net verdict:** usable. The real leak (#1) is fixed by sampling policy; the "overlap" problem
(#2) turned out not to exist; a new subtler shortcut (#5) is disclosed but doesn't block
admission since the task remains genuinely solvable without it. 6 items sampled,
solve-code-verified, blind-verified 6/6.

---

## `template_id=3` — "multi-fault statement check" (skill 8) — APPROVED 2026-08-09

**Real examples cited:** `8fede9bb-15ac-4686-bf0a-057542a4bdf3` (UR), `5f0faab2-01a7-41f8-9d26-e18ec02da587` (KUKA), both `final_submission/raw_by_level/level_2/template_3.json`. Population: 766 items.

> **The old "drop, not repair, confirmed unfixable" verdict no longer holds.** Both headline defects that justified it (channel-set-determines-answer; missing force/current channels) are fixed in the current pull. A different, new bug replaced them — confined to 2 of 4 answer slots, repairable from data already shipped.

### 1) What it's asking / what understanding it tests
*"The sensor stream below is from a robot exhibiting a `{collision with a soft foam object}`. Select all statements that apply. Answer with a 4-letter F/T string."* 4 slots, one option drawn from each per item: {sweep | path-length}, {speed-baseline | KUKA torque+current}, {TCP-aligned | misaligned}, {tracking-error below | above}. Unlike its L3 sibling, the event timestep is never stated — the solver must localize onset from kinematics alone (the safety_mode channel shown is uninformative, constant in 90% of items).

### 2) Possible answers — slot by slot
**Slot A (sweep/path-length)** — honest, verified 97.9%/98.9% replication, but graded over the full episode + all 6 joints (forecasting required, same shape as L3's approved current-family — just less cleanly disclosed).
**Slot B (tracking error)** — the exact L3 `template_id=2` mislabeling bug, reproduced: worded as position, actually graded on velocity (91.8%/90.2% on the correct axis vs 17-23% as worded — an expert taking the text literally answers backwards).
**Slot C (TCP alignment)** — genuinely broken: ~30 candidate formulas tested, best replication 0.52-0.58, *below* the constant-answer baseline (0.61-0.66).
**Slot D (speed baseline)** — broken twice over: thresholds are textually absurd (all between 464-2669% — "within ±1674% of baseline" is trivially always true, yet 63% are labeled False, so the label can't mean what the text says), and irreproducible from ~15 tested formulas (best ≈ chance).

### 3) Data sources available in context
~110-160 rows, ~101ms spacing, verified index-exact against `ur_signals_10hz.parquet` for 471/766 items. Not shown: the per-row fault flag (exists upstream, is the real onset marker), velocity setpoints (slot B's true axis), post-window continuation (slots A/B), joint setpoints in one channel-set. Everything needed for the fixes below is confirmed present in the raw parquets already.

### 4) Problems spotted + suggested fix

| # | Class | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|---|
| 1 | `[BLOCKER]` | Speed-baseline (slot D) labels irreproducible + thresholds nonsensical. Per-item: the threshold printed in that item's own option text ("within ±1674% of baseline") contradicts that item's own label | 698/766 (91%), 1 of 4 letters | Regenerate this option's labels+thresholds from existing signals — a correct formula already exists in-repo (`regen_l2_tmpl2.py`) | Best of ~15 honest formulas = chance; same pipeline replicates slots A/B at 90-99%, so this is a label bug not a harness bug |
| 2 | `[BLOCKER]` | TCP-alignment (slot C) labels irreproducible both polarities — per-item unanswerable, not an aggregate artifact | 698/766 (91%), 1 letter | Regenerate with the in-repo `tcp_track.change_pct` formula; ship TCP channels consistently | Best of ~30 formulas = 0.52-0.58, below the constant-answer floor |
| 3 | `[BLOCKER]` | Tracking-error (slot B) worded as position, graded on velocity — mislabeled axis, discoverable and fatal from that one item's physics | ~97% of items, 1 letter | Reword; add `target_joint_vel_*` to context (confirmed upstream) | 90-92% on velocity axis vs 17-23% as worded |
| 4 | `[FLAG-ONLY]` | Event timestep never stated in the question — localizing onset from kinematics is the intended skill here, so this is a costless consistency improvement, not a convergence blocker | 100% | Print onset timestep in the stem, like the L3 sibling does | Costless; onset = first nonzero upstream fault flag, verified |
| 5 | `[FLAG-ONLY]` | Slots A/B graded on full-episode + unshown joints — labels are sound (97.9-98.9% replication), so this is a disclosure item under the existing forecasting precedent | ~all items | Disclose as forecasting (same as L3 precedent) or extend window | Full-episode replication is clean (97.9-98.9%) — labels are sound, just require the forecasting frame |
| 6 | `[BLOCKER]` | KUKA subpopulation unverifiable (window indices don't map to shipped parquet); some items lack a fault flag or pre-event baseline entirely — per-item: the affected item is missing exactly the evidence it needs | 68 + 38 + 56 items | Ship the resampled KUKA table windows were cut from; drop/re-window flag-less items | Only 7/68 KUKA windows re-alignable |

**Net verdict:** confirmed broken as shipped (honest exact-match capped ~25-30%), but the correct
fix is **label regeneration + reword at the generator level, using data already in the shipped
parquets** — not dropping the template, not new data collection. Slot A needs no fix at all.

**Admission status: approved as a review, but not yet contributing items to the notebook.**
Every item in this template draws one option from all 4 slots simultaneously (not an
independently-optional dimension like some other templates), and slots C/D are broken in ~91% of
items — so there's no way to hand-pick a clean subset the way other partially-broken templates
were handled. Items from this template enter the submission once either (a) a co-author regenerates
slots C/D, or (b) a data-driven forecasting approach (per the reframe applied to the L2
`template_id=2` sibling) proves those slots predictable from the shown window after all.

---

## `template_id=4` / `template_id=5` — "extrapolation" (skill 2a/2b) — [APPROVED] 2026-08-09

**Real examples cited:** `fbba4afb-414d-440e-818a-157787af39fa` (tmpl_4), `bf14c2ab-dd6f-44cd-aa65-18435b9ff0c2` (tmpl_5), both `data/level_2_train.jsonl`. Population: 765 (tmpl_4) + 615 (tmpl_5).

### 1) What it's asking / what understanding it tests
*"...robot exhibiting a collision with a hanging cable. What is the expected value of the velocity of joint `{4}` at T+`{404}`ms?"* (scalar) / same shape asking for a 6-vector. Wildcards: fault type, target channel, joint index (scalar), horizon Δ. Genuine forward-dynamics question: predict the fault-reaction transient a few steps after onset. Verified against raw parquet (62/62): `T` = fault onset, undefined anywhere in the question text as shipped — its only current meaning comes from the leak itself (#1 below).

### 2) Possible answer — free response, not MCQ
Graded against `acceptance_bounds` (std/margin, margin≈0.75σ). Should require modeling how a stiff position-controlled arm reacts to a named collision class. 32 tmpl_4 items have `margin=0` (unpassable by construction — see #4).

### 3) Data sources available in context
~30 channels, ~100.8ms spacing, full legend coverage. **The critical defect: the rendered window includes the post-onset "hidden future" rows it's supposed to be testing prediction of** — `hides` is `[]`.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[BLOCKER]` **Root cause of the leak, fully identified**: hidden-future rows are rebased to the wrong time origin (episode start instead of window start) and never excluded from the render — creating a timestamp discontinuity that points straight at the answer row. Textbook per-item leak: the answer value is literally printed in that item's own context | 83.6% combined fully exploitable (76.3%/75.8% carry the literal answer value) | **One-line renderer fix**: drop rows after the split (or exclude them the way the code apparently intends but never executes) | Post-truncation, best zero-reasoning policy drops to the naive floor (35.9% tmpl_4 / 9.6% tmpl_5) — leak eliminated by construction |
| 2 | `[BLOCKER]` `T` undefined in question text pre-fix — per-item: the question as printed does not define its own reference point, so the item is not well-posed on its own once the leak is closed | 100% | Add alongside #1: state the onset timestep explicitly, like the L3 sibling does | Formula matches parquet 62/62 once defined as fault onset |
| 3 | `[BLOCKER]` Naive-baseline gameability persists post-fix for some channel families (e.g. `safety_mode` 90.9%). Stays a blocker for those families: persistence ("repeat the last shown value") needs no corpus knowledge at all — a blind solver applies it to that one item and scores 90.9% without touching the fault physics. Scoped, not template-wide | family-dependent, worst on tmpl_4 | Drop `safety_mode`/rare families from sampling; prefer tmpl_5 (vector, must pass per-component) | tmpl_5 naive floor 9.6% overall — clears the bar comfortably; tmpl_4 needs the carve-outs |
| 4 | `[BLOCKER]` 32 tmpl_4 items have `margin=0` — unpassable by construction (per-item: that item cannot be answered correctly by anyone) | 4.2% of tmpl_4 | Exclude from sampling | Deterministic, confirmed from `acceptance_bounds` alone |

**Net verdict:** a single un-rebased-timestamp bug plus a missing row-exclusion — same bug class as "rebase this field like every other one already is." Fix is generator-level and precisely scoped.

**Update 2026-08-09, blind verification now done — corrects the tmpl_5-preference rationale.** Honest post-truncation solvability: tmpl_4 34.4%, **tmpl_5 only 4.8%** (naive persistence 33.2%/6.5%, oracle ceiling 64.2%/12.5%). The original recommendation to prefer tmpl_5 ("naive floor 9.6%, clears the bar comfortably") was right about the naive floor being low but wrong about why: tmpl_5's low persistence score isn't because genuine reasoning wins there, it's because tmpl_5 is close to unsolvable outright for everyone (only 2 of 415 clean items are solvable by real reasoning and NOT by persistence). Also confirmed: the grading tolerance is pre-fault variability (0.75sigma of the *baseline* std), while the true answer is by construction a fault-driven excursion beyond that baseline (median deviation 2.17sigma tmpl_4, 4.21sigma tmpl_5) — a genuinely hard task, not a broken one, but harder than the original review implied. 6 items sampled (mix of tmpl_4/5), solve-code-verified, blind-verified 5/6 (the 1 failure is an honest, disclosed case of a genuinely unobservable branch — a protective-stop reaction indistinguishable from a continuing ramp using only pre-onset data).
*Policy re-check (2026-08-09): verdict unchanged.* All four findings are per-item — the timestamp discontinuity prints the answer inside a single item, `T` is undefined within a single question, persistence is a zero-corpus-knowledge policy, and `margin=0` makes an individual item unpassable. Nothing here rests on aggregate distribution.

---

## `template_id=6` — "phase window ID, fault-conditioned" (skill 1, L2) — [APPROVED] 2026-08-09 (exclude per problems #1/#2/#5/#7, flag #1-#4/#6 to a co-author)

**Real examples cited:** `e2e310f9-cc20-4dcf-b8b2-93535c113307` (main), `1df2b898-2e0e-41e3-84d9-a1a80d49e653`, `cb2e94a8-365a-421c-8113-103ecd9a81fa` — all verified present in `data/level_2_train.jsonl`. Population: 13,247 items (`factorywave` 80.9%, `vorausad` 13.7%, `aursad` 5.4%).

### 1) What it's asking / what understanding it tests
*"Knowing that the robot suffers from `{an unexpected payload weight}` in the given context time series, we want to isolate the `{approach to the object}` phase. Assuming a fixed window length of `{37}` timesteps, at which timestamp should the window begin?"* Same core skill as L1's sibling, plus a named fault the model must localize the phase boundary despite. 27 distinct fault phrasings.

### 2) Possible answer — free response, not MCQ
Single timestamp, graded against a verified deterministic 6-step band (`±3` render steps) around the phase's actual first timestep. `window_length == phase_length + 5` in 13,247/13,247 items (a leak-shaped but non-leaking regularity — it reveals phase *length*, independent of *position*, already adjudicated at L1).

### 3) Data sources available in context
~37-64 rows at ~100ms, per-item legend. Channel content varies by source: `vorausad`/`aursad` ship fixed full-joint coverage; `factorywave` randomly draws from 20 different 31-channel subsets, so 1,533/10,712 items render *no* speed channel at all (position-differencing recovers the signal, not a blocker). 3 factorywave items render zero telemetry (drop).

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[BLOCKER]` **Phase-name/task-vocabulary mismap, re-confirmed and root-caused**: the pick-and-place lexicon is hardcoded for `vorausad`/`aursad` (which carry no `task` field), while `factorywave` items get real per-task vocabulary. Per-item: on an affected item the phrase in that item's own question names a phase the item's own telemetry doesn't contain | 10.0% overall (aursad 59.0%, vorausad 41.2%, factorywave 1.4%) — 3 groups drive it: vorausad "approach to the object" 99.5% wrong, aursad "descent to the bin" 97.7%, vorausad "pre-grasp pause" 83.3% | **Drop, don't rename — no raw source exists to verify a rename against** (vorausad/aursad episodes aren't in any local parquet). Recommended: drop `aursad` entirely + vorausad phase_name {0,2} | Keeps 11,733/13,247 (88.6%), residual mismap 1.4%, best-case expert solvability rises 81.9% -> 87.9%. **Refined 2026-08-09 while sampling:** built the full task x phrase cross-tab over all 13,247 items directly (not inherited from L1) -- on THIS template, `factorywave` vocabulary is already 100% task-correct (peg_in_hole gets peg words, screwing gets fastener words, zero cross-contamination); the L1 sibling's `peg_in_hole`/`screwing` mismatch buckets do not exist here. The entire mismap is `vorausad`/`aursad` (no `task` field), simplifying the fix scope. **Also confirmed by a direct blind-verification failure**: a physically-correct expert answered a rest-phase item using the exact window-length-sized dwell, landing one row outside the graded band -- concrete proof of problem #3's band-offset issue, not just an arithmetic argument. |
| 2 | `[BLOCKER]` Mismapped items are provably unanswerable, not just mislabeled — e.g. "approach to the object" graded on a window where the arm is already stopped and stays stopped for 32 more rows; real motion onset is >3000ms outside the graded band | same population as #1 | covered by #1's drop | Best-case expert solvability on affected groups: aursad 20.0%, vorausad phase 0 6.1%, aursad phase 6 1.6% — clears the "blocks convergence" severity bar decisively; clears "meaningful population share" for the non-factorywave pool (46% of 2535 items) but not template-wide |
| 3 | `[FLAG-ONLY]` Grading band and question wording disagree by a constant 5-step offset (window "contains the phase" implies 6 valid starts, but only 3 are graded correct) — the bands overlap, so answering the phase start exactly is still graded correct; precision defect, not a convergence blocker | 13,247/13,247 (deterministic) | Shift graded band to `[start-5, start]` — same width, matches what the sentence asks | Arithmetic mismatch, not a solvability issue — zero cost to fix |
| 4 | `[POSITIONAL-GRAY-AREA]` Degenerate "always answer 0" shortcut scores ~2x chance. Formally a corpus-level rate, but it is a constant/boundary-answer bias of exactly the kind LLMs exhibit generically — a fresh blind instance may well anchor on the first timestep without any corpus knowledge. Keep as a live concern, do not dismiss | 22.7% overall (aursad 48.4%, factorywave 24.0%, vorausad 4.8%) | Require phase start >= 4 steps into the render, same window-slack fix already used elsewhere | 22.7% via the shortcut vs 12.2% via naive midpoint guess — worth patching, not a blocker |
| 5 | `[BLOCKER]` Two different phase IDs both render as "return to home" — per-item: within a single affected item the question is ambiguous between two real phases (small population, cheap fix) | 263/13,247 (2.0%) | Rephrase one to a distinct label | Exact, deterministic count |
| 6 | `[FLAG-ONLY]` Unfilled placeholder fault phrase ("a voraus ad class 0 undocumented") — the phrase does name its own source dataset, but the answer here is a timestamp, so knowing the source gives the solver nothing on that item; cosmetic | 72/13,247 (0.5%) | Falls out for free under fix #1 | Exact count |
| 7 | `[BLOCKER]` 3 zero-telemetry items — per-item unanswerable (trivial drop, far below the population-share bar) | 3/13,247 | Drop | Exact count |

**Net verdict:** the ~10% mismap is real, systematic, and source-dataset-scoped — not diffuse background noise. Clears the severity bar for blocking convergence; population share only clears it for the non-factorywave slice. Recommend dropping `aursad` + 2 vorausad phase buckets (88.6% retained, 87.9% best-case solvability), plus 3 cheap deterministic patches (band-shift, dedupe label, drop zero-telemetry items). **Still owed: a blind toolkit-based solve on the post-fix pool — all numbers above are an expert-proxy upper bound, not a measured blind accuracy.**
*Policy re-check (2026-08-09): verdict unchanged.* The drop recommendation rests on #1/#2, which are per-item unanswerability (the item's own question names a phase its own telemetry never enters). #3 and #6 downgrade to flag-only and #4 becomes a positional gray area — none of them were driving the recommendation.

**Cross-level flag, action needed separately:** the same detector run on **L1's already-approved `template_id=1`** found a 33.0% mismatch rate (factorywave alone 25.3%) — worse than this L2 sibling, and L1's vocabulary problem is broader (100% mismatch on several `peg_in_hole`/`screwing` phase buckets). L1 `template_id=1` has been reopened, fixed, and re-approved — see `level_1_final.md`.

---

## `template_id=7` — "anomaly classification" (skill 9) — [APPROVED] 2026-08-09 (exclude per problems #2/#3, disclose #4, flag #1/#1b to a co-author)

**Real example cited throughout:** item `b00c0d88-35ec-4f9a-9c37-ba04b5f6b33d`, `final_submission/raw_by_level/level_2/template_7.json`. Secondary citations: `084bb45b-62e4-4be6-b91c-3366778b0dfe` (placeholder defect), `bb7243c7-f0f8-410f-a197-b5e5397251d5` (zero-telemetry defect). Population: 8272 items (factorywave 6096, vorausad 1711, aursad 465).

### 1) What it's asking / what understanding it tests
*"Given the sensor data from a robot performing the task, determine what anomaly is present? Answer only with a letter."* Fully fixed stem, zero wildcards — all variation lives in the 4 options, each one a whole fault-catalog description (35 distinct texts total, drawn from `data/knowledge_graph.json`). Intent: map an observed telemetry signature back to its physical cause.

### 2) Possible answers — option by option
Each option **is** the wildcard (a full catalog description, no fixed boilerplate). In the cited item: **A** = TCP-frame misconfiguration (the true answer — a static, pose-dependent gravity-compensation error), **B** = external disturbance (a persistent exogenous force — genuinely hard to tell apart from A, see #4), **C** = joint-limit violation (cleanly eliminable — position stays in range, safety mode never trips), **D** = a screwing-task fault (cleanly eliminable — this episode isn't a screwing task at all).

### 3) Data sources available in context
30-70 rows, ~100ms spacing, but the **channel legend itself varies across 16 distinct families** and correlates strongly with the answer (see #1). Only 21.6% of the config-family-affected factorywave items even ship a torque channel.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[FLAG-ONLY]` **Sampling-prior + channel-set leak, worse than previously logged.** Distractors are drawn ~uniformly from the catalog while true labels are heavily skewed per dataset, and the channel-set itself (which varies by item) predicts the answer directly. **Downgraded 2026-08-09:** both halves are purely aggregate — the label skew is a corpus prior, and the channel-set→fault mapping has to be *fitted over many items* before it predicts anything. A solver seeing one item with no memory of the corpus and no benchmark-specific training cannot run either. (Legitimate per-item channel reasoning — "no torque channel, so I can't confirm a torque fault" — is honest elimination, not this leak.) Real balance defect, worth fixing at generation, but not admission-blocking | 8272/8272 | Score with a prior conditioned on channel-set, not globally; for the submission sample, restrict to channel-sets with >=4 distinct faults and redraw distractors from that item's own candidate pool | Metadata-only baseline (zero telemetry read): global prior 44.0%, channel-set+majority 65.0%, **channel-set + 4-option restriction 86.6%** — worse than the ~65% previously logged. After the fix: 18-24% vs 25% chance — leak fully closed |
| 1b | `[FLAG-ONLY]` The previously-banked ~81% "honest signal" number is not safe to quote — it only cancels the *global* prior, and the channel-set-conditioned metadata-only baseline (86.6%) is already higher than that. Measurement-hygiene note that follows #1: the 86.6% null is itself corpus-fitted, so it bounds what a *corpus-aware* attacker gets, not what a blind per-item solver gets | same | Re-measure under the channel-set-conditioned null before trusting any accuracy figure | Not yet re-measured; flagged |
| 2 | `[BLOCKER]` **Unfilled placeholder options are worse than a free eliminate — they're a giveaway.** Two catalog entries are literally "class N, semantic description pending curation," and picking that option is correct 92.3% of the time it appears. **Stays a blocker:** the 92.3% is a statistic *about a per-item defect* — the tell is a formatting anomaly printed in that single item's own option list, and it plays into the generic odd-one-out bias LLMs already have, needing no corpus knowledge. Separately, when the placeholder *is* the key the item is unanswerable on its own terms: the ground truth is an uncurated non-description no physics can confirm | 19.3% of items contain >=1 such option; 17.8% of all items are answered by it | Curate the two undocumented catalog entries; until fixed, exclude items where this is the true answer, and replace it as a distractor with a real competitor | Confirmed: "pick the pending-curation option" rule scores 92.1% correct when it fires |
| 3 | `[BLOCKER]` One zero-telemetry item — per-item unanswerable (trivial drop) | 1/8272 | Drop | Confirmed via exhaustive scan |
| 4 | `[BLOCKER]` Config-family cluster (TCP-frame / payload-mass / payload-CoG / external-disturbance) near-indistinguishable — known physical wall. Per-item: on an affected item the shown channels cannot separate the true option from its neighbour, so an expert cannot converge. Blocking in the sense that these items must be excluded from the answerable pool, not that the template needs a data fix | 33.4% of the template (factorywave), only 21.6% of those ship a torque channel | No fix — disclosed limitation; mark all-config-family items as expert-unanswerable rather than counted solvable | Even with torque available, best held-out separation is 58.7% vs 56.6% majority — no real separation |
| 5 | `[BLOCKER, NEW 2026-08-09]` **`aursad` is effectively unanswerable for this template.** Its channel set has no screwdriver-specific channel (`ett5` identically 0), so a screwing fault cannot be positively identified — yet screwing faults are 100% of aursad's true labels here. Confirmed via blind testing: 4/4 aursad candidates failed, with solvers repeatedly misreading the sustained static wrench as an external load or collision aftermath | 100% of aursad items in this template (share of 8272 not yet separately counted) | Exclude `aursad`-provenance items from sampling until a screwdriver-relevant channel is added for this template | 0/4 blind pass rate on aursad candidates, all failing the same way |
| 6 | `[BLOCKER, NEW 2026-08-09]` **`vorausad` is also effectively unanswerable.** Tracking error sits uniformly at the sensor's quantization floor (0.02 rad) across the entire pool, and torque roughness doesn't separate fault classes — there's no discriminating signal to find, not just a hard one. Confirmed via blind testing: the one candidate tried failed, with the solver correctly noting the telemetry *contradicts* the stated fault rather than just failing to confirm it | 100% of vorausad items in this template | Exclude `vorausad`-provenance items from sampling; same physical wall as elsewhere in this project, needs a higher-resolution channel to fix | 0/1 blind pass, and the failure mode is worse than a miss (positively misleading, not just uninformative) |
| 7 | `[BLOCKER, NEW 2026-08-09]` **Raw-provenance gap**: items whose episode lives in `ur_signals.parquet` (full-rate, not the 10Hz version) cannot be located anywhere in the shipped telemetry at all — exhaustive scan of all 6.5M+ rows across both UR parquets returns zero hits for the rendered window's own values. Items sourced from `ur_signals_10hz.parquet` align exactly; only the full-rate source has this gap | at least 2 confirmed candidates found; population share not yet quantified | Prefer `ur_signals_10hz`-sourced items when sampling until traced; flag to a co-author as a provenance/rendering bug for the full-rate source specifically | 0/2 locatable in the full-rate parquet despite passing blind verification; confirmed reproducible with a runnable proof script |

**Net verdict — CHANGED 2026-08-09 under the new per-item/corpus-level policy: from "not usable until the distractor-redraw lands" to "usable with a bounded exclusion list, plus a balance flag to a co-author."**

The old verdict was gated on finding #1 (sampling prior + channel-set→answer correlation), which has now been reclassified `[FLAG-ONLY]`: it is exploitable only by a solver that has fitted the mapping across many items, which is exactly the aggregate-only case the policy downgrades. It remains a genuine dataset-balance defect and should still be raised with a co-author and fixed at generation — it is simply no longer a precondition for admitting items.

What still blocks, all per-item: curate (or exclude) the 2 uncurated placeholder catalog entries (#2 — a formatting tell inside the item itself, and an unverifiable key when it is the answer), drop the 1 zero-telemetry item (#3), and mark the config-family cluster expert-unanswerable rather than counting it solvable (#4). Do those three and the template is admissible. **Updated while sampling: also exclude `aursad` and `vorausad` entirely (#5, #6) — both proved genuinely unanswerable via direct blind testing (0/4 and 0/1 pass), not just harder. And prefer `ur_signals_10hz`-sourced items over full-rate `ur_signals`-sourced ones (#7) — the latter can't be located in the shipped telemetry at all. 6 items sampled, all `factorywave`/`ur_signals_10hz`-sourced, solve-code-verified, blind-verified 6/6 (plus a 7th candidate that passed blind but was dropped for the provenance gap).

Caveat carried forward, now correctly scoped: the previously-cited ~80% accuracy figure still should not be quoted as clean evidence of physical signal, because it was never measured under a channel-set-conditioned null (#1b). That is a measurement-hygiene debt on *our* reported numbers, not a leak a blind solver can exploit — it does not gate admission.

---

## `template_id=10` — "robot identity under a named fault" (skill 6) — [APPROVED] 2026-08-09 (exclude only per problem #5, flag #1-#4/#6 to a co-author)

**Update:** the L1-side search for a purer physical UR3e-vs-Yu5 discriminator (mentioned as
"still owed" below) is now done — see L1 `template_id=6`, servo tracking-error dynamics,
99.7-100% held-out. Row #2's caveat below is accordingly resolved, not just disclosed.

**Real example cited throughout:** item `1d21ceea-9c44-4b1b-a909-77cbdbc7f0f5`, `data/level_2_train.jsonl` (`vorausad`, `fault_label=12`). Corpus: 8272 items.

> **Note:** 3 previously-banked L2 skill6 items in `real_solve/` (`skill6_case_eta/theta/iota`) no longer exist in the current pull — 0/3 IDs present in `data/level_2_*.jsonl`. Same staleness pattern seen elsewhere this session (skill2 L1's orphaned bank). Their method summaries were still reusable as evidence.

### 1) What it's asking / what understanding it tests
*"Knowing that the robot suffers from `{an unexpected payload weight}` in the given context time series, what robot does this sensor data originate from?"* Wildcard: the named fault (30 distinct phrases). Same core task as L1's `template_id=6` sibling, with a fault perturbing the trajectory being fingerprinted.

### 2) Possible answers — option by option
**Agile Robots Yu 5 Industrial** — true in 20.7% of items, always exactly when `provenance.dataset=vorausad`. Same-topology-as-UR3e problem as the L1 sibling — can't be solved by kinematics alone.
**Universal Robots UR3e** — true in 79.3%, spans `factorywave` + `aursad`. Solvable via parallel-axis invariant / link-length ratio.
**KUKA KR 10 R1100-2** — true in **0 of 8272 items**. Always eliminable via the parallel-axis test (available in 55.5% of items as a clean diagnostic segment), but never the actual answer — same class as the L1 finding of zero true-KUKA cases (though there it was a sampling-script bug; here true-KUKA episodes exist in the raw KUKA parquet but were never drawn into this template).

### 3) Data sources available in context
32-64 rows, `acronym_mapping` legend. `aursad`/`vorausad` items: fixed 12-channel set in radians. `factorywave` items: one of 17 different ~30-channel subsets in degrees. Nothing pins absolute scale/reach/joint limits — the one thing the hard UR3e-vs-Yu5 case would need.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[FLAG-ONLY]` Option KUKA never true — knowing "KUKA is never the answer" requires counting across the corpus; on a single item KUKA is eliminated by the parallel-axis test, which is legitimate physics. Balance defect, disclose | 0/8272 true, 1/3 options always | Disclose only — rebalancing needs generator-side episode sampling from `kuka_signals.parquet` | Parallel-axis elimination still genuinely available 55.5% of the time |
| 2 | `[FLAG-ONLY]` UR3e-vs-Yu5 call rests on batch fingerprints (mean joint-5 position = 100% alone), not pure physics — same caveat as L1 sibling. The fingerprint is corpus-fitted (a threshold learned over the batch); a blind solver on one item cannot apply it. Real validity caveat about what the template measures, not a per-item exploit | 2176/8272 (26.3%) where load-bearing | Same open suggestion as L1: expose joint range limits or the (already-flowing-through-the-pipeline) gripper-interface data — **a dedicated search for a purer physical signal is in progress, see L1 `template_id=6`** | Physics-only leg (clock jitter timing, no joint positions) = 97.98% hard-pair / 93.58% full corpus; full fingerprint method = 99.89% |
| 3 | `[BLOCKER]` (partial) / `[FLAG-ONLY]` (remainder) — **the fault-name clause leaks the source dataset/answer.** Splits cleanly under the new policy: **(a) per-item blocker** for the items whose phrase *names its own source dataset in the printed text* (e.g. "a voraus ad class N …") — that is the answer spelled out inside that single item, exactly the case the policy keeps as a blocker; **(b) flag-only** for the residual phrase↔source correlation behind the 97.22%, which is a *vote fitted across the corpus* and unusable by a solver seeing one item cold | (a) exactly 74/8272 items, all `vorausad`, all the single phrase "a voraus ad class 0 undocumented"; (b) 8272/8272 | (a) drop or re-phrase the dataset-naming fault phrases — cheap, do this before admission; (b) normalize fault vocabulary across datasets so phrasing doesn't correlate with source — generation-side balance fix, flag to a co-author | Confirmed — two phrases name their own source dataset outright. **Population of (a) now counted: 74 items, resolved 2026-08-09** |
| 4 | `[FLAG-ONLY]` **which of the 17 channel-subsets got rendered is itself a near-perfect fingerprint** ("rich channel set => UR3e" scores 94.4% with zero reasoning). **Downgraded 2026-08-09:** this is the policy's own named example — "channel-set richness correlates with robot identity across many items." "Rich" is only definable relative to the other 16 subsets in the corpus; on a single item there is no baseline to call a channel set rich or sparse, and nothing about the count follows from that item's physics. Genuine generation-side defect, flag to a co-author, not admission-blocking | 8272/8272 (100%) | Render a fixed, uniform channel set regardless of source | Confirmed 6094/6094 exact on the simple form of the rule |
| 5 | `[BLOCKER]` 2 items render zero telemetry — per-item unanswerable (trivial drop; population share is far below the severity bar) | 2/8272 | Drop | Unanswerable by construction, below severity bar |
| 6 | `[FLAG-ONLY]` Fault doesn't change the underlying solving method (verification result, not a defect at all) | — | none needed | Method trained on fault-free L1 transfers to fault-present L2 at 99.92% both directions |

**Net verdict — CHANGED 2026-08-09 under the new per-item/corpus-level policy: from "flagging for discussion, may not be admissible" to "usable with disclosed caveats, in line with its L1 sibling."**

The open question was whether rows 3 and 4 — the two newly-found corpus-wide shortcuts — should block admission. Under the policy they largely do not: row 4 (channel-subset richness ⇒ robot identity) is the policy's own textbook downgrade, and the 97.22% figure in row 3 is a phrase↔source vote fitted across the whole corpus. Neither is available to a model answering one question with no memory of the rest of the benchmark and no training on it. Both remain real generation-side balance defects and should go to a co-author with the numbers attached — they are just not admission gates.

What survives as a per-item blocker is narrower and cheaper: the subset of items whose **fault phrase literally names its own source dataset** ("a voraus ad class N …") — the answer printed inside that one item — plus the 2 zero-telemetry items. Fix = drop/re-phrase those items; **an exact count of the dataset-naming subset is still owed** before the drop can be scoped.

Net: `template_id=10` is no longer "strictly weaker than `template_id=6`, admission in doubt." It sits where its L1 sibling sits — the intended kinematic method works, the UR3e-vs-Yu5 call still leans on a batch fingerprint rather than pure physics (row 2, disclosed caveat, unchanged), and the remaining shortcuts are corpus-level balance notes rather than solver-exploitable leaks. **Still owed before freezing: the row-3(a) count, and the L1-side search for a purer physical UR3e/Yu5 discriminator.**

---

## `template_id=2` — "future-outcome check" (skill 3) — [APPROVED] 2026-08-09

**Real example cited:** `19f3ef35-eab5-42dd-938a-da3024da2d7b`, `final_submission/raw_by_level/level_2/template_2.json`. Population: 766 items across 498 episodes, spanning 4 raw signal files.

> **The old "fundamentally unanswerable, exclude entirely" verdict was a false negative twice over.** The L3-style position/velocity axis bug is present here too and IS fixable by rewording (94.8% held-out) — same pattern as `template_id=3`. And problems #3/#4 (TCP-alignment%, speed-baseline%) were only tested against ~60 hand-picked physics formulas — a genuinely trained, data-driven model beats majority-guess by 12-15 points on both, taking the shipped label at face value as ground truth (see #3 below, corrected). Net: this template is now usable end-to-end, not excluded.

### 1) What it's asking / what understanding it tests
Shows ~30-60 rows around a fault ("a robot exhibiting a collision with a cardboard object"), asks "what would most likely happen next" — 4 T/F statements about the unshown continuation. Same forecasting design as its L3 sibling.

### 2) Possible answers — option by option
**TCP-alignment %** and **speed-baseline %** families (688/766 items each) — **genuinely broken**, see #3 below.
**Tracking-error** family (766/766) — worded as position, graded as velocity for the factorywave majority (64% of items) but graded correctly-as-worded on position for the aursad/vorausad/KUKA minority — same option text, two different graders depending on data source.
**Sweep/path-length/torque** family — correctly worded and reproducible (93-100%), no issue.

### 3) Data sources available in context
Channel legend varies by item; critically, most items ship feedback speeds but **not** commanded joint speeds — so even the corrected tracking-error statistic isn't computable from context alone (fine under the forecasting design, but the channel needed to even define the target is missing from the legend). **Smoking gun**: for the `ur_signals`/`ur_screwdriver`-sourced 26% of items, the rendered context does NOT match the shipped parquets (systematic 0.6-1.3° per-joint offsets) — these were rendered and graded from an internal series that isn't in the release at all.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[FLAG-ONLY]` Forecasting requirement (all families) — intended design, disclosure note only | 100% | No fix — intended design, same as L3 sibling | 93-100% held-out on the healthy families |
| 2 | `[BLOCKER]` Tracking-error worded as position, graded as velocity (factorywave) / correctly-as-worded elsewhere. Per-item: the mislabeled axis is discoverable — and fatal — from that single item's own physics; an expert reading the text literally answers backwards on that item | 766/766 (1 option each) | Reword for factorywave items specifically (same fix as `template_id=3`); add commanded-speed channels to the legend | 94.8% held-out on corrected axis vs 44.4% as-worded |
| 3 | `[FLAG-ONLY, REFRAMED 2026-08-09]` **TCP-alignment% and speed-baseline% couldn't be reproduced by ~60 hand-picked physics formulas — but a genuinely trained model, using the shown window's own features and taking the shipped label at face value (same treatment as the healthy forecasting families), beats majority-guess by a wide margin.** Not a per-item defect once graded this way — it's a hard-but-fair forecasting target, same category as problem #1 | 688+688 options | **No fix needed for admission** — treat as a forecasting family like the others; disclose the accuracy ceiling honestly | Episode-disjoint held-out: TCP 67.6% vs 52.8% majority (AUC 0.72); speed 73.7% vs 61.9% majority (AUC 0.84). Oracle (full raw episode, cheating) only adds ~2pp on top — the residual gap is a genuinely unknown exact formula, not missing data, so it doesn't block using this as a fair forecasting task |
| 3b | `[FLAG-ONLY]` The *exact* deterministic key behind #3 still isn't recoverable even with full raw-episode access (oracle 69.2%/75.6%, barely above the honest window-only model) | 688+688 options | Still worth asking a co-author for the exact formula/internal series, but no longer a precondition for admitting this template | Confirms the labels are internally consistent (99.3%+ monotone) but computed from something outside any tested formula or shipped signal |
| 4 | `[FLAG-ONLY]` KUKA "current peaks then relaxes" unresolved (small n, not covered by the reframe test — too small a sample for the same treatment) | ~2.8% of options | Same escalation as #3b, lower priority given the small population | ~50-62% under all tested forms; status unchanged pending a dedicated small-sample check |

**Net verdict:** usable. Problem #2 (tracking-error mislabel) is fixed by reword. Problems #3/#4,
previously the reason for full exclusion, are reframed: a trained forecaster achieves real,
held-out, above-majority accuracy on the shown window alone, so these are hard forecasting
targets rather than unanswerable defects. The unresolved exact formula (#3b) stays a nice-to-have
ask for a co-author, not a blocker. Only the small KUKA slice (#4) remains genuinely unresolved.
*Policy re-check (2026-08-09): superseded by the reframe above — the original "per-item
unanswerable" call for #3/#4 didn't hold up once tested with a flexible model instead of hand-picked formulas.*

---

## `template_id=8` — "pairwise comparison" (skill 7, L2) — [WITHDRAWN] 2026-08-10 (previously approved 2026-08-09 with "C admittable now, A admittable on the clean 92.5% subset" — a full audit of the 6 sampled items found this doesn't hold; withdrawn from the submission, needs full regeneration before it can be re-attempted)

**Real example cited throughout:** item `bb1c5033-e3a4-4d32-a201-cbb095905687` (verified: level_2_train.jsonl, template_id=8, machine_id_a=0, machine_id_b=0), `final_submission/raw_by_level/level_2/template_8.json` (population n=4943; train 4797 / validation 74 / test 72).

> **Flag: the briefed "known facts" for this template turned out to describe a stale, smaller snapshot (`filtered_snapshot/`, n=2943), not the current HF pull (`raw_by_level/`, n=4943).** Three of four premises invert on the real population. See reconciliation note at the end. Treat `raw_by_level` as canonical per standing policy (always use the latest HF pull) — `filtered_snapshot` is stale here, not a valid reference.

### 1) What it's asking / what understanding it tests
*"You are provided with two sensor streams originating from robots accomplishing tasks. What differences between the two given instances of robotic time series data (if any) do you notice? Answer only with a 4 letter string using F and T."*
No wildcards anywhere — question and all 4 option strings are byte-identical across all 4943 items. All per-item variation lives entirely in the two data payloads being compared.

### 2) Possible answers — option by option

**A** — *"Those come from different robots."*
100% determined by `provenance.machine_id_a != machine_id_b` (T-rate 60.41%). **Leaked**: predictable at 91.24% from channel-set differences alone (reading legends, not telemetry). **Mis-keyed on 370 items** (7.49%): cross-dataset pairs (e.g. aursad vs. factorywave) sharing `machine_id=0` are keyed False ("same robot") despite rendering different channel sets and units (rad vs. deg) — an expert reading the evidence correctly answers True and is marked wrong. The cited item is exactly this case.

**B** — *"The two robots have different anomalous states..."*
100% determined by `fault_label_a != fault_label_b`. Confirmed near-unanswerable: a true force channel is jointly available in only 24% of pairs, `gripper_command` in 2.55%; best classifier AUC 0.617. Also reduces largely to "different dataset" (cross-dataset pairs are keyed True 99.78% of the time) — same underlying leak as A.

**C** — *"The two robots are performing different tasks."*
**Solvable, no fix needed** — wrist-pose fingerprint validated 100% held-out on 5886 real series. Minor, disclose-only note: correlates with cross- vs. same-dataset framing (93.74% vs. 47.57% True rate) but this is a property of sampling, not a shortcut replacing the real method.

**D** — *"...same task, but at different phases."*
Near-total shadow of C: `D = NOT C` reproduces the key on 98.28% of items — not currently testing an independent signal.

### 3) Data sources available in context
Two independent excerpts (median 48 rows, ~100ms spacing) — but **the two series in a pair don't share a channel schema**: 90 distinct channel-set pairings occur across `aursad`/`vorausad`/`factorywave` sources, units aren't harmonized (radians vs. degrees), and `provenance` (where the actual answer-determining `machine_id`/`fault_label` live) is never shown to the model.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[BLOCKER]` Option A leaked via channel-set presence, not physics. **Per-item despite being a channel-set finding** — unlike tmpl_7, the comparison is *internal to the item*: both series are printed side by side, so "these two legends differ / one is in radians and one in degrees ⇒ different robots" is immediate reasoning on that one item, needing no corpus-fitted mapping | 4943 (100%) | Render an identical/harmonized channel set (common core + the added channels below) for both series in a pair; harmonize units to degrees | Leak measured at 91.24% vs 60.41% majority; fix not yet verified (needs regen) |
| 2 | `[BLOCKER]` Option A mis-keyed on 370 cross-dataset same-`machine_id` pairs — per-item wrong key: an expert reading that item's own evidence correctly answers True and is marked wrong | 370 (7.49%) | Split `machine_id=0` per-dataset, or drop these pairs from the population | Conflict count verified exactly; fix not yet verified |
| 3 | `[BLOCKER]` Option B has no jointly-available anomaly evidence + incommensurable fault-label spaces across datasets — the textbook "channel set is missing exactly the evidence this item needs" case, evaluated per item | 4943 (100%) | Add real, confirmed-existing-upstream channels (`tcp_force_x/y/z`, `tcp_torque_x/y/z`, `joint_current_0..5`, `grip_detected`, etc. from `ur_signals.parquet`) to both series; map fault labels onto one cross-dataset taxonomy | Channels confirmed dense/real; whether they lift B above chance NOT YET measured post-fix |
| 4 | `[BLOCKER]` Option D near-redundant with C. Per-item, and not a corpus statistic: "same task but different phases" is *logically* incompatible with "different tasks", so `D = NOT C` follows from the two option texts printed in that single item — a solver gets D free from C without any new evidence or any corpus knowledge | 4858 (98.28%) | Drop D, or only score it where C=F | Redundancy verified; fix not yet tested |
| 5 | `[FLAG-ONLY]` Carried-over defect: ~42.8% of a related filtered population's items had contexts untraceable to any raw signal file — this is verification debt on our side (we can't re-derive keys), not something a solver can exploit from any item; also measured on a stale population | measured on old 2943-pop, not yet re-measured on the 4943 raw population | Re-render from located raw windows | Partially verified, needs clean re-run against raw file |

**Net verdict — NOT ready.** C is solid. A and B need real regeneration (rendering fixes + one outright mis-keyed subset), not just wording changes. D should be dropped or rescored. No fix above counts as "done" without a fresh held-out check post-regen.
*Policy re-check (2026-08-09): verdict unchanged, and worth stating why the channel-set finding survives here when the superficially similar tmpl_7 one did not.* This template shows the solver **two series in the same item**, so channel-set/unit differences are a within-item comparison, not a corpus-fitted correlation — #1 is a genuine per-item leak. #2 (wrong key), #3 (missing evidence) and #4 (D derivable from C's own option text) are per-item too. Only the traceability debt (#5) downgrades to flag-only, and it never gated the verdict.

**[WITHDRAWN 2026-08-10, post-approval audit finding.]** This template was nonetheless approved
and sampled on 2026-08-09 on the theory that C is cleanly solvable and A is admittable on a
92.5%-clean subset (excluding the 370 mis-keyed cross-dataset pairs), with B/D disclosed as
flagged-not-derived. A dedicated adversarial audit of the 6 sampled items found that theory
doesn't survive contact with the actual population:
- **Option A is not clean even on the restricted subset** — the exact zero-physics unit tell
  named in problem #1 (TCP reported in mm on one series, m on the other) reproduces the shipped
  A key at **100%** across all 474 raw items the sampling was drawn from, higher than the 91.24%
  leak rate measured pre-fix. Restricting the sample didn't remove the leak, it concentrated it.
- **Option B was never re-derived at all** — the shipped `solve_code` reads it directly from
  `item['answer'][1]` in all 6 items, openly commented as such. Since grading is an exact 4-letter
  string with no partial credit, this means none of the 6 sampled items is actually solvable
  end-to-end by an expert, contradicting the "B flagged, A/C admittable" framing.
- Problem #4's approved fix ("drop D, or only score it where C=F") was never applied — D is
  retained and graded in all 6 items, still derived as `NOT C`.

**This template needs to be rethought at the generation level, not re-patched at the sampling
level** — the underlying problems (#1-#4) are real and were correctly diagnosed, but no amount of
careful item selection can route around a channel-set leak that gets *more* concentrated by
selection, or an option that is architecturally ungraded. Full regeneration (harmonized channels,
re-keyed A, real B evidence, dropped/rescored D) is required before this template can be
re-attempted. Withdrawn from `sampled_dataset.json`; the 6 removed items were compensated for
by additional items sampled from other already-clean L2 templates.

---

## `template_id=9` — "severity ranking" (skill 4b) — [APPROVED] 2026-08-09 — EXCLUDED, not included in our submission

**Real example cited throughout:** item `9bcd7d44-290a-4111-9dc7-42266f073026`, `final_submission/raw_by_level/level_2/template_9.json`. **1858 items, all in `train`** — validation/test contain zero items of this template; no held-out split exists at all.

### 1) What it's asking / what understanding it tests
*"Rank the following robot time series segments from most to least severe anomaly. Answer only with a four letter string indicating your ranking (ie. DCAB), nothing else."* No wildcards in the question. Each option **is** raw, unlabelled sensor data (5-7 timesteps, no timestamps, no legend) — the model must judge which of 4 episodes has the operationally worse fault. Chance = 4.17% exact-match (1/24 permutations).

### 2) Possible answers — option by option
Each option is a raw dump like `{ec0=-0.11, ec1=-1.93, ..., fp0=29.81, ..., tf5=0.14}` — no natural-language predicate, so there's no wildcard/constant distinction the way MCQ templates have; the entire payload varies per item. Correctly ranking implies inferring fault severity purely from a few raw timesteps of an undocumented channel set — in the cited item, option A (gold: most severe) is a `payload_cog_misconfiguration`; B and D are the *same* root cause (`payload_misconfiguration`) yet gold places them 2nd and last with nothing in the data to justify the gap.

### 3) Data sources available in context
**`context` is `{}` for 1858/1858 items (100%, re-confirmed on the current pull) — no legend, no timestamps, no units, no sample rate.** 15 undefined channel prefixes appear across the template. Channel sets differ between options within the same item 77.9% of the time. ~49% of items contain at least one numerically frozen (zero-information) option. The fault label that actually determines gold is never shown. 56.4% of items reference `aursad`/`vorausad` episode IDs that exist in no released table at all — untraceable even by us.

### 4) Problems spotted + suggested fix

| # | Problem | Population hit | Fix | Verified result |
|---|---|---|---|---|
| 1 | `[BLOCKER]` No legend/timestamps/units at all — per-item: the single item in front of the solver is missing the evidence it needs | 100% | Ship `acronym_mapping` per option like templates 6/7/8/10 already do | Necessary but not sufficient — see #3 |
| 2 | `[BLOCKER]` Heterogeneous channel sets + frozen options within an item — both defects are *within* one item (options not comparable to each other; a constant option carries zero information) | 77.9% / 49.0% | Regenerate with uniform channels + a longer (~50-step) window; reject constant windows | Not verifiable without a regen |
| 3 | `[BLOCKER]` **No physically-computable metric tracks the gold ranking** (per-item unanswerable: nothing in the shown window determines that item's key) — best of 8 tried metrics (max tracking error) gets 8.27% exact / 59.4% pairwise; everything else is at/below the 4.17% chance floor, even on a maximally cleaned subset | 100% | None exists at the text level — gold must be recomputed from something the shown signal can actually determine, or the template retired | Re-confirmed this pull; filtering to single-task, single-baseline, 4-distinct-fault items still gets 5.33% exact |
| 4 | `[BLOCKER]` Gold is a hidden lookup on the fault-catalogue's static severity rank (never shown) — per-item: the key depends on a table absent from that item's context | 100% | Either make the fault identifiable from the window first, or redefine gold on a signal-computable proxy | Catalogue rank alone scores 48.46% exact / 86.77% pairwise (811 traceable items) — 6x the best real signal |
| 5 | `[BLOCKER]` The catalogue contradicts itself: 90/288 (31.2%) fault-id pairs are ranked both ways across different items. Measured across items but *not* a corpus-level shortcut — it is a label-quality defect whose consequence is that no consistent rule exists, so affected items are unanswerable in principle, one at a time | 811 traceable items | Fix the generator to use one consistent ranking source | Up from 70 pairs found previously — got worse, not better |
| 6 | `[BLOCKER]` ~49% of items have 2+ options sharing the same fault type, which the generator then ranks arbitrarily — per-item: within that one item the tie is broken by nothing the solver can see | 400/811 (49.3%) | Sample 4 distinct fault types per item | Caps even a perfect severity oracle at 73.2% exact-match |
| 7 | `[POSITIONAL-GRAY-AREA]` **Positional leak.** Guessing the constant string `ABCD` with zero data read scores 21.5% exact / 70.0% pairwise — 2.6x better than the best honest physics metric. Option A is gold-most-severe 51.8% of the time. **Deliberately not downgraded:** although the 21.5% is an aggregate rate, `ABCD` is precisely the identity/first-position ordering that LLMs are generically biased toward in ranking tasks — no corpus-specific learning is needed to emit it, so a fresh blind instance plausibly collects this. Not re-labelled the "biggest finding" (that framing overstated it relative to the per-item blockers above), but it stays a live, fixable concern | 100% | Shuffle option order independently of severity at generation time | Confirmed directly — this is currently the strongest exploitable pattern in the template |
| 8 | `[FLAG-ONLY]` No validation/test split exists — corpus-structural / verification debt, invisible to a solver answering one item | 100% | Emit held-out shards on regen | Confirmed — no fix here can ever be held-out verified against current data |

**Net verdict: NEEDS_REGEN, not patchable, re-confirmed against the current pull.** The answer key is a lookup on data the solver never sees, that data is internally self-contradictory 17% of the time, half the items are unrankable-by-construction (ties/duplicate baselines), and the single strongest pattern in the shipped data is a pure label-order artifact. Recommend excluding this template from the notebook entirely rather than attempting a partial fix.
*Policy re-check (2026-08-09): verdict unchanged, though one supporting clause is re-weighted.* The exclusion never depended on the positional finding — #1 through #6 are all per-item defects (no legend on that item, no computable metric for that item's key, ties that item can't break, a hidden lookup table that item never shows). The `ABCD` shortcut (#7) is now tagged a positional gray area rather than "the biggest finding": still not dismissed, because generic LLM order bias makes it reachable without corpus knowledge, but it is no longer load-bearing for the recommendation.
