# One-time data migrations

Every script here ran **once**, against the already-published
`FactoryBench/FactoryBench` v1.0.0 pool, to repair a defect that had shipped.
None of them is part of the generation pipeline, and none of them should be run
again: the generators under `src/question_generation/` were fixed in the same
commit as each repair, so a corpus regenerated today never has the defect in
the first place.

They are kept because they are the record of how the released data was
produced. If a number in the paper does not match a number you compute from the
current HF release, the difference is almost certainly one of these.

## Why none of them will run as written

All five address the release through the old layout:

```python
SPLITS = ("train", "validation", "test")
remote = f"factorybench_qa/level_{lvl}/{split}.jsonl"
```

That layout no longer exists. The train/val/test partition was dropped in
`3879fbc` because nothing consumed it, and `39e4075` flattened the release to
one file per level plus FactoryBench-Lite. Pointing these scripts at the
current repository will simply 404 on every path.

This is deliberately **not** fixed. Rewriting them for the flat layout would
produce six scripts that look runnable but whose target defects are already
gone from both the generators and the data, which is a worse trap than a script
that fails immediately.

## The five repairs

| Script | Templates | Defect it fixed | Added in |
|---|---|---|---|
| `strip_identification_channel_leak.py` | L1.6, L2.10 | The 30-channel request fingerprinted the robot. Only 4 distinct surviving channel-name signatures across 11,979 items, each mapping to exactly one robot; a lookup fitted on train predicted test at 100% without reading a value. Restricted to the 12 channels every dataset renders (`feedback_pos_0..5`, `setpoint_pos_0..5`), which put the lookup back at the 73.7% majority floor. | `961bcd4` |
| `align_ranking_option_features.py` | L2.1, L3.1 | Option segments were encoded from the full episode row while the context was filtered to the template's `important_features`. All 1,334 ranking items showed a context over ~19 channels and asked for an ordering over ~98, 79 of which appeared in no acronym mapping in the item. Drops every `key=value` token whose acronym is not in the item's own mapping; skips an item if that would make two options identical. | `961bcd4` |
| `trim_ranking_context_overlap.py` | L2.1, L3.1 | Option segments were drawn from `post_event_rows`, which overlaps the context window. 696 of 5,336 segments (13%) appeared verbatim inside their own context, so the ordering could be recovered by string matching rather than reasoning; 304 of 1,334 items affected. Truncates each affected context to end just before the earliest row one of its own segments reproduces. Ground truth unchanged. | `961bcd4` |
| `backfill_level3_onset_timestep.py` | L3 | The generator quoted the counterfactual onset on the raw episode clock while shipping a context renormalised to its own base, so 98% of items named a timestep outside the window the model can see. Rewrites that one number to `first_context_timestamp + event_time_ms`. | `961bcd4` |
| `swap_balanced_comparison_items.py` | L1.3, L2.8 | Every one of the eight yes/no marginals was lopsided (P(B)=0.000 on L1.3 against P(B)=0.953 on L2.8 for the same proposition). Per-position majority guessing scored 0.723 and 0.744 against 0.500 chance. Unlike the others this **cannot** be repaired in place, because the skew is a property of which two episodes are compared: it *swaps* items for regenerations aimed at one of eight (robot, anomaly, task) cells. Counts and positions are preserved, but the ids are new, so any score previously reported on an L1.3 or L2.8 item no longer applies. | `c1f4998` |

## Not here: `scripts/backfill_fault_id_metadata.py`

It looks like one of these and it is not. It repairs local normalized episodes
rather than the HF release, it makes no assumption about the split layout, and
`src/data/data_normalization/mapped_dataset_normalizer.py` still writes episode
metadata without a `fault_id` field, encoding the fault only as a per-row
`fault_label`, while every generator reads `meta.fault_id`. So it remains a
required step after normalizing aursad or voraus-AD, and it stays in
`scripts/`.

## If you ever do need one again

Read it, port it to the flat `level_{N}.jsonl` layout, and dry-run it first.
All the HF-facing scripts default to a dry run and only write with `--push`.
