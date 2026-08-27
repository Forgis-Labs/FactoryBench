"""Streamlit app: human quality review of FactoryBench QA pairs.

Loads a sample JSONL produced by ``scripts.sample_qa_for_review`` and walks
through one item at a time. Reviewer marks each item as good / minor /
bad / unverifiable, optionally tags issues, and writes a free-text comment.
Verdicts are appended to a JSONL (one verdict per line) so the app survives
crashes and can be resumed, already-reviewed ids are skipped on next launch.

Run::

    streamlit run scripts/qa_review_app.py -- \\
        --sample output/qa_review_sample.jsonl \\
        --verdicts output/qa_review_verdicts.jsonl

Feature charts: time_series rows are parsed back into a DataFrame and grouped
by joint index (e.g. ett0..5 plotted together), so a reviewer can see the
phase / event the question is asking about. The window the question covers
is shaded if ``provenance.subseries_start_index`` / ``subseries_length`` are
present.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Streamlit eats argv after the `--` separator. Parse those into a sub-Namespace.
def _parse_app_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, required=True,
                    help="JSONL produced by scripts.sample_qa_for_review")
    ap.add_argument("--verdicts", type=Path,
                    default=Path("output/qa_review_verdicts.jsonl"),
                    help="JSONL where verdicts are appended (one per line).")
    return ap.parse_args(sys.argv[1:])


_TS_ROW_RE = re.compile(r"^t=(\d+):\s*(.*)$")
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?)")


@st.cache_data(show_spinner=False)
def _load_sample(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _load_done_ids(verdicts_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map id -> last verdict for that id (keeps newest write)."""
    out: Dict[str, Dict[str, Any]] = {}
    if not verdicts_path.exists():
        return out
    with verdicts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except Exception:
                continue
            qid = v.get("id")
            if qid:
                out[qid] = v
    return out


def _parse_time_series(rows: List[str]) -> pd.DataFrame:
    """Turn ['t=0: a=1, b=2'...] into a long DataFrame."""
    records = []
    for row in rows or []:
        m = _TS_ROW_RE.match(row.strip())
        if not m:
            continue
        t = int(m.group(1))
        for k, v in _KV_RE.findall(m.group(2)):
            try:
                records.append({"t": t, "feature": k, "value": float(v)})
            except ValueError:
                continue
    if not records:
        return pd.DataFrame(columns=["t", "feature", "value"])
    return pd.DataFrame.from_records(records)


def _group_features(features: List[str]) -> Dict[str, List[str]]:
    """Group acronyms by joint family: ett0..5 → 'ett', sp0..5 → 'sp', etc."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for f in features:
        m = re.match(r"^([a-z_]+?)([0-9]+)$", f)
        if m:
            groups[m.group(1)].append(f)
        else:
            groups[f].append(f)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: int(re.search(r"[0-9]+$", x).group()) if re.search(r"[0-9]+$", x) else 0)
    return groups


def _append_verdict(verdicts_path: Path, payload: Dict[str, Any]) -> None:
    verdicts_path.parent.mkdir(parents=True, exist_ok=True)
    with verdicts_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_app_args()

    st.set_page_config(page_title="FactoryBench QA Review", layout="wide")
    items = _load_sample(str(args.sample))
    if not items:
        st.error(f"No items found in {args.sample}")
        return
    done = _load_done_ids(args.verdicts)

    # ---- Sidebar -----------------------------------------------------------
    st.sidebar.header("Review session")
    reviewer = st.sidebar.text_input(
        "Reviewer name",
        value=st.session_state.get("reviewer", ""),
        key="reviewer",
        help="Goes into each verdict record.")

    # Stratum filter
    strata = sorted({it.get("stratum", "") for it in items})
    stratum_filter = st.sidebar.multiselect(
        "Filter strata", options=strata, default=strata)
    filtered = [it for it in items if it.get("stratum") in stratum_filter]

    # Skip-already-reviewed toggle
    skip_done = st.sidebar.checkbox("Skip already-reviewed", value=True)
    queue = [it for it in filtered if not (skip_done and it.get("id") in done)]

    n_total = len(filtered)
    n_done_in_filter = sum(1 for it in filtered if it.get("id") in done)
    st.sidebar.metric("Reviewed (in filter)", f"{n_done_in_filter} / {n_total}")
    st.sidebar.metric("Queue length", len(queue))

    if not queue:
        st.success("All items in this filter have been reviewed. ✓")
        st.write(f"Verdicts file: `{args.verdicts}`")
        return

    # Pointer into the queue (resets when filter changes)
    state_key = f"idx::{','.join(stratum_filter)}::{skip_done}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    idx = max(0, min(st.session_state[state_key], len(queue) - 1))
    item = queue[idx]
    qid = item.get("id", "")

    # Nav buttons
    cols = st.sidebar.columns(2)
    if cols[0].button("← Prev"):
        st.session_state[state_key] = max(0, idx - 1)
        st.rerun()
    if cols[1].button("Next →"):
        st.session_state[state_key] = min(len(queue) - 1, idx + 1)
        st.rerun()

    # ---- Main panel --------------------------------------------------------
    prev = done.get(qid)
    st.title(f"{item.get('stratum', '')}  ·  id={qid[:8]}…")
    st.caption(
        f"level={item.get('level')}  template_id={item.get('template_id')}  "
        f"template_type={item.get('template_type')}  "
        f"answer_format={item.get('answer_format', '?')}"
    )
    if prev:
        st.info(
            f"Previously reviewed by **{prev.get('reviewer', '?')}** at "
            f"{prev.get('ts', '?')} → verdict=`{prev.get('verdict')}`"
        )

    # Question + options
    st.subheader("Question")
    st.write(item.get("question", ""))

    options = item.get("options") or {}
    if isinstance(options, dict) and options:
        st.markdown("**Options**")
        for k in sorted(options):
            st.markdown(f"- **{k}**: {options[k]}")

    # Ground truth + acceptance bounds
    st.subheader("Ground truth")
    st.code(json.dumps(item.get("answer"), indent=2, ensure_ascii=False), language="json")
    ab = item.get("acceptance_bounds")
    if ab:
        st.markdown(f"**Acceptance bounds:** `{json.dumps(ab)}`")

    # Provenance
    prov = item.get("provenance") or {}
    with st.expander("Provenance", expanded=False):
        st.json(prov)

    # Time-series visualization
    ts_rows = ((item.get("context") or {}).get("time_series")) or []
    df = _parse_time_series(ts_rows)
    st.subheader(f"Time series  ·  {len(ts_rows)} rows  ·  {df['feature'].nunique() if not df.empty else 0} features")

    if df.empty:
        st.warning("No time-series rows parsed.")
    else:
        # Optional shaded window for the question's subseries / phase
        sub_start = prov.get("subseries_start_index")
        sub_len = prov.get("subseries_length")
        phase_start_in_sub = prov.get("phase_start_in_subseries")
        phase_len = prov.get("phase_length")

        groups = _group_features(sorted(df["feature"].unique()))
        chosen = st.multiselect(
            "Feature groups to plot",
            options=sorted(groups),
            default=[g for g in ("ett", "ecf", "fp", "sp") if g in groups][:3])
        for group in chosen:
            feats = groups[group]
            sub = df[df["feature"].isin(feats)]
            if sub.empty:
                continue
            wide = sub.pivot_table(index="t", columns="feature", values="value")
            wide = wide.reindex(columns=feats)
            st.markdown(f"**{group}**  ({', '.join(feats)})")
            st.line_chart(wide)
            if sub_start is not None and sub_len is not None:
                st.caption(
                    f"subseries window in row index space: "
                    f"[{sub_start}:{sub_start + sub_len}]  "
                    + (f"phase: [{sub_start + (phase_start_in_sub or 0)}:"
                       f"{sub_start + (phase_start_in_sub or 0) + (phase_len or 0)}]"
                       if phase_start_in_sub is not None else "")
                )

        with st.expander("Raw time-series rows", expanded=False):
            st.text("\n".join(ts_rows[:200]) + ("\n..." if len(ts_rows) > 200 else ""))

    # ---- Verdict form ------------------------------------------------------
    st.subheader("Verdict")
    if not reviewer.strip():
        st.warning("Enter a reviewer name in the sidebar to enable submission.")

    default_verdict = (prev or {}).get("verdict", "good")
    default_issues = (prev or {}).get("issues", [])
    default_comment = (prev or {}).get("comment", "")
    default_corrected = (prev or {}).get("corrected_answer", "")

    verdict = st.radio(
        "Overall",
        options=["good", "minor", "bad", "unverifiable"],
        index=["good", "minor", "bad", "unverifiable"].index(default_verdict)
            if default_verdict in ("good", "minor", "bad", "unverifiable") else 0,
        horizontal=True)
    issues = st.multiselect(
        "Issue tags (multi-select)",
        options=[
            "question_unclear",
            "wrong_ground_truth",
            "ambiguous_options",
            "ts_missing_signal",
            "ts_too_short",
            "out_of_scope",
            "leak_in_question",
            "other",
        ],
        default=default_issues)
    comment = st.text_area("Comment", value=default_comment, height=120)
    corrected = st.text_input(
        "Corrected ground truth (optional)", value=default_corrected,
        help="If you'd change the ground truth, paste the proposed value here.")

    submit_disabled = not reviewer.strip()
    if st.button("Save verdict & next", disabled=submit_disabled, type="primary"):
        payload = {
            "id": qid,
            "stratum": item.get("stratum"),
            "level": item.get("level"),
            "template_id": item.get("template_id"),
            "template_type": item.get("template_type"),
            "answer_format": item.get("answer_format"),
            "verdict": verdict,
            "issues": issues,
            "comment": comment,
            "corrected_answer": corrected,
            "reviewer": reviewer.strip(),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        _append_verdict(args.verdicts, payload)
        # Refresh in-memory cache of done ids and advance
        done[qid] = payload
        st.session_state[state_key] = min(len(queue) - 1, idx + 1)
        st.rerun()


if __name__ == "__main__":
    main()
