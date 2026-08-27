"""
Normalize FactoryWave KUKA signals to the same UR3e schema used by the UR
normalizer, so kuka episodes drop into the same downstream pipeline.

Reads ``kuka_signals.parquet`` and the merged ``episode.parquet``, then writes
per-episode JSON files alongside the UR outputs in
``data/normalized_episodes/factorywave/``.

KUKA-specific points:
  * Channel mapping covers what KUKA actually records (positions, motor
    current/torque/temp, TCP pose incl. a/b/c Euler, IMU acc as vibration,
    gripper force, process_state, speed_override). Slots that don't exist on
    KUKA (TCP setpoints, F/T, joint_mode/voltage, safety_mode, etc.) are None.
  * The 16 individual ``digital_input_N``/``digital_output_N`` columns are
    packed into ``digital_input_bits``/``digital_output_bits``.
  * Counterfactual groups use the precomputed ``selected_cf_variant_id`` on
    the baseline row (set by the signature-kernel MMD step). No on-the-fly
    selection is performed here.

Usage:
    python -m src.data.factorywave_kuka_normalizer \\
        --input data/factorywave/data \\
        --output data/normalized_episodes
"""

import ast
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import pyarrow.parquet as pq

from src.data._decimation import decimate_dataframe
from src.data.factorywave_normalizer import (
    _NaNSafeEncoder,
    _to_int,
    _to_float,
    _NON_CONTINUOUS,
    TARGET_HZ,
    _find_fault_onset,
    load_episode_metadata,
    _inject_fault_flip_if_missing,
    _align_fault_flip_to_protective_stop,
    _infer_task_from_phases)

logger = logging.getLogger(__name__)


# KUKA decimation: kuka_signals are ~80 Hz (~2100 samples / 26 s).
# Override TARGET_HZ via CLI if needed.
KUKA_SOURCE_HZ_HINT = 80


# ---------------------------------------------------------------------------
# Column mapping kuka_signals -> UR3e schema (matches UR normalizer keys)
# ---------------------------------------------------------------------------

_DIGITAL_INPUT_BITS_TOKEN  = "_DIGITAL_INPUT_BITS"
_DIGITAL_OUTPUT_BITS_TOKEN = "_DIGITAL_OUTPUT_BITS"


def _build_kuka_column_mapping() -> Dict[str, Optional[str]]:
    m: Dict[str, Optional[str]] = {}

    for i in range(6):
        # INTENT, joint commands. KUKA records setpoint position only.
        m[f"setpoint_pos_{i}"]   = f"setpoint_pos_{i}"
        m[f"setpoint_speed_{i}"] = None
        m[f"setpoint_acc_{i}"]   = None

        # OUTCOME, joint feedback
        m[f"feedback_pos_{i}"]   = f"joint_{i}"
        m[f"feedback_speed_{i}"] = None

        # OUTCOME, effort. KUKA exposes motor current and motor torque.
        m[f"effort_current_{i}"]        = f"motor_current_{i}"
        m[f"effort_target_current_{i}"] = None
        m[f"effort_target_torque_{i}"]  = f"motor_torque_{i}"
        m[f"control_output_{i}"]        = None

        # CONTEXT, per-joint
        m[f"joint_temp_{i}"]    = f"motor_temp_{i}"
        m[f"joint_mode_{i}"]    = None
        m[f"joint_voltage_{i}"] = None

    # INTENT, TCP commands (KUKA records feedback only)
    for i in range(6):
        m[f"setpoint_tcp_{i}"]       = None
        m[f"setpoint_tcp_speed_{i}"] = None

    # OUTCOME, TCP feedback. KUKA stores pose as (x, y, z, a, b, c) Euler.
    # Mapping lines up with UR's (x, y, z, rx, ry, rz) by index.
    for i, axis in enumerate(["x", "y", "z", "a", "b", "c"]):
        m[f"feedback_tcp_{i}"]       = f"tcp_{axis}"
        m[f"feedback_tcp_speed_{i}"] = None

    # No external F/T sensor on this KUKA setup
    for i in range(6):
        m[f"true_force_{i}"]        = None
        m[f"est_contact_force_{i}"] = None

    # OUTCOME, vibration: KUKA IMU accelerometer at end-effector
    for i, axis in enumerate(["x", "y", "z"]):
        m[f"vibration_{i}"] = f"acc_{axis}"

    m["acoustic_0"]            = None
    m["protective_stop_state"] = None

    # INTENT, gripper (same column name on both robots)
    m["gripper_command"] = "force"

    # CONTEXT, system-level
    m["robot_mode"]            = "process_state"
    m["safety_mode"]           = None
    # Sentinel values: handled specially in normalize_kuka_episode_df below.
    m["digital_input_bits"]    = _DIGITAL_INPUT_BITS_TOKEN
    m["digital_output_bits"]   = _DIGITAL_OUTPUT_BITS_TOKEN
    m["runtime_state"]         = None
    m["main_voltage"]          = None
    m["robot_voltage"]         = None
    m["robot_current"]         = None
    m["speed_scaling"]         = "speed_override"
    m["target_speed_fraction"] = None
    m["tool_momentum"]         = None

    return m


KUKA_COLUMN_MAPPING = _build_kuka_column_mapping()


# Schema columns that must end up as integers (mirrors UR normalizer).
_INT_COLS = {
    "joint_mode_0", "joint_mode_1", "joint_mode_2",
    "joint_mode_3", "joint_mode_4", "joint_mode_5",
    "robot_mode", "safety_mode", "runtime_state",
    "digital_input_bits", "digital_output_bits",
}


def _pack_bits(row: pd.Series, prefix: str) -> Optional[int]:
    """Pack ``digital_input_1..16`` (or output) into a single 16-bit integer.

    Returns None if every column in the row is null.
    """
    mask = 0
    seen_any = False
    for i in range(1, 17):
        v = row.get(f"{prefix}_{i}")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        seen_any = True
        try:
            if int(v) != 0:
                mask |= 1 << (i - 1)
        except (TypeError, ValueError):
            continue
    return mask if seen_any else None


def normalize_kuka_episode_df(
    ep_df: pd.DataFrame,
    first_timestamp_us: int,
    metadata_fault_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert one KUKA episode DataFrame to the UR3e-style row dicts."""
    rows: List[Dict[str, Any]] = []

    signal_faults = pd.to_numeric(ep_df["fault"], errors="coerce").fillna(0)
    signal_has_fault = (signal_faults != 0).any()

    for _, row in ep_df.iterrows():
        out: Dict[str, Any] = {}

        t = row.get("time")
        if pd.notna(t):
            t_us = int(pd.Timestamp(t).value // 1000)
            out["timestamp_ms"] = (t_us - first_timestamp_us) // 1000
        else:
            out["timestamp_ms"] = None

        for schema_col, src_col in KUKA_COLUMN_MAPPING.items():
            if src_col is None:
                out[schema_col] = None
            elif src_col == _DIGITAL_INPUT_BITS_TOKEN:
                out[schema_col] = _pack_bits(row, "digital_input")
            elif src_col == _DIGITAL_OUTPUT_BITS_TOKEN:
                out[schema_col] = _pack_bits(row, "digital_output")
            elif src_col not in row.index:
                out[schema_col] = None
            else:
                val = row[src_col]
                if schema_col in _INT_COLS:
                    out[schema_col] = _to_int(val)
                else:
                    out[schema_col] = _to_float(val)

        signal_fault = _to_int(row.get("fault")) or 0
        if signal_has_fault:
            out["fault_label"] = signal_fault
        else:
            out["fault_label"] = metadata_fault_id if metadata_fault_id else signal_fault

        out["task_phase"] = (
            str(row.get("task_phase")) if pd.notna(row.get("task_phase")) else None
        )
        out["event"] = out["fault_label"]

        rows.append(out)

    return rows


def _decimate_kuka_episode(ep_df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """Decimate a kuka episode to target_hz.

    Notes:
        kuka_signals' ``time`` column is ``datetime64[ns]``, so casting to int64
        yields **nanoseconds** since epoch (the UR normalizer treats this as
        microseconds, which is fine for it because UR loads pre-decimated 10Hz
        files; the raw KUKA file is ~80 Hz so the unit must be respected).
    """
    ep_df = ep_df.reset_index(drop=True)
    times_ns = ep_df["time"].values.astype("int64")  # ns since epoch
    if len(times_ns) <= 5:
        return ep_df

    diffs_ns = pd.Series(times_ns).diff().dropna()
    pos_diffs_ns = diffs_ns[diffs_ns > 0]
    if len(pos_diffs_ns) <= 3:
        return ep_df

    med_dt_ns = pos_diffs_ns.median()
    hz = 1e9 / med_dt_ns
    q = max(1, int(round(hz / target_hz)))
    if q <= 1:
        return ep_df

    onset_idx = _find_fault_onset(ep_df)

    continuous = set(ep_df.columns) - _NON_CONTINUOUS
    decimated = decimate_dataframe(ep_df, q=q, continuous_cols=continuous)

    if onset_idx is not None:
        onset_row = ep_df.iloc[[onset_idx]]
        decimated_onset = onset_idx // q
        if decimated_onset >= len(decimated):
            decimated_onset = len(decimated) - 1
        dec_faults = pd.to_numeric(decimated["fault"], errors="coerce").fillna(0).values
        already = (
            decimated_onset > 0
            and dec_faults[decimated_onset] != 0
            and dec_faults[decimated_onset - 1] == 0
        )
        if not already and decimated_onset < len(decimated):
            decimated = pd.concat(
                [
                    decimated.iloc[:decimated_onset],
                    onset_row.reset_index(drop=True),
                    decimated.iloc[decimated_onset + 1 :],
                ],
                ignore_index=True)
    return decimated


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def normalize_kuka_dataset(
    input_dir: Path,
    output_dir: Path,
    target_hz: int = TARGET_HZ,
    limit: Optional[int] = None,
    cf_limit: Optional[int] = None,
    tasks: Optional[List[str]] = None) -> None:
    episode_path = input_dir / "episode.parquet"
    sig_path     = input_dir / "kuka_signals.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode table not found: {episode_path}")
    if not sig_path.exists():
        raise FileNotFoundError(f"kuka_signals not found: {sig_path}")

    logger.info("Loading episode metadata...")
    ep_table = pd.read_parquet(episode_path)
    ep_table["_meta"] = ep_table["episode_metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or {})
    )
    ep_table["_robot"]     = ep_table["_meta"].apply(lambda m: m.get("robot_model") if isinstance(m, dict) else None)
    ep_table["_task"]      = ep_table["_meta"].apply(lambda m: m.get("task")        if isinstance(m, dict) else None)
    ep_table["_condition"] = ep_table["_meta"].apply(lambda m: m.get("condition")   if isinstance(m, dict) else None)

    ep_table = ep_table[ep_table["_robot"] == "kuka"].copy()
    if tasks:
        ep_table = ep_table[ep_table["_task"].isin(tasks)]
    logger.info(f"KUKA episodes in scope: {len(ep_table)}")

    ep_lookup = {str(r["id"]): r for _, r in ep_table.iterrows()}

    # CF groups, using the precomputed selected_cf_variant_id from each baseline.
    cf_pairs: Dict[str, str]    = {}   # baseline_id -> chosen variant_id
    cf_variant_ids: set         = set()  # ALL variants (selected or not), never written standalone
    selected_variant_ids: set   = set()  # only the selected ones, bundled into their baseline file
    baseline_ids: set           = set()
    has_selection_col = "selected_cf_variant_id" in ep_table.columns
    for _, r in ep_table.iterrows():
        if r["_condition"] != "counterfactual":
            continue
        cf_flag = int(r["counterfactual"]) if pd.notna(r["counterfactual"]) else 0
        if cf_flag >= 1:
            cf_variant_ids.add(str(r["id"]))
            continue
        # baseline (cf_flag == 0): record its selection
        if has_selection_col:
            sel = r.get("selected_cf_variant_id")
            if pd.notna(sel) and sel:
                cf_pairs[str(r["id"])] = str(sel)
                baseline_ids.add(str(r["id"]))
                selected_variant_ids.add(str(sel))
    logger.info(f"KUKA cf pairs (baseline -> chosen variant): {len(cf_pairs)}")
    logger.info(f"KUKA cf variants total: {len(cf_variant_ids)} (skipped from regular pass)")
    if not has_selection_col:
        logger.warning("episode.parquet has no `selected_cf_variant_id` column, "
                       "cf pairs will not be written. Run the sig-kernel MMD step first.")

    # Episodes whose signals we actually need.
    needed_ids: set = set()
    cf_pairs_to_process = list(cf_pairs.items())
    if cf_limit is not None:
        cf_pairs_to_process = cf_pairs_to_process[:cf_limit]
    for bl_id, cf_id in cf_pairs_to_process:
        needed_ids.add(bl_id)
        needed_ids.add(cf_id)

    if limit is None or limit > 0:
        regular_count = 0
        for ep_id in ep_lookup:
            if ep_id in cf_variant_ids or ep_id in baseline_ids:
                continue
            needed_ids.add(ep_id)
            regular_count += 1
            if limit is not None and regular_count >= limit:
                break

    out_dir = output_dir / "factorywave"
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = {f.stem for f in out_dir.glob("*.json") if "_metadata" not in f.name}
    needed_before = len(needed_ids)
    needed_ids -= already_done
    if needed_before - len(needed_ids):
        logger.info(f"Skipping signal load for {needed_before - len(needed_ids)} "
                    f"already-normalized KUKA episodes (metadata will be refreshed).")

    # ---------------------------------------------------------------
    # Load required KUKA signal data (row-group streamed to keep RAM low)
    # ---------------------------------------------------------------
    logger.info(f"Loading kuka_signals for {len(needed_ids)} episodes...")
    pf = pq.ParquetFile(sig_path)
    episode_dfs: Dict[str, pd.DataFrame] = {}
    n_rgs = pf.metadata.num_row_groups
    for rg_idx in range(n_rgs):
        df_rg = pf.read_row_group(rg_idx).to_pandas()
        df_rg = df_rg[df_rg["episode_id"].isin(needed_ids)]
        if df_rg.empty:
            continue
        for eid, g in df_rg.groupby("episode_id", sort=False):
            eid = str(eid)
            if eid in episode_dfs:
                episode_dfs[eid] = pd.concat([episode_dfs[eid], g], ignore_index=True)
            else:
                episode_dfs[eid] = g.reset_index(drop=True)
        if n_rgs > 1:
            logger.info(f"  row group {rg_idx + 1}/{n_rgs} ({len(episode_dfs)} episodes loaded so far)")

    for eid in episode_dfs:
        episode_dfs[eid] = episode_dfs[eid].sort_values("time").reset_index(drop=True)

    decimated = 0
    for eid, ep_df in episode_dfs.items():
        new_df = _decimate_kuka_episode(ep_df, target_hz)
        if len(new_df) < len(ep_df):
            decimated += 1
        episode_dfs[eid] = new_df
    logger.info(f"Loaded {len(episode_dfs)} KUKA episodes ({decimated} decimated to {target_hz} Hz)")

    # ---------------------------------------------------------------
    # Pass 1, counterfactual pairs (baseline + selected variant)
    # ---------------------------------------------------------------
    cf_processed, cf_skipped = 0, 0
    for bl_id, cf_id in cf_pairs_to_process:
        if cf_limit is not None and cf_processed >= cf_limit:
            break
        if bl_id not in episode_dfs or cf_id not in episode_dfs:
            cf_skipped += 1
            continue
        out_file = out_dir / f"{bl_id}.json"
        if out_file.exists():
            cf_processed += 1
            continue

        bl_df = episode_dfs[bl_id]
        cf_df = episode_dfs[cf_id]

        cf_meta_row = ep_lookup.get(cf_id)
        cf_fault_id = _to_int(cf_meta_row.get("cf_fault_id")) if cf_meta_row is not None else None

        bl_first_t = int(pd.Timestamp(bl_df["time"].values[0]).value // 1000)
        cf_first_t = int(pd.Timestamp(cf_df["time"].values[0]).value // 1000)

        bl_rows = normalize_kuka_episode_df(bl_df, bl_first_t)
        cf_rows = normalize_kuka_episode_df(cf_df, cf_first_t, metadata_fault_id=cf_fault_id)
        if not bl_rows or not cf_rows:
            cf_skipped += 1
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"baseline": bl_rows, "counterfactual": cf_rows}, f, indent=2, cls=_NaNSafeEncoder)

        bl_row = ep_lookup[bl_id]
        meta_dict = load_episode_metadata(bl_row)
        meta_dict["robot_type"] = "kuka"
        if cf_meta_row is not None:
            meta_dict["cf_fault_id"]            = _to_int(cf_meta_row.get("cf_fault_id"))
            meta_dict["cf_injection_timestep"]  = _to_int(cf_meta_row.get("cf_injection_timestep"))
            meta_dict["cf_injection_time_s"]    = _to_float(cf_meta_row.get("cf_injection_time_s"))
            meta_dict["cf_baseline_episode_id"] = bl_id
            meta_dict["counterfactual"]         = True

        meta_dict["baseline"] = {
            "num_samples": len(bl_rows),
            "last_timestamp_ms": bl_rows[-1]["timestamp_ms"],
        }
        meta_dict["counterfactual"] = {
            "episode_id": cf_id,
            "num_samples": len(cf_rows),
            "last_timestamp_ms": cf_rows[-1]["timestamp_ms"],
            "cf_fault_id": _to_int(cf_meta_row.get("cf_fault_id")) if cf_meta_row is not None else None,
            "cf_injection_timestep": _to_int(cf_meta_row.get("cf_injection_timestep")) if cf_meta_row is not None else None,
            "selection_method": "signature_kernel_mmd",
        }
        meta_dict["schema_version"] = "1.0"
        with open(out_dir / f"{bl_id}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
        cf_processed += 1
        if cf_processed % 10 == 0:
            logger.info(f"  cf groups: {cf_processed}/{len(cf_pairs_to_process)}")
    logger.info(f"  CF pairs: {cf_processed} processed, {cf_skipped} skipped")

    # ---------------------------------------------------------------
    # Pass 2, non-cf episodes (normal, fault)
    # ---------------------------------------------------------------
    regular_processed, regular_skipped = 0, 0
    for eid, ep_df in episode_dfs.items():
        if eid in cf_variant_ids or eid in baseline_ids:
            continue
        if eid not in ep_lookup:
            continue
        if limit and regular_processed >= limit:
            break
        out_file = out_dir / f"{eid}.json"
        if out_file.exists():
            regular_processed += 1
            continue

        ep_row  = ep_lookup[eid]
        ep_meta = ep_row.get("_meta", {})
        meta_fault_id = _to_int(ep_meta.get("fault_id")) if isinstance(ep_meta, dict) else None
        if meta_fault_id is None:
            fm_str = ep_row.get("fault_metadata")
            if isinstance(fm_str, str):
                try:
                    fm = ast.literal_eval(fm_str)
                    meta_fault_id = _to_int(fm.get("fault_id")) if isinstance(fm, dict) else None
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(fm_str, dict):
                meta_fault_id = _to_int(fm_str.get("fault_id"))

        first_t = int(pd.Timestamp(ep_df["time"].values[0]).value // 1000)
        normalized = normalize_kuka_episode_df(ep_df, first_t, metadata_fault_id=meta_fault_id)
        if not normalized:
            regular_skipped += 1
            continue

        meta_dict = load_episode_metadata(ep_row)
        meta_dict["robot_type"] = "kuka"
        if meta_dict.get("task") in (None, "unknown", ""):
            meta_dict["task"] = _infer_task_from_phases(normalized, meta_dict.get("fault_id"))
        _inject_fault_flip_if_missing(normalized, meta_dict.get("fault_id"))
        _align_fault_flip_to_protective_stop(normalized, meta_dict.get("fault_id"))

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, cls=_NaNSafeEncoder)

        meta_dict["num_samples"]    = len(normalized)
        meta_dict["schema_version"] = "1.0"
        meta_dict["timestamp_info"] = {
            "format": "milliseconds since episode start",
            "last_timestamp_ms": normalized[-1]["timestamp_ms"],
        }
        with open(out_dir / f"{eid}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
        regular_processed += 1
        if regular_processed % 100 == 0:
            logger.info(f"  regular: {regular_processed}")
    logger.info(f"  Regular episodes: {regular_processed} processed, {regular_skipped} skipped")

    # ---------------------------------------------------------------
    # Pass 3, refresh metadata for already-done KUKA episodes
    # ---------------------------------------------------------------
    if already_done:
        logger.info(f"Refreshing metadata for {len(already_done)} pre-existing files...")
        refreshed = 0
        for eid in already_done:
            if eid not in ep_lookup:
                continue
            ep_row = ep_lookup[eid]
            meta_dict = load_episode_metadata(ep_row)
            meta_dict["robot_type"] = "kuka"
            data_file = out_dir / f"{eid}.json"
            if data_file.exists():
                try:
                    with open(data_file, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        meta_dict["num_samples"] = len(data)
                        if data:
                            meta_dict["timestamp_info"] = {
                                "format": "milliseconds since episode start",
                                "last_timestamp_ms": data[-1].get("timestamp_ms"),
                            }
                    elif isinstance(data, dict) and "baseline" in data:
                        bl = data["baseline"]
                        cf = data.get("counterfactual", [])
                        meta_dict["baseline"] = {
                            "num_samples": len(bl),
                            "last_timestamp_ms": bl[-1].get("timestamp_ms") if bl else None,
                        }
                        meta_dict["counterfactual"] = {
                            "num_samples": len(cf),
                            "last_timestamp_ms": cf[-1].get("timestamp_ms") if cf else None,
                        }
                except Exception:
                    pass
            meta_dict["schema_version"] = "1.0"
            with open(out_dir / f"{eid}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)
            refreshed += 1
        logger.info(f"  Refreshed {refreshed} metadata files")

    logger.info(f"Done. Output: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize FactoryWave KUKA signals to UR3e schema JSON.")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input",  type=Path, default=repo_root / "data" / "factorywave" / "data")
    parser.add_argument("--output", type=Path, default=repo_root / "data" / "normalized_episodes")
    parser.add_argument("--target-hz", type=int, default=TARGET_HZ,
                        help=f"Target sampling rate after decimation (default: {TARGET_HZ}).")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Limit number of regular (non-CF) episodes.")
    parser.add_argument("--cf-limit",  type=int, default=None,
                        help="Limit number of CF baseline groups.")
    parser.add_argument("--tasks",     nargs="+", default=None,
                        help="Filter by task (e.g. pick_and_place).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    try:
        normalize_kuka_dataset(
            input_dir=args.input,
            output_dir=args.output,
            target_hz=args.target_hz,
            limit=args.limit,
            cf_limit=args.cf_limit,
            tasks=args.tasks)
        return 0
    except Exception as e:
        logger.error(f"KUKA normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
