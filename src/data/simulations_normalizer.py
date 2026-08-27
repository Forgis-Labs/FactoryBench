"""
Normalize ANON-ORG/simulations HuggingFace dataset to UR3e schema format.

Converts baseline_steps.csv and counterfactual_steps.csv per episode into
the standard JSON format conforming to the UR3e schema used by the pipeline.

Dataset source: https://huggingface.co/datasets/ANON-ORG/simulations
Expected input layout (after cloning / downloading):
    <input_dir>/ur5_pick_and_place/counterfactual/episode_NNNNN/
        baseline_steps.csv
        counterfactual_steps.csv
        metadata.csv

Output:
    <output_dir>/simulations/episode_NNNNN.json
    <output_dir>/simulations/episode_NNNNN_metadata.json

Each episode JSON has the structure:
    {
        "baseline":       [ ...rows... ],   # same schema as other normalizers
        "counterfactual": [ ...rows... ]    # same schema, fault active after t_inj
    }

Usage:
    python -m factorybench.data.simulations_normalizer \\
        --input  <path/to/hf_download>  \\
        --output <path/to/normalized_episodes>
"""

import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

from src.data._decimation import decimate_dataframe

logger = logging.getLogger(__name__)


class _NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that replaces NaN with None."""
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


# UR5 joint order → index 0-5
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# TCP components: (x, y, z, roll, pitch, yaw)
TCP_POS_COLS  = ["ee_pos_x_m",       "ee_pos_y_m",       "ee_pos_z_m",
                 "ee_euler_roll_rad", "ee_euler_pitch_rad","ee_euler_yaw_rad"]
TCP_SPD_COLS  = ["ee_linvel_x_mps",  "ee_linvel_y_mps",  "ee_linvel_z_mps",
                 "ee_angvel_x_radps","ee_angvel_y_radps", "ee_angvel_z_radps"]
CONTACT_COLS  = ["contact_force_x_n","contact_force_y_n","contact_force_z_n"]


def _build_column_mapping() -> Dict[str, Optional[str]]:
    """Return schema_column -> source_column mapping (None = not available, set to null)."""
    m: Dict[str, Optional[str]] = {}

    for i, joint in enumerate(JOINTS):
        # Setpoint (commanded)
        m[f"setpoint_pos_{i}"]   = f"joint_cmd_pos_rad_{joint}"
        m[f"setpoint_speed_{i}"] = f"joint_cmd_vel_radps_{joint}"
        m[f"setpoint_acc_{i}"]   = None  # not logged in simulation

        # Feedback (actual)
        m[f"feedback_pos_{i}"]   = f"joint_pos_rad_{joint}"
        m[f"feedback_speed_{i}"] = f"joint_vel_radps_{joint}"

        # Effort
        m[f"effort_current_{i}"]        = None  # no motor current in sim
        m[f"effort_target_current_{i}"] = None
        m[f"effort_target_torque_{i}"]  = f"joint_torque_nm_{joint}"

        # Control output (commanded torque)
        m[f"control_output_{i}"] = f"joint_cmd_torque_nm_{joint}"

        # Context (not available in simulation)
        m[f"joint_temp_{i}"]    = None
        m[f"joint_mode_{i}"]    = None
        m[f"joint_voltage_{i}"] = None

    # TCP setpoint = TCP feedback in simulation (no separate commanded pose)
    for i, col in enumerate(TCP_POS_COLS):
        m[f"setpoint_tcp_{i}"]  = col
        m[f"feedback_tcp_{i}"]  = col
    for i, col in enumerate(TCP_SPD_COLS):
        m[f"setpoint_tcp_speed_{i}"] = col
        m[f"feedback_tcp_speed_{i}"] = col

    # Contact forces: x, y, z → indices 0-2; joints 3-5 not applicable
    for i, col in enumerate(CONTACT_COLS):
        m[f"est_contact_force_{i}"] = col
    for i in range(3, 6):
        m[f"est_contact_force_{i}"] = None

    # Signals not present in simulation
    for i in range(3):
        m[f"vibration_{i}"] = None
    m["acoustic_0"] = None
    for key in ["main_voltage", "robot_voltage", "robot_current",
                "safety_mode", "robot_mode", "runtime_state",
                "speed_scaling", "target_speed_fraction",
                "digital_input_bits", "digital_output_bits",
                "tool_momentum"]:
        m[key] = None

    m["gripper_command"] = "gripper_attached"

    # Add event field (critical for Level 2 generator)
    m["event"] = "event_id"

    return m


COLUMN_MAPPING = _build_column_mapping()


def _to_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_row(row: "pd.Series", first_timestamp_ms: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    # Timestamp: sim_time_s → ms, relative to episode start
    sim_time_s = row.get("sim_time_s")
    if sim_time_s is not None and not pd.isna(sim_time_s):
        result["timestamp_ms"] = int(float(sim_time_s) * 1000 - first_timestamp_ms)
    else:
        result["timestamp_ms"] = None

    # Standard schema fields
    for schema_col, src_col in COLUMN_MAPPING.items():
        if src_col is None or src_col not in row.index:
            result[schema_col] = None
        else:
            if schema_col == "event":
                eid = _to_int(row[src_col])
                if eid and eid != 0:
                    ep = row.get("event_params")
                    ep_s = str(ep).strip() if ep is not None and not pd.isna(ep) else ""
                    result[schema_col] = f"{eid}_{ep_s}" if ep_s else eid
                else:
                    result[schema_col] = None
            else:
                result[schema_col] = _to_float(row[src_col])

    # Fault label: 0 for no event, otherwise use event_id (same as event)
    event_id = row.get("event_id")
    if event_id is None or pd.isna(event_id):
        result["fault_label"] = 0
    else:
        try:
            result["fault_label"] = int(float(event_id))
        except (TypeError, ValueError):
            result["fault_label"] = 0

    # task_phase: preserved as an extra field for phase-aware processing
    task_phase = row.get("task_phase")
    result["task_phase"] = None if (task_phase is None or pd.isna(task_phase)) else str(task_phase)

    return result


def normalize_csv(csv_path: Path, src_hz: int = 60, target_hz: int = 10) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    # Anti-aliased downsample from src_hz to target_hz
    step = src_hz // target_hz
    _NON_CONTINUOUS = {"event_id", "event_params", "task_phase", "gripper_attached"}
    _continuous = set(df.columns) - _NON_CONTINUOUS
    df = decimate_dataframe(df, q=step, continuous_cols=_continuous)

    first_timestamp_ms = float(df.iloc[0]["sim_time_s"]) * 1000

    # Build columns for the DataFrame output
    cols: Dict[str, Any] = {
        "timestamp_ms": (df["sim_time_s"] * 1000 - first_timestamp_ms).astype(int),
    }

    for schema_col, src_col in COLUMN_MAPPING.items():
        if src_col is not None and src_col in df.columns:
            if schema_col == "event":
                event_nums = pd.to_numeric(df[src_col], errors="coerce")
                if "event_params" in df.columns:
                    params_col = df["event_params"].fillna("").astype(str)
                    tokens = []
                    for eid, ep in zip(event_nums, params_col):
                        if pd.isna(eid) or int(eid) == 0:
                            tokens.append(None)
                        else:
                            ep_s = ep.strip()
                            tokens.append(f"{int(eid)}_{ep_s}" if ep_s else int(eid))
                    cols[schema_col] = tokens
                else:
                    cols[schema_col] = event_nums.astype("Int64")
            else:
                cols[schema_col] = pd.to_numeric(df[src_col], errors="coerce")
        else:
            cols[schema_col] = None

    # fault_label (can be kept separately, but we already have event)
    cols["fault_label"] = (
        pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)
        if "event_id" in df.columns else 0
    )
    cols["task_phase"] = df["task_phase"].astype(str) if "task_phase" in df.columns else None

    out = pd.DataFrame(cols, index=df.index).astype(object)
    # Replace NaN with None
    out = out.where(out.notna(), None)
    return out.to_dict(orient="records")


def normalize_episode(ep_dir: Path, output_dir: Path) -> None:
    episode_id = ep_dir.name

    baseline_csv       = ep_dir / "baseline_steps.csv"
    counterfactual_csv = ep_dir / "counterfactual_steps.csv"

    if not baseline_csv.exists() or not counterfactual_csv.exists():
        logger.warning(f"  Skipping {episode_id}: missing baseline or counterfactual CSV.")
        return

    baseline_rows       = normalize_csv(baseline_csv)
    counterfactual_rows = normalize_csv(counterfactual_csv)

    if not baseline_rows or not counterfactual_rows:
        logger.warning(f"  Skipping {episode_id}: empty data.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{episode_id}.json"
    with open(out_file, "w") as f:
        json.dump({"baseline": baseline_rows, "counterfactual": counterfactual_rows}, f, indent=2, cls=_NaNSafeEncoder)

    # Episode-level metadata
    ep_meta = {}
    meta_json = ep_dir / "metadata.json"
    if meta_json.exists():
        try:
            with open(meta_json) as f:
                ep_meta = json.load(f)
        except Exception:
            pass

    raw_baseline = ep_meta.get("baseline", {})
    raw_cf = ep_meta.get("counterfactual", {})

    meta = {
        "episode_id": episode_id,
        "schema": "ur3e_v1",
        "task": "pick_and_place",
        "robot": "ur5",
        "baseline": {
            "num_samples": len(baseline_rows),
            "first_timestamp_ms": 0,
            "last_timestamp_ms": baseline_rows[-1]["timestamp_ms"],
            "duration_ms": baseline_rows[-1]["timestamp_ms"],
            "task_success": bool(raw_baseline.get("success", 1)),
        },
        "counterfactual": {
            "num_samples": len(counterfactual_rows),
            "first_timestamp_ms": 0,
            "last_timestamp_ms": counterfactual_rows[-1]["timestamp_ms"],
            "duration_ms": counterfactual_rows[-1]["timestamp_ms"],
            "task_success": bool(raw_cf.get("success", 1)),
        },
        "episode_meta": ep_meta,
    }
    with open(output_dir / f"{episode_id}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def normalize_dataset(input_dir: Path, output_dir: Path, limit: Optional[int] = None) -> None:
    split_dir = input_dir / "ur5_pick_and_place" / "counterfactual"
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected split directory not found: {split_dir}")

    episode_dirs = sorted(split_dir.glob("episode_*"))
    if limit is not None:
        episode_dirs = episode_dirs[:limit]
    logger.info(f"Normalizing {len(episode_dirs)} episodes from {split_dir}")

    out = output_dir / "simulations"

    for ep_dir in episode_dirs:
        normalize_episode(ep_dir, out)
        logger.info(f"  ✓ {ep_dir.name}")

    logger.info(f"Done → {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize ANON-ORG/simulations HuggingFace dataset to UR3e schema JSON."
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Root directory of the downloaded HF dataset")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory (normalized_episodes)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only normalize the first N episodes (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return 1

    try:
        normalize_dataset(args.input, args.output, limit=args.limit)
        return 0
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())