"""
Normalize real UR3 robot data to UR3e schema format.

Converts CSV files recorded from a live UR3 robot (with ur3_robot_* prefixed
column names) to the standardized JSON format used by the FactoryBench pipeline.

Input CSV columns follow the naming convention used by the robot data recorder:
    ur3_robot_joint_0 .. 5         → joint positions (rad)
    ur3_robot_joint_vel_0 .. 5     → joint velocities (rad/s)
    ur3_robot_tcp_x/y/z/rx/ry/rz   → TCP pose (m / rad)
    ur3_robot_tcp_force_x/y/z      → TCP forces (N)
    ur3_robot_tcp_torque_x/y/z     → TCP torques (N·m)
    ur3_robot_target_joint_0 .. 5  → commanded joint positions (rad)
    ur3_robot_target_joint_vel_0..5→ commanded joint velocities (rad/s)
    ur3_robot_joint_current_0 .. 5 → actual motor currents (A)
    ur3_robot_target_joint_current_0..5 → target motor currents (A)
    ur3_robot_joint_temp_0 .. 5    → joint temperatures (°C)
    ur3_robot_joint_control_output_0..5 → controller output (N·m)
    ur3_robot_joint_mode_0 .. 5    → joint operating mode
    ur3_robot_target_tcp_x/y/z/rx/ry/rz  → commanded TCP pose
    ur3_robot_tcp_speed_x/y/z/rx/ry/rz   → actual TCP speed
    ur3_robot_target_tcp_speed_x/y/z/rx/ry/rz → commanded TCP speed
    ur3_robot_robot_mode            → robot mode enum
    ur3_robot_safety_mode           → safety mode enum
    ur3_robot_digital_inputs        → digital input bits
    ur3_robot_digital_outputs       → digital output bits
    ur3_robot_runtime_state         → runtime state
    ur3_robot_main_voltage          → main voltage (V)
    ur3_robot_robot_voltage         → robot voltage (V)
    ur3_robot_robot_current         → robot current (A)
    ur3_robot_speed_scaling         → speed scaling factor [0–1]
    ur3_robot_target_speed_fraction → target speed fraction [0–1]
    ur3_robot_momentum              → tool momentum (kg·m/s)
    timestamp                       → Unix epoch (seconds, float)

Unmapped source columns (not part of the schema):
    machine_id, realsense_camera_*, ur3_robot_robot_status,
    ur3_robot_safety_status, ur3_robot_analog_input/output_*

Usage:
    python -m factorybench.data.ur3_real_normalizer \\
        --input  <csv_file_or_dir> \\
        --output <output_dir> \\
        [--episode-id <id>]
"""

import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

from src.data._decimation import decimate_dataframe

logger = logging.getLogger(__name__)


class _NaNSafeEncoder(json.JSONEncoder):
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


def _build_column_mapping() -> Dict[str, Optional[str]]:
    """Return schema_column -> source_csv_column mapping (None = not available)."""
    m: Dict[str, Optional[str]] = {}

    for i in range(6):
        # INTENT, joint commands
        m[f"setpoint_pos_{i}"]          = f"ur3_robot_target_joint_{i}"
        m[f"setpoint_speed_{i}"]        = f"ur3_robot_target_joint_vel_{i}"
        m[f"setpoint_acc_{i}"]          = None  # not exposed by UR3

        # OUTCOME, joint feedback
        m[f"feedback_pos_{i}"]          = f"ur3_robot_joint_{i}"
        m[f"feedback_speed_{i}"]        = f"ur3_robot_joint_vel_{i}"

        # OUTCOME, effort / current
        m[f"effort_current_{i}"]        = f"ur3_robot_joint_current_{i}"
        m[f"effort_target_current_{i}"] = f"ur3_robot_target_joint_current_{i}"
        m[f"effort_target_torque_{i}"]  = None  # not in UR3 data stream

        # OUTCOME, controller output
        m[f"control_output_{i}"]        = f"ur3_robot_joint_control_output_{i}"

        # CONTEXT, per-joint
        m[f"joint_temp_{i}"]            = f"ur3_robot_joint_temp_{i}"
        m[f"joint_mode_{i}"]            = f"ur3_robot_joint_mode_{i}"
        m[f"joint_voltage_{i}"]         = None  # not in UR3 data stream

    # INTENT, TCP commands  (x, y, z, rx, ry, rz → indices 0-5)
    for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        m[f"setpoint_tcp_{i}"]          = f"ur3_robot_target_tcp_{axis}"
        m[f"setpoint_tcp_speed_{i}"]    = f"ur3_robot_target_tcp_speed_{axis}"

    # OUTCOME, TCP feedback
    for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        m[f"feedback_tcp_{i}"]          = f"ur3_robot_tcp_{axis}"
        m[f"feedback_tcp_speed_{i}"]    = f"ur3_robot_tcp_speed_{axis}"

    # OUTCOME, contact forces / torques
    # tcp_force_x/y/z → true_force 0-2; tcp_torque_x/y/z → true_force 3-5
    for i, axis in enumerate(["x", "y", "z"]):
        m[f"true_force_{i}"]            = f"ur3_robot_tcp_force_{axis}"
        m[f"true_force_{i + 3}"]        = f"ur3_robot_tcp_torque_{axis}"
    for i in range(6):
        m[f"est_contact_force_{i}"]     = None  # not available from real robot

    # OUTCOME, other
    for i in range(3):
        m[f"vibration_{i}"]             = None
    m["acoustic_0"]                     = None
    m["protective_stop_state"]          = None

    # INTENT, gripper
    m["gripper_command"]                = None

    # CONTEXT, system-level
    m["robot_mode"]                     = "ur3_robot_robot_mode"
    m["safety_mode"]                    = "ur3_robot_safety_mode"
    m["digital_input_bits"]             = "ur3_robot_digital_inputs"
    m["digital_output_bits"]            = "ur3_robot_digital_outputs"
    m["runtime_state"]                  = "ur3_robot_runtime_state"
    m["main_voltage"]                   = "ur3_robot_main_voltage"
    m["robot_voltage"]                  = "ur3_robot_robot_voltage"
    m["robot_current"]                  = "ur3_robot_robot_current"
    m["speed_scaling"]                  = "ur3_robot_speed_scaling"
    m["target_speed_fraction"]          = "ur3_robot_target_speed_fraction"
    m["tool_momentum"]                  = "ur3_robot_momentum"

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
        return int(value)
    except (TypeError, ValueError):
        return None


# Integer-typed schema columns
_INT_COLS = {
    "joint_mode_0", "joint_mode_1", "joint_mode_2",
    "joint_mode_3", "joint_mode_4", "joint_mode_5",
    "robot_mode", "safety_mode", "runtime_state",
    "digital_input_bits", "digital_output_bits",
    "protective_stop_state",
}


def normalize_row(
    row: pd.Series,
    first_timestamp_ms: int) -> Dict[str, Any]:
    """Convert one CSV row to UR3e schema dict."""
    result: Dict[str, Any] = {}

    # Timestamp → ms since episode start
    raw_ts = row.get("timestamp")
    if raw_ts is not None and not pd.isna(raw_ts):
        abs_ms = int(float(raw_ts) * 1000)
        result["timestamp_ms"] = abs_ms - first_timestamp_ms
    else:
        result["timestamp_ms"] = None

    for schema_col, src_col in COLUMN_MAPPING.items():
        if src_col is None or src_col not in row.index:
            result[schema_col] = None
        else:
            val = row[src_col]
            if schema_col in _INT_COLS:
                result[schema_col] = _to_int(val)
            else:
                result[schema_col] = _to_float(val)

    return result


_NON_CONTINUOUS_SRC_COLS = {
    "timestamp", "machine_id",
    "ur3_robot_joint_mode_0", "ur3_robot_joint_mode_1", "ur3_robot_joint_mode_2",
    "ur3_robot_joint_mode_3", "ur3_robot_joint_mode_4", "ur3_robot_joint_mode_5",
    "ur3_robot_robot_mode", "ur3_robot_safety_mode", "ur3_robot_runtime_state",
    "ur3_robot_digital_inputs", "ur3_robot_digital_outputs",
}


def normalize_dataset(
    input_file: Path,
    output_dir: Path,
    episode_id: str,
    include_metadata: bool = True,
    downsample_q: int = 1) -> None:
    """Normalize a single UR3 real-robot CSV to UR3e schema JSON."""
    logger.info(f"Loading {input_file.name} ...")
    df = pd.read_csv(input_file)

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column (Unix epoch seconds).")

    # Anti-aliased downsampling (q=1 means no downsampling)
    if downsample_q > 1:
        continuous = set(df.columns) - _NON_CONTINUOUS_SRC_COLS
        logger.info(f"Decimating by {downsample_q}x ({len(df)} → ~{len(df) // downsample_q} rows) ...")
        df = decimate_dataframe(df, q=downsample_q, continuous_cols=continuous)

    # Compute first timestamp for relative time
    first_ts_ms = int(float(df.iloc[0]["timestamp"]) * 1000)

    logger.info(f"Normalizing {len(df)} rows ...")
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        rows.append(normalize_row(row, first_ts_ms))
        if (idx + 1) % 1000 == 0:
            logger.info(f"  {idx + 1}/{len(df)} rows processed ...")

    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{episode_id}.json"
    with open(out_file, "w") as f:
        json.dump(rows, f, indent=2, cls=_NaNSafeEncoder)
    logger.info(f"Saved → {out_file}")

    if include_metadata:
        available = {
            "joint_positions": True,
            "joint_velocities": True,
            "joint_currents": True,
            "joint_temperatures": True,
            "joint_modes": True,
            "joint_control_outputs": True,
            "tcp_pose": True,
            "tcp_speed": True,
            "tcp_forces_torques": True,
            "setpoint_pos": True,
            "setpoint_speed": True,
            "setpoint_acc": False,
            "setpoint_tcp": True,
            "setpoint_tcp_speed": True,
            "gripper": False,
            "robot_mode": True,
            "safety_mode": True,
            "digital_io": True,
            "voltages_current": True,
            "speed_scaling": True,
            "tool_momentum": True,
            "effort_target_torque": False,
            "joint_voltage": False,
            "est_contact_force": False,
            "vibration": False,
            "acoustic_emission": False,
            "protective_stop_state": False,
        }
        metadata = {
            "episode_id": episode_id,
            "source_file": input_file.name,
            "robot_type": "ur3",
            "data_source": "real_robot",
            "num_samples": len(rows),
            "schema_version": "1.0",
            "description": "UR3 real-robot data normalized from recorder CSV format",
            "timestamp_info": {
                "format": "milliseconds since episode start",
                "first_timestamp_ms": first_ts_ms,
                "last_timestamp_ms": rows[-1]["timestamp_ms"] if rows else None,
            },
            "available_data": available,
        }
        meta_file = output_dir / f"{episode_id}_metadata.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata → {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize real UR3 robot CSV data to UR3e schema JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to CSV file (or directory containing CSVs) to normalize.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for normalized JSON files.")
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Episode identifier. Defaults to the input file stem.")
    parser.add_argument(
        "--source-hz",
        type=int,
        default=None,
        help="Source sampling rate in Hz.  Required when --target-hz is given.")
    parser.add_argument(
        "--target-hz",
        type=int,
        default=None,
        help="Target sampling rate in Hz.  Enables anti-aliased decimation.")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip generating metadata files.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.")
    args = parser.parse_args()

    downsample_q = 1
    if args.target_hz is not None:
        if args.source_hz is None:
            logger.error("--source-hz is required when --target-hz is given.")
            return 1
        if args.source_hz < args.target_hz:
            logger.error("--source-hz must be >= --target-hz.")
            return 1
        downsample_q = args.source_hz // args.target_hz

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    input_path: Path = args.input

    # Collect CSV files to process
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {input_path}")
            return 1
    elif input_path.is_file():
        csv_files = [input_path]
    else:
        logger.error(f"Input path not found: {input_path}")
        return 1

    for csv_file in csv_files:
        episode_id = args.episode_id or csv_file.stem
        try:
            normalize_dataset(
                csv_file,
                args.output,
                episode_id=episode_id,
                include_metadata=not args.no_metadata,
                downsample_q=downsample_q)
        except Exception as e:
            logger.error(f"Failed to normalize {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
