"""
Normalize real KUKA KR10 robot data to UR3e schema format.

Converts CSV files recorded from a live KUKA robot (with kuka_robot_* prefixed
column names) to the standardized JSON format used by the FactoryBench pipeline.

Input CSV column → schema column mapping overview
--------------------------------------------------
kuka_robot_joint_0 .. 5            → feedback_pos_0 .. 5       (rad)
kuka_robot_tcp_x/y/z               → feedback_tcp_0/1/2        (mm → m)
kuka_robot_tcp_a/b/c               → feedback_tcp_3/4/5        (deg → rad)  [KUKA A/B/C Euler]
kuka_robot_setpoint_pos_0 .. 5     → setpoint_pos_0 .. 5       (rad)
kuka_robot_motor_current_0 .. 5    → effort_current_0 .. 5     (A)
kuka_robot_motor_torque_0 .. 5     → effort_target_torque_0..5 (N·m, actual motor torque)
kuka_robot_motor_temp_0 .. 5       → joint_temp_0 .. 5         (°C)
kuka_robot_cart_accel_x/y/z        → vibration_0/1/2           (m/s²)
kuka_robot_digital_inputs          → digital_input_bits        (bitmask)
kuka_robot_digital_outputs         → digital_output_bits       (bitmask)
kuka_robot_speed_override          → speed_scaling             (% → [0–1])
kuka_robot_process_state           → runtime_state
timestamp                          → timestamp_ms              (Unix s → ms since episode start)

Unmapped source columns (not part of the schema):
    machine_id, kuka_robot_digital_input_1..16,
    kuka_robot_digital_output_1..16, kuka_robot_cart_accel_abs

Not available from KUKA KR10 data stream (set to null):
    feedback_speed_*, setpoint_speed_*, setpoint_acc_*,
    setpoint_tcp_*, setpoint_tcp_speed_*, feedback_tcp_speed_*,
    effort_target_current_*, control_output_*, joint_mode_*, joint_voltage_*,
    true_force_*, est_contact_force_*, acoustic_0, protective_stop_state,
    gripper_command, robot_mode, safety_mode, main_voltage, robot_voltage,
    robot_current, target_speed_fraction, tool_momentum

KUKA-specific conventions
--------------------------
- TCP position (x, y, z) is in millimetres; converted to metres by dividing by 1000.
- TCP orientation (A, B, C) is in degrees; converted to radians.
  Mapping: A → feedback_tcp_3, B → feedback_tcp_4, C → feedback_tcp_5.
- speed_override is a percentage (0–100); divided by 100 to match the [0–1] schema range.

Usage:
    python -m factorybench.data.kuka_real_normalizer \\
        --input  <csv_file_or_dir> \\
        --output <output_dir> \\
        [--episode-id <id>] \\
        [--no-mm-conversion]   # skip mm→m conversion if data is already in metres
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

DEG_TO_RAD = math.pi / 180.0


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
    """Return schema_column -> source_csv_column (None = not available, set to null).

    Columns that need unit conversion are handled separately in normalize_row().
    """
    m: Dict[str, Optional[str]] = {}

    for i in range(6):
        # INTENT, joint setpoints
        m[f"setpoint_pos_{i}"]          = f"kuka_robot_setpoint_pos_{i}"
        m[f"setpoint_speed_{i}"]        = None  # not available
        m[f"setpoint_acc_{i}"]          = None  # not available

        # OUTCOME, joint feedback
        m[f"feedback_pos_{i}"]          = f"kuka_robot_joint_{i}"
        m[f"feedback_speed_{i}"]        = None  # not available

        # OUTCOME, effort
        m[f"effort_current_{i}"]        = f"kuka_robot_motor_current_{i}"
        m[f"effort_target_current_{i}"] = None  # not available
        m[f"effort_target_torque_{i}"]  = f"kuka_robot_motor_torque_{i}"

        # OUTCOME, controller output (commanded torque not exposed by KUKA)
        m[f"control_output_{i}"]        = None

        # CONTEXT, per-joint
        m[f"joint_temp_{i}"]            = f"kuka_robot_motor_temp_{i}"
        m[f"joint_mode_{i}"]            = None  # not available
        m[f"joint_voltage_{i}"]         = None  # not available

    # INTENT / OUTCOME, TCP
    # Handled with unit conversion in normalize_row(); mark source columns here.
    # feedback_tcp_0/1/2 ← tcp_x/y/z (mm → m), feedback_tcp_3/4/5 ← tcp_a/b/c (deg → rad)
    for i in range(6):
        m[f"setpoint_tcp_{i}"]          = None  # not available
        m[f"setpoint_tcp_speed_{i}"]    = None
        m[f"feedback_tcp_{i}"]          = None  # handled manually in normalize_row
        m[f"feedback_tcp_speed_{i}"]    = None

    # OUTCOME, forces / vibration
    for i in range(6):
        m[f"true_force_{i}"]            = None  # no force sensor
        m[f"est_contact_force_{i}"]     = None
    m["acoustic_0"]                     = None
    m["protective_stop_state"]          = None

    # OUTCOME, vibration: cart accelerometer axes (handled in normalize_row for sign/unit)
    for i in range(3):
        m[f"vibration_{i}"]             = None  # handled manually

    # INTENT, gripper
    m["gripper_command"]                = None

    # CONTEXT, system-level
    m["robot_mode"]                     = None
    m["safety_mode"]                    = None
    m["digital_input_bits"]             = "kuka_robot_digital_inputs"
    m["digital_output_bits"]            = "kuka_robot_digital_outputs"
    m["runtime_state"]                  = "kuka_robot_process_state"
    m["main_voltage"]                   = None
    m["robot_voltage"]                  = None
    m["robot_current"]                  = None
    # speed_override is % → divided by 100 in normalize_row; mark source col here
    m["speed_scaling"]                  = None  # handled manually
    m["target_speed_fraction"]          = None
    m["tool_momentum"]                  = None

    return m


COLUMN_MAPPING = _build_column_mapping()

# Schema columns that should be stored as integers
_INT_COLS = {
    "joint_mode_0", "joint_mode_1", "joint_mode_2",
    "joint_mode_3", "joint_mode_4", "joint_mode_5",
    "robot_mode", "safety_mode", "runtime_state",
    "digital_input_bits", "digital_output_bits",
    "protective_stop_state",
}

# Source columns for TCP position (mm) and orientation (deg)
_TCP_POS_COLS = ["kuka_robot_tcp_x", "kuka_robot_tcp_y", "kuka_robot_tcp_z"]
_TCP_ORI_COLS = ["kuka_robot_tcp_a", "kuka_robot_tcp_b", "kuka_robot_tcp_c"]
_ACCEL_COLS   = ["kuka_robot_cart_accel_x", "kuka_robot_cart_accel_y", "kuka_robot_cart_accel_z"]


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


def normalize_row(
    row: pd.Series,
    first_timestamp_ms: int,
    convert_mm: bool = True) -> Dict[str, Any]:
    """Convert one CSV row to UR3e schema dict.

    Args:
        row: Pandas Series for one CSV row.
        first_timestamp_ms: Absolute ms of the first sample (for relative time).
        convert_mm: If True, divide TCP x/y/z by 1000 (mm → m). Set False if
                    the source data is already in metres.
    """
    result: Dict[str, Any] = {}

    # Timestamp → ms since episode start
    raw_ts = row.get("timestamp")
    if raw_ts is not None and not pd.isna(raw_ts):
        abs_ms = int(float(raw_ts) * 1000)
        result["timestamp_ms"] = abs_ms - first_timestamp_ms
    else:
        result["timestamp_ms"] = None

    # Standard mapped columns
    for schema_col, src_col in COLUMN_MAPPING.items():
        if src_col is None or src_col not in row.index:
            result[schema_col] = None
        else:
            val = row[src_col]
            if schema_col in _INT_COLS:
                result[schema_col] = _to_int(val)
            else:
                result[schema_col] = _to_float(val)

    # TCP feedback position: x/y/z (mm → m) and A/B/C orientation (deg → rad)
    for i, col in enumerate(_TCP_POS_COLS):
        v = _to_float(row.get(col)) if col in row.index else None
        if v is not None and convert_mm:
            v = v / 1000.0
        result[f"feedback_tcp_{i}"] = v

    for i, col in enumerate(_TCP_ORI_COLS):
        v = _to_float(row.get(col)) if col in row.index else None
        if v is not None:
            v = v * DEG_TO_RAD
        result[f"feedback_tcp_{i + 3}"] = v

    # Vibration: Cartesian accelerometer x/y/z
    for i, col in enumerate(_ACCEL_COLS):
        v = _to_float(row.get(col)) if col in row.index else None
        result[f"vibration_{i}"] = v

    # Speed override: % → [0, 1]
    raw_ovr = row.get("kuka_robot_speed_override")
    if raw_ovr is not None and not pd.isna(raw_ovr):
        result["speed_scaling"] = float(raw_ovr) / 100.0
    else:
        result["speed_scaling"] = None

    return result


_NON_CONTINUOUS_SRC_COLS = {
    "timestamp", "machine_id",
    "kuka_robot_digital_inputs", "kuka_robot_digital_outputs",
    "kuka_robot_process_state",
    *(f"kuka_robot_digital_input_{i}" for i in range(1, 17)),
    *(f"kuka_robot_digital_output_{i}" for i in range(1, 17)),
}


def normalize_dataset(
    input_file: Path,
    output_dir: Path,
    episode_id: str,
    include_metadata: bool = True,
    convert_mm: bool = True,
    downsample_q: int = 1) -> None:
    """Normalize a single KUKA real-robot CSV to UR3e schema JSON."""
    logger.info(f"Loading {input_file.name} ...")
    df = pd.read_csv(input_file)

    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column (Unix epoch seconds).")

    # Anti-aliased downsampling (q=1 means no downsampling)
    if downsample_q > 1:
        continuous = set(df.columns) - _NON_CONTINUOUS_SRC_COLS
        logger.info(f"Decimating by {downsample_q}x ({len(df)} → ~{len(df) // downsample_q} rows) ...")
        df = decimate_dataframe(df, q=downsample_q, continuous_cols=continuous)

    first_ts_ms = int(float(df.iloc[0]["timestamp"]) * 1000)

    logger.info(f"Normalizing {len(df)} rows ...")
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        rows.append(normalize_row(row, first_ts_ms, convert_mm=convert_mm))
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
            "joint_velocities": False,
            "joint_currents": True,
            "joint_temperatures": True,
            "joint_modes": False,
            "joint_control_outputs": False,
            "tcp_pose": True,
            "tcp_speed": False,
            "tcp_forces_torques": False,
            "setpoint_pos": True,
            "setpoint_speed": False,
            "setpoint_acc": False,
            "setpoint_tcp": False,
            "setpoint_tcp_speed": False,
            "motor_torque": True,
            "gripper": False,
            "robot_mode": False,
            "safety_mode": False,
            "digital_io": True,
            "voltages_current": False,
            "speed_scaling": True,
            "tool_momentum": False,
            "vibration": True,
            "acoustic_emission": False,
            "est_contact_force": False,
            "true_force": False,
            "protective_stop_state": False,
        }
        metadata = {
            "episode_id": episode_id,
            "source_file": input_file.name,
            "robot_type": "kuka_kr10",
            "data_source": "real_robot",
            "num_samples": len(rows),
            "schema_version": "1.0",
            "description": "KUKA KR10 real-robot data normalized from recorder CSV format",
            "unit_conventions": {
                "tcp_position": "metres (converted from mm)" if convert_mm else "metres (no conversion)",
                "tcp_orientation": "radians (converted from degrees)",
                "speed_scaling": "fraction [0-1] (converted from percent)",
            },
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
        description="Normalize real KUKA KR10 robot CSV data to UR3e schema JSON."
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
        "--no-mm-conversion",
        action="store_true",
        help="Skip mm→m conversion for TCP position (use if data is already in metres).")
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
                convert_mm=not args.no_mm_conversion,
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
