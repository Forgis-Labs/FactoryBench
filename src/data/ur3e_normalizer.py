"""
Normalize UR3+CobotOps dataset to UR3e schema format.

Converts the Excel-based UR3+CobotOps dataset to JSON format conforming to the
UR3e schema specified in ur3e_schema.md. The script maps raw columns to the
standardized schema, replacing missing values with null.

Usage:
    python -m factorybench.data.ur3e_normalizer --input <excel_file> --output <output_dir> [--episode-id <id>]
"""

import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import argparse

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


# Mapping from Excel columns to UR3e schema columns
COLUMN_MAPPING = {
    # Time
    "timestamp_ms": "Timestamp",  # Will need to convert to milliseconds
    
    # INTENT - Joint-level commands (setpoint)
    # These are not in the current dataset - will be null
    "setpoint_pos_0": None,
    "setpoint_pos_1": None,
    "setpoint_pos_2": None,
    "setpoint_pos_3": None,
    "setpoint_pos_4": None,
    "setpoint_pos_5": None,
    "setpoint_speed_0": None,
    "setpoint_speed_1": None,
    "setpoint_speed_2": None,
    "setpoint_speed_3": None,
    "setpoint_speed_4": None,
    "setpoint_speed_5": None,
    "setpoint_acc_0": None,
    "setpoint_acc_1": None,
    "setpoint_acc_2": None,
    "setpoint_acc_3": None,
    "setpoint_acc_4": None,
    "setpoint_acc_5": None,
    
    # INTENT - TCP commands (not in dataset)
    "setpoint_tcp_0": None,
    "setpoint_tcp_1": None,
    "setpoint_tcp_2": None,
    "setpoint_tcp_3": None,
    "setpoint_tcp_4": None,
    "setpoint_tcp_5": None,
    
    # INTENT - Gripper command
    "gripper_command": None,  # Not available in dataset
    
    # CONTEXT - Dynamic operating conditions
    "joint_temp_0": "Temperature_T0",
    "joint_temp_1": "Temperature_J1",
    "joint_temp_2": "Temperature_J2",
    "joint_temp_3": "Temperature_J3",
    "joint_temp_4": "Temperature_J4",
    "joint_temp_5": "Temperature_J5",
    "main_voltage": None,  # Not in dataset
    "safety_mode": None,  # Not in dataset
    
    # OUTCOME - Joint feedback
    "feedback_pos_0": None,  # Not in dataset
    "feedback_pos_1": None,
    "feedback_pos_2": None,
    "feedback_pos_3": None,
    "feedback_pos_4": None,
    "feedback_pos_5": None,
    "feedback_speed_0": "Speed_J0",
    "feedback_speed_1": "Speed_J1",
    "feedback_speed_2": "Speed_J2",
    "feedback_speed_3": "Speed_J3",
    "feedback_speed_4": "Speed_J4",
    "feedback_speed_5": "Speed_J5",
    
    # OUTCOME - Effort/current
    "effort_current_0": "Current_J0",
    "effort_current_1": "Current_J1",
    "effort_current_2": "Current_J2",
    "effort_current_3": "Current_J3",
    "effort_current_4": "Current_J4",
    "effort_current_5": "Current_J5",
    
    # OUTCOME - System protection state
    "protective_stop_state": "Robot_ProtectiveStop",
    
    # OUTCOME - Vibration (not in dataset)
    "vibration_0": None,
    "vibration_1": None,
    "vibration_2": None,
    
    # OUTCOME - Acoustic emission (not in dataset)
    "acoustic_0": None,
    
    # OUTCOME - Contact forces/torques (not in dataset)
    "est_contact_force_0": None,
    "est_contact_force_1": None,
    "est_contact_force_2": None,
    "est_contact_force_3": None,
    "est_contact_force_4": None,
    "est_contact_force_5": None,
    "true_force_0": None,
    "true_force_1": None,
    "true_force_2": None,
    "true_force_3": None,
    "true_force_4": None,
    "true_force_5": None,
}


def parse_timestamp(timestamp_str: str) -> int:
    """
    Convert ISO timestamp string to milliseconds since episode start.
    
    Since we don't have per-episode start times, we'll track the first timestamp
    and calculate relative milliseconds.
    """
    try:
        # Strip any surrounding quotes and whitespace
        cleaned = str(timestamp_str).strip().strip('"').strip("'")
        # Parse ISO format: "2022-10-26T08:17:21.847Z"
        dt = datetime.fromisoformat(cleaned.replace('Z', '+00:00'))
        return int(dt.timestamp() * 1000)
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return None


def normalize_row(row: pd.Series, first_timestamp_ms: int = None) -> Dict[str, Any]:
    """
    Convert a single Excel row to UR3e schema format.
    
    Args:
        row: Pandas Series representing one row
        first_timestamp_ms: Milliseconds of first sample (for relative timestamps)
    
    Returns:
        Dictionary conforming to UR3e schema
    """
    result = {}
    
    # Process timestamp first to get relative milliseconds
    if "Timestamp" in row and pd.notna(row["Timestamp"]):
        abs_ms = parse_timestamp(row["Timestamp"])
        if first_timestamp_ms is not None and abs_ms is not None:
            result["timestamp_ms"] = abs_ms - first_timestamp_ms
        else:
            result["timestamp_ms"] = abs_ms
    else:
        result["timestamp_ms"] = None
    
    # Map all schema columns
    for schema_col, excel_col in COLUMN_MAPPING.items():
        if schema_col == "timestamp_ms":
            continue  # Already handled
        
        if excel_col is None:
            # Column not in source data
            result[schema_col] = None
        else:
            # Get value from Excel, convert to proper type
            if excel_col in row.index:
                value = row[excel_col]
                if pd.isna(value):
                    result[schema_col] = None
                else:
                    # Convert to native Python type for JSON serialization
                    if isinstance(value, (int, float, str, bool, type(None))):
                        result[schema_col] = float(value) if isinstance(value, (int, float)) else value
                    else:
                        result[schema_col] = str(value)
            else:
                result[schema_col] = None
    
    return result


def normalize_dataset(
    input_file: Path,
    output_dir: Path,
    episode_id: str = "ur3e_episode_001",
    include_metadata: bool = True,
) -> None:
    """
    Normalize the entire UR3+CobotOps dataset to UR3e schema format.
    
    Args:
        input_file: Path to Excel file
        output_dir: Directory to save normalized JSON
        episode_id: Episode identifier
        include_metadata: Whether to generate metadata file
    """
    logger.info(f"Loading dataset from {input_file.name}...")
    df = pd.read_excel(input_file)
    
    # Anti-aliased downsampling (default: halve the sampling frequency)
    _CONTINUOUS_EXCEL_COLS = {
        "Speed_J0", "Speed_J1", "Speed_J2", "Speed_J3", "Speed_J4", "Speed_J5",
        "Current_J0", "Current_J1", "Current_J2", "Current_J3", "Current_J4", "Current_J5",
        "Temperature_T0", "Temperature_J1", "Temperature_J2", "Temperature_J3", "Temperature_J4", "Temperature_J5",
    }
    df = decimate_dataframe(df, q=2, continuous_cols=_CONTINUOUS_EXCEL_COLS)

    logger.info(f"Normalizing {len(df)} rows...")

    # Get first timestamp for relative time calculation
    first_timestamp_ms = None
    if "Timestamp" in df.columns and len(df) > 0:
        first_ts = df.iloc[0]["Timestamp"]
        if pd.notna(first_ts):
            first_timestamp_ms = parse_timestamp(first_ts)
    
    # Normalize all rows
    normalized_rows = []
    for idx, row in df.iterrows():
        normalized_row = normalize_row(row, first_timestamp_ms)
        normalized_rows.append(normalized_row)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} rows...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save normalized data as JSON
    output_file = output_dir / f"{episode_id}.json"
    with open(output_file, "w") as f:
        json.dump(normalized_rows, f, indent=2, cls=_NaNSafeEncoder)
    
    logger.info(f"✓ Saved normalized data to {output_file.name}")
    
    # Generate metadata
    if include_metadata:
        metadata = {
            "episode_id": episode_id,
            "source_file": input_file.name,
            "num_samples": len(normalized_rows),
            "schema_version": "1.0",
            "description": "UR3e time-series dataset normalized from UR3+CobotOps Excel format",
            "columns": {
                "intent": {
                    "joint_commands": [f"setpoint_pos_{i}" for i in range(6)] +
                                     [f"setpoint_speed_{i}" for i in range(6)] +
                                     [f"setpoint_acc_{i}" for i in range(6)],
                    "tcp_commands": [f"setpoint_tcp_{i}" for i in range(6)],
                    "gripper_command": "gripper_command",
                },
                "context": [f"joint_temp_{i}" for i in range(6)] + ["main_voltage", "safety_mode"],
                "outcome": {
                    "joint_feedback": [f"feedback_pos_{i}" for i in range(6)] +
                                     [f"feedback_speed_{i}" for i in range(6)],
                    "joint_effort": [f"effort_current_{i}" for i in range(6)],
                    "system_protection": ["protective_stop_state"],
                    "vibration": [f"vibration_{i}" for i in range(3)],
                    "acoustic": ["acoustic_0"],
                    "forces": [f"est_contact_force_{i}" for i in range(6)] +
                             [f"true_force_{i}" for i in range(6)],
                },
            },
            "available_data": {
                "setpoint (commands)": False,
                "tcp_commands": False,
                "gripper": False,
                "joint_temperatures": True,
                "main_voltage": False,
                "safety_mode": False,
                "joint_positions": False,
                "joint_velocities": True,
                "joint_currents": True,
                "protective_stop": True,
                "vibration": False,
                "acoustic_emission": False,
                "contact_forces": False,
            },
            "timestamp_info": {
                "format": "milliseconds since episode start",
                "first_timestamp_ms": first_timestamp_ms,
                "last_timestamp_ms": normalized_rows[-1]["timestamp_ms"] if normalized_rows else None,
            },
        }
        
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Saved metadata to {metadata_file.name}")


def main():
    """Command-line interface for UR3e normalization."""
    parser = argparse.ArgumentParser(
        description="Normalize UR3+CobotOps dataset to UR3e schema JSON format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Excel file to normalize",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for normalized JSON files",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default="ur3e_episode_001",
        help="Episode identifier (default: ur3e_episode_001)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip creating metadata file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # Validate paths
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    try:
        # If output is a .json file, extract directory and episode_id
        if args.output.suffix == '.json':
            output_dir = args.output.parent
            episode_id = args.output.stem
        else:
            output_dir = args.output
            episode_id = args.episode_id
        
        normalize_dataset(
            args.input,
            output_dir,
            episode_id=episode_id,
            include_metadata=not args.no_metadata,
        )
        logger.info("Normalization complete!")
        return 0
    
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
