"""
Batch normalize multiple datasets to UR3e schema format.

Discovers and normalizes all Excel files in dataset folders, storing all
normalized outputs in a central 'normalized' directory with organized structure.

Usage:
    python -m src.data.batch_normalizer --datasets-dir datasets/open_datasets [--output normalized_data]
    python -m src.data.batch_normalizer --dataset datasets/open_datasets/dummy --output datasets/open_datasets/normalized
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import argparse

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")


logger = logging.getLogger(__name__)


# Mapping from Excel columns to UR3e schema columns
COLUMN_MAPPING = {
    # Time
    "timestamp_ms": "Timestamp",  # Will need to convert to milliseconds
    
    # INTENT - Joint-level commands (setpoint)
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
    "feedback_effort_0": "Current_J0",
    "feedback_effort_1": "Current_J1",
    "feedback_effort_2": "Current_J2",
    "feedback_effort_3": "Current_J3",
    "feedback_effort_4": "Current_J4",
    "feedback_effort_5": "Current_J5",
    
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
    Normalize a single row of data to UR3e schema format.
    
    Args:
        row: Pandas Series representing one row from the Excel file
        first_timestamp_ms: Milliseconds of first timestamp (for relative time calculation)
    
    Returns:
        Dictionary with normalized schema
    """
    result = {}
    
    # Process timestamp first to get relative time
    for schema_col, excel_col in COLUMN_MAPPING.items():
        if schema_col == "timestamp_ms":
            if excel_col and excel_col in row.index:
                value = row[excel_col]
                if pd.notna(value):
                    ts_ms = parse_timestamp(value)
                    if ts_ms is not None and first_timestamp_ms is not None:
                        result[schema_col] = ts_ms - first_timestamp_ms
                    else:
                        result[schema_col] = ts_ms
                else:
                    result[schema_col] = None
            else:
                result[schema_col] = None
        elif excel_col is None:
            result[schema_col] = None
        else:
            if excel_col in row.index:
                value = row[excel_col]
                if pd.notna(value):
                    if isinstance(value, str) and value.lower() in ['nan', 'none', '']:
                        result[schema_col] = None
                    else:
                        result[schema_col] = float(value) if isinstance(value, (int, float)) else value
                else:
                    result[schema_col] = None
            else:
                result[schema_col] = None
    
    return result


def normalize_dataset(
    input_file: Path,
    output_dir: Path,
    episode_id: str = None,
    include_metadata: bool = True,
) -> tuple[str, str]:
    """
    Normalize an entire dataset to UR3e schema format.
    
    Args:
        input_file: Path to Excel file
        output_dir: Directory to save normalized JSON
        episode_id: Episode identifier (defaults to input filename without extension)
        include_metadata: Whether to generate metadata file
    
    Returns:
        Tuple of (episode_id, json_file_path)
    """
    if episode_id is None:
        episode_id = input_file.stem
    
    logger.info(f"Loading dataset from {input_file.name}...")
    df = pd.read_excel(input_file)
    
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
        json.dump(normalized_rows, f, indent=2)
    
    logger.info(f"✓ Saved normalized data to {output_file.name}")
    
    # Generate metadata
    if include_metadata:
        metadata = {
            "episode_id": episode_id,
            "source_file": input_file.name,
            "num_samples": len(normalized_rows),
            "schema": "ur3e_v1",
            "first_timestamp_ms": first_timestamp_ms,
        }
        
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Saved metadata to {metadata_file.name}")
    
    return episode_id, str(output_file)


def discover_and_normalize(
    datasets_dir: Path,
    output_dir: Path,
    include_metadata: bool = True,
) -> List[Dict[str, str]]:
    """
    Discover all Excel files in dataset folders and normalize them.
    
    Args:
        datasets_dir: Root directory containing dataset folders
        output_dir: Central output directory for all normalized data
        include_metadata: Whether to generate metadata files
    
    Returns:
        List of normalized dataset info
    """
    results = []
    
    # Find all .xlsx files recursively, excluding temporary files (starting with ~$)
    excel_files = [f for f in datasets_dir.rglob("*.xlsx") if not f.name.startswith("~$")]
    
    if not excel_files:
        logger.warning(f"No Excel files found in {datasets_dir}")
        return results
    
    logger.info(f"Found {len(excel_files)} Excel files to normalize")
    
    for excel_file in excel_files:
        try:
            # Use relative path from datasets_dir to create episode ID
            rel_path = excel_file.relative_to(datasets_dir)
            episode_id = rel_path.stem
            
            # Create subdirectory structure preserving dataset organization
            dataset_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "default"
            episode_output_dir = output_dir / dataset_name
            
            logger.info(f"\n--- Normalizing: {excel_file.name} ---")
            ep_id, json_path = normalize_dataset(
                excel_file,
                episode_output_dir,
                episode_id=episode_id,
                include_metadata=include_metadata,
            )
            
            results.append({
                "dataset": dataset_name,
                "episode_id": ep_id,
                "json_file": json_path,
                "source": excel_file.name,
            })
            
        except Exception as e:
            logger.error(f"Failed to normalize {excel_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch normalize multiple datasets to UR3e schema"
    )
    
    # Mode 1: Normalize all datasets in a directory
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        help="Root directory containing dataset folders (discover all .xlsx files)",
    )
    
    # Mode 2: Normalize a single dataset
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Single dataset folder to normalize",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/open_datasets/normalized"),
        help="Output directory for all normalized data (default: data/open_datasets/normalized)",
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata file generation",
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
    
    # Determine mode and validate paths
    if args.datasets_dir:
        if not args.datasets_dir.exists():
            logger.error(f"Datasets directory not found: {args.datasets_dir}")
            return 1
        
        logger.info(f"Discovering datasets in {args.datasets_dir}...")
        results = discover_and_normalize(
            args.datasets_dir,
            args.output,
            include_metadata=not args.no_metadata,
        )
        
        logger.info(f"\n✓ Normalization complete! Processed {len(results)} datasets.")
        logger.info(f"Output saved to: {args.output}")
        
        # Print summary
        for result in results:
            logger.info(f"  - {result['dataset']}/{result['episode_id']}")
        
        return 0
    
    elif args.dataset:
        if not args.dataset.exists():
            logger.error(f"Dataset directory not found: {args.dataset}")
            return 1
        
        # Find .xlsx files in the dataset folder
        excel_files = list(args.dataset.glob("*.xlsx"))
        
        if not excel_files:
            logger.error(f"No .xlsx files found in {args.dataset}")
            return 1
        
        results = []
        dataset_name = args.dataset.name
        dataset_output_dir = args.output / dataset_name
        
        for excel_file in excel_files:
            try:
                logger.info(f"Normalizing: {excel_file.name}")
                ep_id, json_path = normalize_dataset(
                    excel_file,
                    dataset_output_dir,
                    episode_id=excel_file.stem,
                    include_metadata=not args.no_metadata,
                )
                results.append({
                    "episode_id": ep_id,
                    "json_file": json_path,
                })
            except Exception as e:
                logger.error(f"Failed to normalize {excel_file}: {e}")
                import traceback
                traceback.print_exc()
                return 1
        
        logger.info(f"\n✓ Normalization complete! Processed {len(results)} files.")
        logger.info(f"Output saved to: {dataset_output_dir}")
        
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
