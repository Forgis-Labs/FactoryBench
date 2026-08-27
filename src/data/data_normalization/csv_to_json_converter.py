"""
Convert UR3e schema CSV files to normalized JSON format.

Handles CSV datasets that are already in UR3e schema format (e.g., dummy datasets),
converting them to the standard normalized JSON structure for question generation.

Usage:
    python -m src.data.csv_to_json_converter --input datasets/dummy/ABB.csv --output datasets/normalized_episodes/dummy/ABB.json
    python -m src.data.csv_to_json_converter --dataset datasets/dummy --output datasets/normalized_episodes
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import argparse

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")


logger = logging.getLogger(__name__)


def convert_csv_to_json(
    input_file: Path,
    output_dir: Path,
    episode_id: str = None,
    include_metadata: bool = True,
) -> tuple[str, str]:
    """
    Convert UR3e schema CSV file to normalized JSON format.
    
    Args:
        input_file: Path to CSV file (must have UR3e schema columns)
        output_dir: Directory to save normalized JSON
        episode_id: Episode identifier (defaults to input filename without extension)
        include_metadata: Whether to generate metadata file
    
    Returns:
        Tuple of (episode_id, json_file_path)
    """
    if episode_id is None:
        episode_id = input_file.stem
    
    logger.info(f"Loading CSV from {input_file.name}...")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        raise
    
    logger.info(f"Converting {len(df)} rows to JSON...")
    
    # Convert each row to a dictionary (CSV columns are already in UR3e schema)
    data = []
    for idx, row in df.iterrows():
        # Convert row to dict, replacing NaN with None
        row_dict = row.to_dict()
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
        data.append(row_dict)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} rows...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    output_file = output_dir / f"{episode_id}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"✓ Saved JSON to {output_file.name}")
    
    # Generate metadata
    if include_metadata:
        # Get first and last timestamp if available
        first_ts = data[0].get("timestamp_ms") if data else None
        last_ts = data[-1].get("timestamp_ms") if data else None
        
        metadata = {
            "episode_id": episode_id,
            "source_file": input_file.name,
            "num_samples": len(data),
            "schema": "ur3e_v1",
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "duration_ms": (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None,
        }
        
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Saved metadata to {metadata_file.name}")
    
    return episode_id, str(output_file)


def discover_and_convert(
    dataset_dir: Path,
    output_dir: Path,
    include_metadata: bool = True,
) -> List[Dict[str, str]]:
    """
    Discover all CSV files in a dataset folder and convert them.
    
    Args:
        dataset_dir: Directory containing CSV files
        output_dir: Root output directory for normalized data
        include_metadata: Whether to generate metadata files
    
    Returns:
        List of converted dataset info
    """
    results = []
    
    # Find all .csv files (not hidden)
    csv_files = [f for f in dataset_dir.glob("*.csv") if not f.name.startswith(".")]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {dataset_dir}")
        return results
    
    logger.info(f"Found {len(csv_files)} CSV files to convert")
    
    dataset_name = dataset_dir.name
    dataset_output_dir = output_dir / dataset_name
    
    for csv_file in csv_files:
        try:
            logger.info(f"\n--- Converting: {csv_file.name} ---")
            ep_id, json_path = convert_csv_to_json(
                csv_file,
                dataset_output_dir,
                episode_id=csv_file.stem,
                include_metadata=include_metadata,
            )
            
            results.append({
                "dataset": dataset_name,
                "episode_id": ep_id,
                "json_file": json_path,
                "source": csv_file.name,
            })
            
        except Exception as e:
            logger.error(f"Failed to convert {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert UR3e schema CSV files to normalized JSON"
    )
    
    # Mode 1: Convert single CSV file
    parser.add_argument(
        "--input",
        type=Path,
        help="Single CSV file to convert",
    )
    
    # Mode 2: Convert all CSV files in a dataset folder
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Dataset folder containing CSV files",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/normalized_episodes"),
        help="Output directory for normalized JSON (default: data/normalized_episodes)",
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
    
    # Determine mode
    if args.input:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        
        if not args.input.suffix.lower() == '.csv':
            logger.error(f"Input file must be CSV, got: {args.input.suffix}")
            return 1
        
        try:
            # Extract dataset name from parent folder
            dataset_name = args.input.parent.name
            dataset_output_dir = args.output / dataset_name
            
            logger.info(f"Converting single file: {args.input.name}")
            ep_id, json_path = convert_csv_to_json(
                args.input,
                dataset_output_dir,
                episode_id=args.input.stem,
                include_metadata=not args.no_metadata,
            )
            
            logger.info(f"\n✓ Conversion complete!")
            logger.info(f"Output: {json_path}")
            return 0
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.dataset:
        if not args.dataset.exists():
            logger.error(f"Dataset directory not found: {args.dataset}")
            return 1
        
        logger.info(f"Discovering CSV files in {args.dataset}...")
        results = discover_and_convert(
            args.dataset,
            args.output,
            include_metadata=not args.no_metadata,
        )
        
        logger.info(f"\n✓ Conversion complete! Processed {len(results)} files.")
        logger.info(f"Output saved to: {args.output}")
        
        # Print summary
        for result in results:
            logger.info(f"  - {result['dataset']}/{result['episode_id']}")
        
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
