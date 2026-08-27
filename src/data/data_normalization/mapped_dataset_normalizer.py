"""
Normalize datasets to UR3e schema using feature-mapping JSON files.

Inputs may be parquet or CSV (auto-detected by extension).

Usage:
  python -m src.data.data_normalization.mapped_dataset_normalizer \
    --dataset aursad \
    --input data/open_datasets/aursad \
    --output data/normalized_episodes \
    --episode-column episode_id \
    --max-episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

logger = logging.getLogger(__name__)


def iter_input_chunks(path: Path, chunksize: int = 10000) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from a CSV or Parquet file."""
    if path.suffix.lower() == ".parquet":
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
    else:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            yield chunk


def read_columns(path: Path, columns: List[str]) -> pd.DataFrame:
    """Read selected columns from a CSV or Parquet file."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def peek_columns(path: Path) -> List[str]:
    """Return the column names of a CSV or Parquet file without reading rows."""
    if path.suffix.lower() == ".parquet":
        return pq.ParquetFile(path).schema_arrow.names
    return list(pd.read_csv(path, nrows=0, low_memory=False).columns)


AXES = list(range(6))
FORCE_TORQUE = list(range(6))
VIB = list(range(3))


def expand_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    for dest_key, src_key in mapping.items():
        if "{axis}" in dest_key or "{axis}" in src_key:
            for axis in AXES:
                expanded[dest_key.replace("{axis}", str(axis))] = src_key.replace("{axis}", str(axis))
            continue
        if "{i}" in dest_key or "{i}" in src_key:
            for i in FORCE_TORQUE:
                expanded[dest_key.replace("{i}", str(i))] = src_key.replace("{i}", str(i))
            continue
        if "{k}" in dest_key or "{k}" in src_key:
            for k in VIB:
                expanded[dest_key.replace("{k}", str(k))] = src_key.replace("{k}", str(k))
            continue
        expanded[dest_key] = src_key
    return expanded


def load_mapping(dataset_name: str, repo_root: Path) -> Tuple[Dict[str, str], List[str], Dict[str, Any], Dict[str, Any], Optional[int]]:
    mapping_path = repo_root / "data" / "mappings_of_features" / f"{dataset_name}.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    mapping = expand_mapping(config.get("mapping", {}))
    absent = config.get("absent", [])
    faults = config.get("faults", {})
    phase_map = config.get("phase_map", {})
    machine_id = config.get("machine_id")
    return mapping, absent, faults, phase_map, machine_id


def find_input_files(dataset_name: str, repo_root: Path, input_path: Optional[Path]) -> List[Path]:
    """Discover per-experiment input files (.parquet preferred, .csv fallback)."""
    def _discover(directory: Path) -> List[Path]:
        for pattern in ("experiment_*.parquet", "experiment_*.csv", "*.parquet", "*.csv"):
            hits = sorted(directory.glob(pattern))
            if hits:
                return hits
        return []

    if input_path:
        if input_path.is_dir():
            files = _discover(input_path)
            if not files:
                raise FileNotFoundError(f"No parquet or CSV files found in {input_path}")
            return files
        return [input_path]

    dataset_dir = repo_root / "data" / "open_datasets" / dataset_name
    files = _discover(dataset_dir)
    if not files:
        raise FileNotFoundError(f"No parquet or CSV files found in {dataset_dir}")
    return files


def build_schema_fields(mapping: Dict[str, str], absent: List[str]) -> List[str]:
    fields = set(mapping.keys()) | set(absent)
    return sorted(fields)


def remove_null_features(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Remove any feature (key) that is null in all rows."""
	if not rows:
		return rows
	
	# Identify features that have at least one non-null value
	features_with_values: set = set()
	for row in rows:
		for key, value in row.items():
			if value is not None:
				features_with_values.add(key)
	
	# Filter rows to keep only features with at least one non-null value
	cleaned_rows: List[Dict[str, Any]] = []
	for row in rows:
		cleaned_rows.append({k: v for k, v in row.items() if k in features_with_values})
	
	return cleaned_rows


def normalize_fault_id(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def map_fault_label(value: Any, faults: Dict[str, Any]) -> Any:
    if value is None or not faults:
        return value
    key = str(value)
    if key in faults:
        return faults[key]
    return value


def map_phase_id(value: Any, phase_map: Dict[str, Any]) -> Any:
    if value is None:
        return None
    # No phase_map: source already provides canonical task_phase values; pass through as-is.
    if not phase_map:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    try:
        key = str(int(value))
    except (TypeError, ValueError):
        key = str(value)
    if key in phase_map:
        return phase_map[key]
    return None


def build_row_dict(
    row: pd.Series,
    mapping: Dict[str, str],
    schema_fields: List[str],
    faults: Dict[str, Any],
    phase_map: Dict[str, Any],
    round_floats: bool = False,
) -> Dict[str, Any]:
    row_dict: Dict[str, Any] = {}
    for out_field in schema_fields:
        if out_field in mapping:
            src_field = mapping[out_field]
            value = row.get(src_field, None)
            if isinstance(value, pd.Series):
                print(f"⚠ Skipping row: duplicate column '{src_field}' found (got {len(value)} values).")
                return None
            if value is not None and pd.isna(value):
                value = None
            if out_field == "fault_label":
                value = map_fault_label(value, faults)
            elif out_field == "task_phase":
                value = map_phase_id(value, phase_map)
            # Round floats if requested
            if round_floats and value is not None and isinstance(value, (float, np.floating)):
                value = round(value, 2)
                # Remove trailing zeros
                if value == int(value):
                    value = int(value)
            row_dict[out_field] = value
        else:
            row_dict[out_field] = None
    return row_dict


def build_episode_rows(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    absent: List[str],
    schema_fields: List[str],
    faults: Dict[str, Any],
    phase_map: Dict[str, Any],
    round_floats: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = build_row_dict(row, mapping, schema_fields, faults, phase_map, round_floats)
        if row_dict is not None:
            rows.append(row_dict)

    return rows


def write_episode(
    episode_rows: List[Dict[str, Any]],
    output_dir: Path,
    episode_id: str,
    source_file: str,
    include_metadata: bool,
    machine_id: Optional[int] = None,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = remove_null_features(episode_rows)
    output_file = output_dir / f"{episode_id}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(episode_rows, f, indent=2)

    if include_metadata:
        first_ts = episode_rows[0].get("timestamp_ms") if episode_rows else None
        last_ts = episode_rows[-1].get("timestamp_ms") if episode_rows else None
        metadata = {
            "episode_id": episode_id,
            "source_file": source_file,
            "num_samples": len(episode_rows),
            "schema": "ur3e_v1",
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "duration_ms": (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None,
            "machine_id": machine_id,
        }
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return str(output_file)


def write_episode_streaming(
    episode_rows: Iterable[Dict[str, Any]],
    output_dir: Path,
    episode_id: str,
    source_file: str,
    include_metadata: bool,
    machine_id: Optional[int] = None,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{episode_id}.json"

    # Collect all rows first to identify null features
    all_rows = list(episode_rows)
    all_rows = remove_null_features(all_rows)

    num_samples = 0
    first_ts = None
    last_ts = None

    with output_file.open("w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for row in all_rows:
            if not first:
                f.write(",\n")
            json.dump(row, f, indent=2)
            first = False
            num_samples += 1
            ts = row.get("timestamp_ms")
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        f.write("\n]\n")

    if include_metadata:
        metadata = {
            "episode_id": episode_id,
            "source_file": source_file,
            "num_samples": num_samples,
            "schema": "ur3e_v1",
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "duration_ms": (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else None,
            "machine_id": machine_id,
        }
        metadata_file = output_dir / f"{episode_id}_metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return str(output_file)


def normalize_dataset(
    dataset_name: str,
    input_path: Path,
    output_root: Path,
    episode_column: Optional[str],
    max_episodes: Optional[int],
    episode_size: int,
    include_metadata: bool,
    output_basename: Optional[str] = None,
    round_floats: bool = False,
) -> List[Dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[3]
    mapping, absent, faults, phase_map, machine_id = load_mapping(dataset_name, repo_root)
    schema_fields = build_schema_fields(mapping, absent)

    results: List[Dict[str, str]] = []
    # Check if output_root already ends with dataset_name to avoid double nesting
    if output_root.name == dataset_name:
        output_dir = output_root
    else:
        output_dir = output_root / dataset_name

    if episode_column:
        class EpisodeWriter:
            def __init__(self, episode_id: str) -> None:
                self.episode_id = episode_id
                self.output_file = output_dir / f"{episode_id}.json"
                self.handle = self.output_file.open("w", encoding="utf-8")
                self.handle.write("[\n")
                self.first = True
                self.num_samples = 0
                self.first_ts = None
                self.last_ts = None

            def write_row(self, row_dict: Dict[str, Any]) -> None:
                if not self.first:
                    self.handle.write(",\n")
                json.dump(row_dict, self.handle, indent=2)
                self.first = False
                self.num_samples += 1
                ts = row_dict.get("timestamp_ms")
                if ts is not None:
                    if self.first_ts is None:
                        self.first_ts = ts
                    self.last_ts = ts

            def close(self) -> None:
                self.handle.write("\n]\n")
                self.handle.close()

        writers: Dict[str, EpisodeWriter] = {}
        episode_order: List[str] = []
        stop_reading = False

        episode_prefix = output_basename or dataset_name

        for chunk in iter_input_chunks(input_path):
            if episode_column not in chunk.columns:
                raise ValueError(f"Episode column '{episode_column}' not found in {input_path.name}")

            for _, row in chunk.iterrows():
                ep_value = row.get(episode_column)
                if pd.isna(ep_value):
                    ep_value = "unknown"
                ep_key = str(ep_value)

                if ep_key not in writers:
                    if max_episodes is not None and len(episode_order) >= max_episodes:
                        stop_reading = True
                        break
                    ep_id = f"{episode_prefix}_{ep_key}"
                    writers[ep_key] = EpisodeWriter(ep_id)
                    episode_order.append(ep_key)

                row_dict = build_row_dict(row, mapping, schema_fields, faults, phase_map, round_floats)
                writers[ep_key].write_row(row_dict)

            if stop_reading:
                break

        for ep_key in episode_order:
            writer = writers[ep_key]
            writer.close()
            json_path = str(output_dir / f"{writer.episode_id}.json")

            if include_metadata:
                metadata = {
                    "episode_id": writer.episode_id,
                    "source_file": input_path.name,
                    "num_samples": writer.num_samples,
                    "schema": "ur3e_v1",
                    "first_timestamp_ms": writer.first_ts,
                    "last_timestamp_ms": writer.last_ts,
                    "duration_ms": (writer.last_ts - writer.first_ts)
                    if (writer.first_ts is not None and writer.last_ts is not None)
                    else None,
                    "machine_id": machine_id,
                }
                metadata_file = output_dir / f"{writer.episode_id}_metadata.json"
                with metadata_file.open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

            results.append({
                "dataset": dataset_name,
                "episode_id": writer.episode_id,
                "json_file": json_path,
            })
    else:
        if max_episodes is not None and max_episodes < 1:
            return results

        episode_count = 0
        row_in_episode = 0
        stop_reading = False
        all_rows: List[Dict[str, Any]] = []

        for chunk in iter_input_chunks(input_path):
            for _, row in chunk.iterrows():
                if stop_reading:
                    break

                row_in_episode += 1
                row_dict = build_row_dict(row, mapping, schema_fields, faults, phase_map, round_floats)
                all_rows.append(row_dict)

                if row_in_episode >= episode_size:
                    episode_count += 1
                    row_in_episode = 0
                    if max_episodes is not None and episode_count >= max_episodes:
                        stop_reading = True
                        break

            if stop_reading:
                break

        output_dir.mkdir(parents=True, exist_ok=True)
        all_rows = remove_null_features(all_rows)
        output_name = output_basename or dataset_name
        output_file = output_dir / f"{output_name}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)

        json_path = str(output_file)
        results.append({
            "dataset": dataset_name,
            "episode_id": output_name,
            "json_file": json_path,
        })

        if include_metadata:
            first_ts = all_rows[0].get("timestamp_ms") if all_rows else None
            last_ts = all_rows[-1].get("timestamp_ms") if all_rows else None
            metadata = {
                "episode_id": output_name,
                "source_file": input_path.name,
                "num_samples": len(all_rows),
                "num_episodes": episode_count if row_in_episode == 0 else episode_count + 1,
                "schema": "ur3e_v1",
                "first_timestamp_ms": first_ts,
                "last_timestamp_ms": last_ts,
                "duration_ms": (last_ts - first_ts)
                if (first_ts is not None and last_ts is not None)
                else None,
                "machine_id": machine_id,
            }
            metadata_file = output_dir / f"{output_name}_metadata.json"
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize dataset using feature mapping")
    parser.add_argument("--dataset", required=True, help="Dataset name (mapping file in data/mappings_of_features)")
    parser.add_argument("--input", type=Path, help="Path to input file or directory (parquet or CSV; defaults to data/open_datasets/<dataset>/)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/normalized_episodes"),
        help="Output directory for normalized JSON",
    )
    parser.add_argument(
        "--episode-column",
        type=str,
        default=None,
        help="Optional column to group rows into episodes",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="How many episodes to normalize (default: all)",
    )
    parser.add_argument(
        "--episode-size",
        type=int,
        default=1,
        help="Number of rows per episode (default: 1 = each row is an episode)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata file generation",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: keep only 2 samples per fault_label",
    )
    parser.add_argument(
        "--round",
        action="store_true",
        help="Round float values to 2 decimals and remove trailing zeros",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[3]
    input_files = find_input_files(args.dataset, repo_root, args.input)

    results: List[Dict[str, str]] = []
    multi_input = len(input_files) > 1

    # In test mode, filter which experiments to process
    if args.test:
        # Detect fault label column
        fault_col = None
        sample_cols = peek_columns(input_files[0])
        if "fault_label" in sample_cols:
            fault_col = "fault_label"
        elif "label" in sample_cols:
            fault_col = "label"

        if fault_col:
            # For each experiment, get its fault label
            exp_labels: Dict[Path, Any] = {}
            for file_path in input_files:
                df = read_columns(file_path, [fault_col])
                if not df.empty:
                    label = df[fault_col].iloc[0]
                    exp_labels[file_path] = label

            # Keep only 2 experiments per label
            label_counts: Dict[Any, int] = {}
            files_to_process: List[Path] = []
            for file_path in input_files:
                label = exp_labels.get(file_path)
                if label is not None:
                    count = label_counts.get(label, 0)
                    if count < 2:
                        files_to_process.append(file_path)
                        label_counts[label] = count + 1

            logger.info(f"Test mode: selected {len(files_to_process)} experiments (2 per label)")
            input_files = files_to_process
        else:
            logger.warning(f"No fault_label or label column found")

    # Process selected files through normal pipeline
    iterator = tqdm(input_files, desc=f"Normalizing {args.dataset}", unit="exp") if len(input_files) > 1 else input_files
    for file_path in iterator:
        output_basename = None
        if multi_input or file_path.stem.startswith("experiment_"):
            output_basename = file_path.stem

        results.extend(
            normalize_dataset(
                dataset_name=args.dataset,
                input_path=file_path,
                output_root=args.output,
                episode_column=args.episode_column,
                max_episodes=args.max_episodes,
                episode_size=args.episode_size,
                include_metadata=not args.no_metadata,
                output_basename=output_basename,
                round_floats=args.round,
            )
        )

    if not results:
        logger.warning("No episodes were normalized.")
        return 1

    logger.info(f"✓ Normalization complete! Processed {len(results)} episode(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
