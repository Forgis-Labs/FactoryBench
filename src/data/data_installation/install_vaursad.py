#!/usr/bin/env python3
"""
Download voraus-AD (100 Hz or 500 Hz) and split into per-experiment parquet files.

Features:
- Validates parquet file size
- Streams the source parquet in batches (low RAM)
- Creates one parquet file per unique "sample" value (named experiment_{i}.parquet)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import requests
import time
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

from src.data._decimation import decimate_dataframe

URLS = {
    100: "https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet",
    500: "https://media.vorausrobotik.com/voraus-ad-dataset-500hz.parquet",
}

MIN_SIZE = {
    100: 900 * 1024 * 1024,         # 900 MB
    500: 4 * 1024 * 1024 * 1024,    # 4 GB
}


def download(url: str, dst: Path) -> None:
    print("Downloading dataset...")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()

        with dst.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = 100 * downloaded / total
                    speed = downloaded / max(1e-6, time.time() - t0) / (1024 * 1024)
                    print(f"\r{pct:5.1f}%  ({speed:6.1f} MB/s)", end="")

    print("\nDownload complete.")


def verify_file(path: Path, variant: int) -> bool:
    return path.exists() and path.stat().st_size >= MIN_SIZE[variant]


def convert_to_parquet_streaming(
    parquet_path: Path,
    out_dir: Path,
    max_timestamps: int | None = None,
    decimate_q: int = 10,
) -> None:
    """Stream parquet → per-sample parquet at full rate, then anti-alias decimate in place.

    Two-pass approach avoids cross-batch filter artifacts that would arise from
    decimating inside the streaming loop (a single sample can span many batches).
    """
    print("Opening parquet...")
    parquet_file = pq.ParquetFile(parquet_path)

    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows: {total_rows:,}")

    # --- Pass 1: stream full-rate data into per-sample parquet files --------
    writers: dict = {}
    rows_written: dict = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read 10x more than requested to account for decimation
    read_limit = max_timestamps * decimate_q if max_timestamps is not None else None
    total_read = 0

    try:
        with tqdm(total=total_rows, unit="rows", desc="Streaming") as pbar:
            for batch in parquet_file.iter_batches(batch_size=100_000):
                if read_limit is not None and total_read >= read_limit:
                    break

                df = batch.to_pandas()

                if "sample" not in df.columns:
                    raise ValueError("Column 'sample' not found in parquet file.")

                if read_limit is not None:
                    remaining = read_limit - total_read
                    df = df.iloc[:remaining]

                for sample_val, group in df.groupby("sample", sort=False):
                    safe_val = str(sample_val).replace("/", "_").replace("\\", "_")
                    out_path = out_dir / f"experiment_{safe_val}.parquet"

                    table = pa.Table.from_pandas(group, preserve_index=False)
                    if sample_val not in writers:
                        writers[sample_val] = pq.ParquetWriter(out_path, table.schema)
                        rows_written[sample_val] = 0
                    else:
                        table = table.cast(writers[sample_val].schema)
                    writers[sample_val].write_table(table)
                    rows_written[sample_val] += len(group)

                total_read += len(df)
                pbar.update(len(df))

    finally:
        for w in writers.values():
            w.close()

    # --- Pass 2: anti-alias decimate each per-sample parquet in-place -------
    _NON_CONTINUOUS = {"sample", "category"}
    print(f"\nDecimating {len(writers)} experiments by {decimate_q}x ...")

    for sample_val in tqdm(writers, desc="Decimating", unit="exp"):
        safe_val = str(sample_val).replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"experiment_{safe_val}.parquet"

        sample_df = pd.read_parquet(out_path)
        continuous = set(sample_df.columns) - _NON_CONTINUOUS
        decimated = decimate_dataframe(sample_df, q=decimate_q, continuous_cols=continuous)

        # Convert time/timestamp columns to milliseconds
        for col in decimated.columns:
            if col.lower() == "time" or "timestamp" in col.lower():
                decimated[col] = (decimated[col] * 1000).astype("int64")

        decimated.to_parquet(out_path, index=False)

    print(f"✓ All {len(writers)} experiments decimated and exported to {out_dir}\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download voraus-AD and split into per-experiment parquet files"
    )
    repo_root = Path(__file__).resolve().parents[3]
    default_out_dir = repo_root / "data" / "open_datasets" / "vorausad"
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="Where to store dataset files",
    )
    ap.add_argument(
        "--variant",
        type=int,
        default=100,
        choices=[100, 500],
        help="Dataset frequency variant in Hz (default: 100)",
    )
    ap.add_argument(
        "--max-timestamps",
        type=int,
        default=None,
        help="Optional limit on total number of rows to export",
    )
    ap.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip file size verification",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"vorausad-{args.variant}hz.parquet"
    experiments_dir = out_dir / "vorausad"

    if verify_file(parquet_path, args.variant):
        print(f"Found existing dataset: {parquet_path}")
    else:
        if parquet_path.exists():
            print("Parquet incomplete. Re-downloading.")
            parquet_path.unlink()
        download(URLS[args.variant], parquet_path)

        if not args.skip_verify and not verify_file(parquet_path, args.variant):
            raise RuntimeError("Downloaded file still looks invalid.")

    if max_timestamps := args.max_timestamps:
        if max_timestamps < 1:
            raise ValueError("--max-timestamps must be >= 1")

    if not experiments_dir.exists() or not any(experiments_dir.iterdir()):
        convert_to_parquet_streaming(parquet_path, experiments_dir, args.max_timestamps)
    else:
        print(f"Experiments directory already exists and is non-empty: {experiments_dir}")

    print(f"OK: {parquet_path}")
    print(f"Per-experiment parquet files: {experiments_dir}/")


if __name__ == "__main__":
    main()
