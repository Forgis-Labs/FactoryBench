#!/usr/bin/env python3
"""
Download the "CNC Mill Tool Wear" dataset from Kaggle and unzip it.

Dataset slug used here:
  shasun/tool-wear-detection-in-cnc-mill

Simple run (with auto setup):
  python -m src.data.data_installation.install_cnc

First time setup:
  python -m src.data.data_installation.install_cnc --setup

Requirements:
  - Kaggle account (https://kaggle.com)
  - API token downloaded from Account -> Settings -> API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None


DATASET = "shasun/tool-wear-detection-in-cnc-mill"


def setup_kaggle_credentials_interactive() -> bool:
    """Interactively guide user through Kaggle setup if credentials missing."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        return True  # Already set up

    print("\n" + "=" * 70)
    print("Kaggle API Setup Required")
    print("=" * 70)
    print("\nFollow these steps:")
    print("  1. Go to https://kaggle.com/account/settings/api")
    print("  2. Click 'Create New API Token'")
    print("  3. This downloads 'kaggle.json'")
    print("\nThen paste your API credentials below:")
    print(f"(Will be saved to: {kaggle_json})\n")

    try:
        username = input("Kaggle Username: ").strip()
        key = input("Kaggle API Key: ").strip()

        if not username or not key:
            print("✗ Credentials cannot be empty.")
            return False

        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        
        # Secure: restrict file permissions (Windows: inherited from folder)
        kaggle_json.chmod(0o600)

        print(f"\n✓ Saved credentials to {kaggle_json}")
        return True

    except KeyboardInterrupt:
        print("\n✗ Setup cancelled.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def ensure_kaggle_json(interactive: bool = False) -> bool:
    """
    Check for Kaggle credentials.
    
    Returns True if valid credentials exist.
    If interactive=True, prompts user to set up credentials.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if kaggle_json.exists():
        return True

    if interactive:
        return setup_kaggle_credentials_interactive()

    print("=" * 70)
    print("Kaggle API credentials not found")
    print("=" * 70)
    print(f"\nExpected: {kaggle_json}")
    print("\nQuick fix:")
    print("  python -m src.data.data_installation.install_cnc --setup")
    print("\nOr manually:")
    print("  1. Download API token from https://kaggle.com/account/settings/api")
    print("  2. Save as ~/.kaggle/kaggle.json")
    print("=" * 70 + "\n")

    return False

    # Kaggle package reads KAGGLE_CONFIG_DIR if set; otherwise it uses ~/.kaggle
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_dir))


def unzip_all(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def dataset_already_extracted(out_dir: Path) -> bool:
    """Check if dataset is already extracted."""
    csv_files = list(out_dir.glob("*.csv"))
    return len(csv_files) > 0


def add_timestamp_column(out_dir: Path) -> None:
    """Add timestamp_ms column (starting at 0, incrementing by 100) to all CSV files."""
    if pd is None:
        print("⚠ pandas not installed, skipping timestamp column addition")
        return

    csv_files = sorted(out_dir.glob("*.csv"))
    if not csv_files:
        print("⚠ No CSV files found to process")
        return

    print(f"\nAdding timestamp_ms column to {len(csv_files)} CSV file(s)...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add timestamp_ms column: 0, 100, 200, 300, ...
            df.insert(0, "timestamp_ms", range(0, len(df) * 100, 100))
            df.to_csv(csv_file, index=False)
            print(f"  ✓ {csv_file.name}: added {len(df)} timestamps")
        except Exception as e:
            print(f"  ✗ {csv_file.name}: {e}")

    print()


def normalize_experiment_csv_names(out_dir: Path) -> None:
    """Rename experiment_0i.csv files to experiment_i.csv (strip zero padding)."""
    for csv_file in out_dir.glob("experiment_*.csv"):
        match = re.match(r"^(experiment_)0+(\d+)\.csv$", csv_file.name)
        if not match:
            continue
        prefix, num = match.groups()
        normalized = str(int(num))
        new_name = f"{prefix}{normalized}.csv"
        target = csv_file.parent / new_name
        if target.exists():
            continue
        csv_file.rename(target)


def normalize_experiment_folder_names(out_dir: Path) -> None:
    """Rename experiment_0i style folders to experiment_i (strip zero padding)."""
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^(experiment_)0+(\d+)$", child.name)
        if not match:
            continue
        prefix, num = match.groups()
        normalized = str(int(num))
        new_name = f"{prefix}{normalized}"
        target = child.parent / new_name
        if target.exists():
            for item in child.iterdir():
                shutil.move(str(item), str(target / item.name))
            child.rmdir()
        else:
            child.rename(target)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download CNC Mill Tool Wear dataset from Kaggle",
        epilog="First time? Run: python -m src.data.data_installation.install_cnc --setup",
    )
    repo_root = Path(__file__).resolve().parents[3]
    default_out_dir = repo_root / "data" / "open_datasets" / "cnc"
    
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out_dir,
        help=f"Output directory (default: {default_out_dir})",
    )
    ap.add_argument(
        "--setup",
        action="store_true",
        help="Interactive setup for Kaggle credentials (one-time)",
    )
    ap.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if dataset already extracted",
    )
    args = ap.parse_args()

    # Setup mode: just configure credentials
    if args.setup:
        print("Setting up Kaggle credentials...")
        if setup_kaggle_credentials_interactive():
            print("\n✓ Setup complete! Now run:")
            print(f"  python -m src.data.data_installation.install_cnc\n")
            sys.exit(0)
        else:
            sys.exit(1)

    # Check if already downloaded
    if dataset_already_extracted(args.out) and not args.force:
        print(f"✓ Dataset already extracted at {args.out}")
        print("  Use --force to re-download")
        sys.exit(0)

    # Validate Kaggle credentials (with interactive setup offer)
    if not ensure_kaggle_json(interactive=False):
        print("\nWould you like to set up credentials now? (y/n)")
        if input().strip().lower() == "y":
            if setup_kaggle_credentials_interactive():
                print("\n✓ Credentials set. Retrying download...\n")
                # Continue with download
            else:
                sys.exit(1)
        else:
            sys.exit(1)

    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise SystemExit(
            "Could not import Kaggle API.\n"
            "Install it with:\n"
            "  python -m pip install -U kaggle\n"
        ) from e

    api = KaggleApi()
    api.authenticate()

    print(f"{'='*70}")
    print(f"Downloading: {DATASET}")
    print(f"Output dir: {args.out.resolve()}")
    print(f"{'='*70}\n")

    # Kaggle downloads a single zip into the target directory by default
    args.out.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET, path=str(args.out), unzip=False, quiet=False)

    # Find the zip file Kaggle just downloaded
    zips = sorted(args.out.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise SystemExit(f"✗ No zip found in {args.out}. Download may have failed.")

    zip_path = zips[0]

    print(f"\nExtracting to: {args.out}")
    unzip_all(zip_path, args.out)

    normalize_experiment_folder_names(args.out)

    normalize_experiment_csv_names(args.out)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"✓ Removed zip file")

    # Add timestamp_ms column to all CSVs
    add_timestamp_column(args.out)

    print(f"\n{'='*70}")
    print(f"✓ Done! Dataset ready at:")
    print(f"  {args.out}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()