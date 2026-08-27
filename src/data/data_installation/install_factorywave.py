"""
Download the ANON-ORG/FactoryWave dataset from HuggingFace.

Authenticates using the HF_TOKEN environment variable and downloads
the dataset to the local data directory.

Usage:
    python -m src.data.data_installation.install_factorywave
    python -m src.data.data_installation.install_factorywave --out-dir data/open_datasets/factorywave
    python -m src.data.data_installation.install_factorywave --subset ur5_pick_and_place
"""

from __future__ import annotations

import argparse
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_ID = "ANON-ORG/FactoryWave"


def _merge_parquet_parts(parts_dir: Path, output_path: Path) -> None:
    """Merge partitioned parquet files into a single file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required. Install with: pip install pyarrow")

    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        logger.warning(f"No parquet parts found in {parts_dir}")
        return

    if output_path.exists():
        logger.info(f"  Merged file already exists: {output_path}")
        return

    logger.info(f"  Merging {len(parts)} parquet parts into {output_path.name} ...")
    table = pq.read_table(parts_dir)
    pq.write_table(table, output_path)
    logger.info(f"  ✓ Merged ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def download_dataset(
    out_dir: Path,
    subset: str | None = None,
    token: str | None = None,
) -> None:
    """Download the FactoryWave dataset from HuggingFace Hub.

    Args:
        out_dir: Local directory to store the downloaded files.
        subset: Optional dataset subset/config to download. If None, downloads all.
        token: HuggingFace API token. If None, reads from HF_TOKEN env variable.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        )

    if token is None:
        token = os.environ.get("HF_TOKEN")

    # Try loading from .env if not in environment
    if not token:
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break

    if not token:
        raise ValueError(
            "HF_TOKEN not found. Set it in .env, as an environment variable, "
            "or pass --token."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {DATASET_ID} to {out_dir} ...")
    if subset:
        logger.info(f"  Subset: {subset}")

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=token,
        allow_patterns=f"{subset}/**" if subset else None,
    )

    # Merge ur_signals parts into a single parquet file
    ur_signals_dir = out_dir / "data" / "ur_signals"
    if ur_signals_dir.is_dir():
        _merge_parquet_parts(ur_signals_dir, out_dir / "data" / "ur_signals.parquet")

    logger.info(f"Done → {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Download the {DATASET_ID} dataset from HuggingFace."
    )
    repo_root = Path(__file__).resolve().parents[3]
    default_out_dir = repo_root / "data"

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help=f"Output directory (default: {default_out_dir})",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Download only a specific subset/config (e.g. 'ur5_pick_and_place')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (default: reads HF_TOKEN env variable)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        download_dataset(
            out_dir=args.out_dir,
            subset=args.subset,
            token=args.token,
        )
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
