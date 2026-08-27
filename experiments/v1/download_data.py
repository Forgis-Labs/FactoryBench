"""Download test JSONL splits from the HuggingFace dataset.

Usage:
    # Ensure HF_TOKEN is set or run `huggingface-cli login` first
    python experiments/v1/download_data.py

Downloads level_1 and level_2 test.jsonl files to experiments/v1/data/.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "the authors/FactoryBench_QA_pairs"
OUTPUT_DIR = Path(__file__).parent / "data"


def resolve_token() -> str:
    """Try HF_TOKEN env var, then cached token."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:
        pass
    if not token:
        raise RuntimeError(
            "No HuggingFace token found. "
            "Run `huggingface-cli login` or set HF_TOKEN."
        )
    return token


def list_dataset_folders(token: str) -> list[str]:
    """List top-level folders in the dataset repo."""
    api = HfApi(token=token)
    files = list(api.list_repo_files(REPO_ID, repo_type="dataset"))
    folders = sorted({f.split("/")[0] for f in files if "/" in f})
    return folders


def download_splits(token: str, splits: list[str] = ("test",)) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_folder = "factorynet_qa_150k"
    levels = [1, 2]

    for split in splits:
        for level in levels:
            remote = f"{dataset_folder}/level_{level}/{split}.jsonl"
            out_name = f"level_{level}_{split}.jsonl"
            dest = OUTPUT_DIR / out_name

            if dest.exists():
                print(f"  {out_name} already exists, skipping")
                continue

            print(f"  Downloading {remote}...")
            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote,
                repo_type="dataset",
                token=token,
            )
            shutil.copy2(local, dest)
            size_mb = dest.stat().st_size / 1024 / 1024
            with open(dest) as f:
                n = sum(1 for _ in f)
            print(f"    -> {dest.name} ({size_mb:.1f} MB, {n:,} items)")

    print(f"\nDone. Files in {OUTPUT_DIR}/:")
    for p in sorted(OUTPUT_DIR.glob("*.jsonl")):
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"  {p.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        help="Splits to download (default: test). Use 'all' for train+validation+test.",
    )
    args = parser.parse_args()

    splits = ["train", "validation", "test"] if "all" in args.splits else args.splits
    token = resolve_token()
    download_splits(token, splits=splits)
