"""
Upload QA pair JSON files to a Hugging Face dataset repo, preserving them as
individual files under a ``<dataset_folder>/level_<N>/`` layout.

Target layout inside the HF repo (default ``the authors/FactoryBench_QA_pairs``)::

    the authors/FactoryBench_QA_pairs/
        factorynet_qa_260/
            level_1/
                level1_0000.json
                ...
            level_2/
            level_3/
            level_4/
        factorywave_qa_260/
            level_1/
            ...

Usage:
    python -m src.pipeline.upload_qa_pairs \
        --input datasets/questions/level1_pipeline \
        --dataset-folder factorynet_qa_260 \
        --level 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

load_dotenv()

DEFAULT_REPO_ID = "the authors/FactoryBench_QA_pairs"
ALLOWED_LEVELS = (1, 2, 3, 4)


def resolve_hf_token() -> str:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN (preferred), "
            "HUGGINGFACE_TOKEN, or HUGGING_FACE_HUB_TOKEN in your .env file."
        )
    return token


def collect_json_files(
    input_dir: Path | None,
    files: List[Path] | None,
    recursive: bool,
) -> List[Path]:
    if files:
        resolved = [f.resolve() for f in files]
        for f in resolved:
            if not f.is_file():
                raise FileNotFoundError(f"File does not exist: {f}")
            if f.suffix.lower() != ".json":
                raise ValueError(f"Not a .json file: {f}")
        return resolved

    assert input_dir is not None
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    pattern = "**/*.json" if recursive else "*.json"
    found = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not found:
        raise ValueError(f"No .json files found in {input_dir} (recursive={recursive}).")
    return found


def upload_qa_pairs(
    json_files: List[Path],
    repo_id: str,
    dataset_folder: str,
    level: int,
    private: bool,
    commit_message: str,
    local_root: Path | None,
) -> None:
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"--level must be one of {ALLOWED_LEVELS}, got {level}.")
    if not dataset_folder or "/" in dataset_folder or "\\" in dataset_folder:
        raise ValueError(
            f"--dataset-folder must be a single folder name (no slashes), got: {dataset_folder!r}"
        )

    token = resolve_hf_token()
    api = HfApi(token=token)

    print("=" * 60)
    print(f"Ensuring repo exists: {repo_id} (private={private})")
    print("=" * 60)
    # Probe first: tokens with only write-access to existing repos can't call create_repo
    # (it 403s on the namespace). Skip creation if the repo is already there.
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repo {repo_id} already exists; skipping create.")
    except RepositoryNotFoundError:
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Repo {repo_id} does not exist and your token cannot create it. "
                f"Either create the repo manually on the Hugging Face web UI, or "
                f"use a token with create-repo rights on the '{repo_id.split('/')[0]}' namespace."
            ) from exc

    remote_dir = f"{dataset_folder}/level_{level}"
    print("=" * 60)
    print(f"Staging {len(json_files)} file(s) for a single commit -> '{repo_id}:{remote_dir}/'")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        stage = Path(tmp_dir)
        for src in json_files:
            if local_root is not None:
                try:
                    rel = src.resolve().relative_to(local_root.resolve())
                    remote_name = str(rel).replace("\\", "/")
                except ValueError:
                    remote_name = src.name
            else:
                remote_name = src.name

            dst = stage / remote_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  - {src}  ->  {remote_dir}/{remote_name}")

        print("Uploading staged folder to Hugging Face (single commit)...")
        result = api.upload_folder(
            folder_path=str(stage),
            path_in_repo=remote_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    print(f"Upload complete. Commit: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload QA pair JSON files to a Hugging Face dataset repo, organized "
            "under <dataset_folder>/level_<N>/."
        )
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input",
        type=Path,
        help="Directory containing QA pair JSONs to upload.",
    )
    src_group.add_argument(
        "--files",
        type=Path,
        nargs="+",
        help="Explicit list of QA pair JSON files to upload.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When using --input, walk subdirectories (default: only top level).",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Top-level folder inside the HF repo (e.g. factorynet_qa_260, factorywave_qa_260).",
    )
    parser.add_argument(
        "--level",
        type=int,
        required=True,
        choices=ALLOWED_LEVELS,
        help="Level subfolder (1-4). Uploaded under level_<N>/.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Target Hugging Face dataset repo id (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create/keep the repo as public. Default is private.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload QA pairs",
        help="Base commit message for the uploads.",
    )

    args = parser.parse_args()

    json_files = collect_json_files(
        input_dir=args.input,
        files=args.files,
        recursive=args.recursive,
    )

    upload_qa_pairs(
        json_files=json_files,
        repo_id=args.repo_id,
        dataset_folder=args.dataset_folder,
        level=args.level,
        private=not args.public,
        commit_message=args.commit_message,
        local_root=args.input if args.recursive and args.input else None,
    )


if __name__ == "__main__":
    main()
