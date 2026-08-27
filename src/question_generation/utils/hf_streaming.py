"""Stream-upload helper for large-scale QA generation.

When ``--upload-batch-size`` is set on a generator, this helper flushes the
output directory to a Hugging Face dataset every N produced items and deletes
the local files so disk usage stays bounded. Intended for jobs that produce
tens of thousands of QA pairs and would otherwise fill the disk.

Output formats:

  * ``jsonl`` (default, recommended), bundle each batch into a single
    line-delimited JSON shard ``level<N>_shard_<idx>.jsonl``. This is the
    canonical HF format for variable-schema QA datasets and renders nicely
    in the HF dataset viewer.
  * ``parquet``, same shard layout but in Apache Parquet. Smaller on disk,
    columnar, fastest for ``datasets.load_dataset`` consumption. Nested
    fields are preserved as JSON strings to avoid schema-mismatch errors
    across templates.
  * ``json``, legacy per-file layout (one ``.json`` per QA pair).
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, hf_hub_download

from src.pipeline.upload_qa_pairs import (
    DEFAULT_REPO_ID,
    resolve_hf_token,
    upload_qa_pairs)

logger = logging.getLogger(__name__)


_VALID_FORMATS = ("jsonl", "parquet", "json")
_NESTED_FIELDS_TO_STRINGIFY = ("options", "acceptance_bounds", "provenance", "context")


def _read_qa_files(files: List[Path]) -> List[dict]:
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning(f"[stream] skipping unreadable {f}: {exc}")
    return out


def _write_jsonl_shard(items: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        for item in items:
            out.write(json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n")


def _write_parquet_shard(items: List[dict], path: Path) -> None:
    """Flatten variable-schema QA items into a parquet-compatible row layout.

    Nested fields (``options``, ``acceptance_bounds``, ``provenance``,
    ``context``) become JSON strings so different templates can share a
    schema. Top-level scalars (``id``, ``level``, ``template_id``,
    ``template_type``, ``question``, ``answer``) are kept native.
    """
    import pandas as pd  # local import: parquet path is opt-in

    rows = []
    for item in items:
        row = {
            "id": item.get("id"),
            "level": item.get("level"),
            "template_id": item.get("template_id"),
            "template_type": item.get("template_type"),
            "question": item.get("question"),
            "answer": json.dumps(item.get("answer"), ensure_ascii=False)
            if not isinstance(item.get("answer"), (str, type(None)))
            else item.get("answer"),
        }
        for key in _NESTED_FIELDS_TO_STRINGIFY:
            row[key] = (
                json.dumps(item.get(key), ensure_ascii=False)
                if item.get(key) is not None
                else None
            )
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


class HfStreamUploader:
    """Flushes ``output_dir`` to HF every ``batch_size`` items.

    Each flush produces a single shard file (``jsonl`` / ``parquet``) and
    uploads it as one HF commit. The ``json`` format keeps the legacy
    per-file layout (one commit, many files). After the producing loop
    finishes, call ``flush_remaining`` once to ship any tail < batch_size.
    """

    def __init__(
        self,
        level: int,
        output_dir: Path,
        batch_size: int,
        dataset_folder: str,
        repo_id: str = DEFAULT_REPO_ID,
        private: bool = False,
        output_format: str = "jsonl",
        start_batch_index: Optional[int] = None) -> None:
        if output_format not in _VALID_FORMATS:
            raise ValueError(
                f"output_format must be one of {_VALID_FORMATS}, got {output_format!r}"
            )
        self.level = level
        self.output_dir = output_dir
        self.batch_size = int(batch_size)
        self.dataset_folder = dataset_folder
        self.repo_id = repo_id
        self.private = private
        self.output_format = output_format
        # When resuming, start_batch_index allows the caller to pass the
        # highest existing shard index on HF; new shards will be numbered
        # starting at start_batch_index+1, never overwriting earlier files.
        self.batch_index = int(start_batch_index) if start_batch_index else 0

    def maybe_flush(self, generated: int) -> None:
        """Call after each item write; flushes when ``generated`` hits a
        multiple of ``batch_size``.
        """
        if self.batch_size <= 0:
            return
        if generated <= 0 or generated % self.batch_size != 0:
            return
        self._flush()

    def flush_remaining(self) -> None:
        """Ship anything still on disk (tail batch). Safe to call when empty."""
        if self.batch_size <= 0:
            return
        files = sorted(self.output_dir.glob("*.json"))
        if files:
            self._flush()

    def _flush(self) -> None:
        files = sorted(self.output_dir.glob("*.json"))
        if not files:
            return
        self.batch_index += 1
        if self.output_format == "json":
            self._flush_per_file(files)
        else:
            self._flush_shard(files)
        # Only delete after upload succeeds (exceptions propagate from the
        # upload call and leave files in place for manual recovery).
        for f in files:
            try:
                f.unlink()
            except OSError as exc:
                logger.warning(f"[stream] could not delete {f}: {exc}")
        logger.info(
            f"[stream] batch {self.batch_index} uploaded and cleared from disk"
        )

    def _flush_per_file(self, files: List[Path]) -> None:
        commit_msg = (
            f"L{self.level} stream batch {self.batch_index} ({len(files)} items)"
        )
        logger.info(
            f"[stream] flushing {len(files)} L{self.level} files to "
            f"{self.repo_id}:{self.dataset_folder}/level_{self.level}/ "
            f"(batch {self.batch_index})"
        )
        upload_qa_pairs(
            json_files=files,
            repo_id=self.repo_id,
            dataset_folder=self.dataset_folder,
            level=self.level,
            private=self.private,
            commit_message=commit_msg,
            local_root=None)

    def _flush_shard(self, files: List[Path]) -> None:
        ext = "jsonl" if self.output_format == "jsonl" else "parquet"
        shard_name = f"level{self.level}_shard_{self.batch_index:04d}.{ext}"
        remote_path = f"{self.dataset_folder}/level_{self.level}/{shard_name}"
        logger.info(
            f"[stream] bundling {len(files)} items into {ext} shard "
            f"-> {self.repo_id}:{remote_path} (batch {self.batch_index})"
        )
        items = _read_qa_files(files)
        if not items:
            return
        token = resolve_hf_token()
        api = HfApi(token=token)
        with tempfile.TemporaryDirectory() as tmp:
            shard_path = Path(tmp) / shard_name
            if self.output_format == "jsonl":
                _write_jsonl_shard(items, shard_path)
            else:
                _write_parquet_shard(items, shard_path)
            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=(
                    f"L{self.level} shard {self.batch_index} ({len(items)} items, {ext})"
                ))


def list_completed_combos(
    repo_id: str,
    dataset_folder: str,
    level: int) -> Tuple[set, int]:
    """Return ``(completed_combos, max_shard_index)``.

    ``completed_combos`` is the set of ``(template_id, episode_stem)`` pairs
    already uploaded for this (level, folder) on HF. ``max_shard_index`` is
    the highest existing shard index in the folder (so the caller can resume
    new uploads without overwriting earlier shards).
    """
    completed: set = set()
    max_shard_idx = 0
    api = HfApi(token=resolve_hf_token())
    prefix = f"{dataset_folder}/level_{level}/"
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning(f"[resume] could not list HF repo files: {exc}")
        return completed, max_shard_idx
    shards = [f for f in files if f.startswith(prefix) and f.endswith(".jsonl")]
    if not shards:
        logger.info(f"[resume] no existing shards under {prefix}")
        return completed, max_shard_idx

    # Track highest shard index from the filename ``level{N}_shard_NNNN.jsonl``.
    import re as _re
    shard_re = _re.compile(rf"level{level}_shard_(\d+)\.jsonl$")
    for shard_path in shards:
        m = shard_re.search(shard_path)
        if m:
            try:
                max_shard_idx = max(max_shard_idx, int(m.group(1)))
            except (TypeError, ValueError):
                pass

    logger.info(f"[resume] found {len(shards)} existing shards (max idx={max_shard_idx}); downloading to scan combos")
    for shard_path in shards:
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=shard_path, repo_type="dataset"
            )
        except Exception as exc:
            logger.warning(f"[resume] download failed for {shard_path}: {exc}")
            continue
        try:
            with open(local, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    tid = item.get("template_id")
                    prov = item.get("provenance") or {}
                    # Single-episode templates use prov.episode; paired/ranking
                    # templates use prov.episode_a / prov.episodes (list).
                    eps: List[str] = []
                    if "episode" in prov:
                        eps.append(str(prov["episode"]))
                    elif "episode_a" in prov:
                        eps.append(str(prov["episode_a"]))
                    elif "episodes" in prov and isinstance(prov["episodes"], list):
                        for e in prov["episodes"]:
                            if isinstance(e, dict) and e.get("episode"):
                                eps.append(str(e["episode"]))
                                break  # primary episode only
                    if tid is not None and eps:
                        try:
                            completed.add((int(tid), eps[0]))
                        except (TypeError, ValueError):
                            pass
        except Exception as exc:
            logger.warning(f"[resume] parse failed for {shard_path}: {exc}")

    logger.info(f"[resume] {len(completed)} (template, episode) combos already on HF, will be skipped")
    return completed, max_shard_idx


def add_streaming_args(parser, default_repo: str = DEFAULT_REPO_ID) -> None:
    """Attach the stream-upload CLI flags to a generator's arg parser."""
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=0,
        help="Upload to HF every N produced items, then delete local copies. "
             "0 disables streaming (keep all items locally). Useful for "
             "large-scale jobs that would otherwise fill the disk.")
    parser.add_argument(
        "--upload-format",
        type=str,
        choices=list(_VALID_FORMATS),
        default="jsonl",
        help="HF shard format. 'jsonl' (default) bundles each batch into a "
             "single line-delimited JSON shard (HF-canonical for QA datasets). "
             "'parquet' is columnar and more efficient. 'json' keeps the "
             "legacy per-file layout (slow to load, many tiny files).")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=default_repo,
        help=f"HF dataset repo to upload to (default: {default_repo}). "
             f"Requires --upload-batch-size > 0 and --hf-dataset-folder.")
    parser.add_argument(
        "--hf-dataset-folder",
        type=str,
        default=None,
        help="Folder name inside the HF repo (e.g. factorynet_qa_150k). "
             "Required when --upload-batch-size > 0.")
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the HF dataset repo as private if it doesn't exist yet.")
    parser.add_argument(
        "--resume-from-hf",
        action="store_true",
        help="Before generating, query the HF dataset folder for already-"
             "uploaded shards and skip every (template, episode) combo "
             "already represented in them. Lets you resume an interrupted "
             "run without redoing work.")


def make_uploader_from_args(args, level: int, output_dir: Path) -> Optional[HfStreamUploader]:
    """Construct an uploader from parsed CLI args, or return None if streaming
    is disabled. Validates required flags.
    """
    size = int(getattr(args, "upload_batch_size", 0) or 0)
    if size <= 0:
        return None
    folder = getattr(args, "hf_dataset_folder", None)
    if not folder:
        raise SystemExit("--upload-batch-size > 0 requires --hf-dataset-folder")
    return HfStreamUploader(
        level=level,
        output_dir=output_dir,
        batch_size=size,
        dataset_folder=folder,
        repo_id=getattr(args, "hf_repo", DEFAULT_REPO_ID),
        private=bool(getattr(args, "hf_private", False)),
        output_format=getattr(args, "upload_format", "jsonl"))
