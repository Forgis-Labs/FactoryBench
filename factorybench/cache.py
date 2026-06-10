"""Filesystem helpers for the on-disk judge-vote cache.

The cache lives at ``~/.cache/factorybench/judges/`` by default (see
:class:`factorybench.judges.JudgePanel`). Each cache entry is a single
``<sha256>.json`` file. These helpers let users inspect and clear it from the
CLI without touching ``JudgePanel`` machinery.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_JUDGE_CACHE_DIR = Path.home() / ".cache" / "factorybench" / "judges"


@dataclass
class CacheStats:
    path: Path
    exists: bool
    file_count: int
    total_bytes: int
    oldest_iso: str | None  # ISO 8601 of the oldest entry, or None when empty
    newest_iso: str | None

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "total_mb": round(self.total_mb, 4),
            "oldest_iso": self.oldest_iso,
            "newest_iso": self.newest_iso,
        }


def judge_cache_stats(cache_dir: str | Path | None = None) -> CacheStats:
    """Summarize a judge-cache directory."""
    path = Path(cache_dir) if cache_dir else DEFAULT_JUDGE_CACHE_DIR
    if not path.exists():
        return CacheStats(path=path, exists=False, file_count=0, total_bytes=0,
                          oldest_iso=None, newest_iso=None)

    count = 0
    total = 0
    oldest = None
    newest = None
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        count += 1
        try:
            stat = entry.stat()
        except OSError:
            continue
        total += stat.st_size
        if oldest is None or stat.st_mtime < oldest:
            oldest = stat.st_mtime
        if newest is None or stat.st_mtime > newest:
            newest = stat.st_mtime
    return CacheStats(
        path=path,
        exists=True,
        file_count=count,
        total_bytes=total,
        oldest_iso=_to_iso(oldest),
        newest_iso=_to_iso(newest),
    )


def clear_judge_cache(cache_dir: str | Path | None = None) -> int:
    """Remove every cache file under ``cache_dir``. Returns the number of files deleted."""
    path = Path(cache_dir) if cache_dir else DEFAULT_JUDGE_CACHE_DIR
    if not path.exists():
        return 0
    removed = 0
    for entry in path.iterdir():
        if entry.is_file():
            try:
                entry.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def _to_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
