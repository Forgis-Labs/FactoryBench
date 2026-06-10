from factorybench.cache import clear_judge_cache, judge_cache_stats


def test_stats_nonexistent_path(tmp_path):
    stats = judge_cache_stats(tmp_path / "no-such-dir")
    assert stats.exists is False
    assert stats.file_count == 0
    assert stats.total_bytes == 0


def test_stats_empty_dir(tmp_path):
    stats = judge_cache_stats(tmp_path)
    assert stats.exists is True
    assert stats.file_count == 0
    assert stats.oldest_iso is None
    assert stats.newest_iso is None


def test_stats_after_writes(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text('{"hello": "world"}')
    stats = judge_cache_stats(tmp_path)
    assert stats.file_count == 2
    assert stats.total_bytes > 0
    assert stats.oldest_iso is not None
    assert stats.newest_iso is not None


def test_clear_removes_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    removed = clear_judge_cache(tmp_path)
    assert removed == 2
    assert judge_cache_stats(tmp_path).file_count == 0


def test_clear_nonexistent_returns_zero(tmp_path):
    assert clear_judge_cache(tmp_path / "no-such") == 0


def test_stats_skips_subdirs(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "subdir").mkdir()
    stats = judge_cache_stats(tmp_path)
    assert stats.file_count == 1


def test_to_dict_shape(tmp_path):
    stats = judge_cache_stats(tmp_path)
    d = stats.to_dict()
    assert set(d.keys()) >= {"path", "exists", "file_count", "total_bytes", "total_mb"}
