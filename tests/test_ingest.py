"""The .bz2 -> cache path (F1.5), exercised with a tiny synthetic file."""
import bz2

import pytest

from imsicon import ingest


@pytest.fixture
def tiny_bz2(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    cache.mkdir()
    monkeypatch.setattr(ingest, 'cache_dir', lambda: cache)
    src = tmp_path / 'ICON_ENS_2026082300_TEST.nc.bz2'
    src.write_bytes(bz2.compress(b'CDF\x02' + b'payload' * 1000))
    return src, cache


def test_detects_compressed_input():
    assert ingest.is_compressed('a.nc.bz2') and not ingest.is_compressed('a.nc')


def test_plain_nc_is_passed_through(tmp_path):
    plain = tmp_path / 'x.nc'
    plain.write_bytes(b'CDF\x02')
    assert ingest.resolve(plain) == plain


def test_decompress_writes_to_cache_and_reuses_it(tiny_bz2):
    src, cache = tiny_bz2
    out = ingest.decompress(src)
    assert out.parent == cache and out.name == 'ICON_ENS_2026082300_TEST.nc'
    assert out.read_bytes() == b'CDF\x02' + b'payload' * 1000

    inode, mtime = out.stat().st_ino, out.stat().st_mtime_ns
    again = ingest.decompress(src)               # second open must not redo the work
    assert again == out and again.stat().st_ino == inode
    # mtime must survive: the stats sidecar is keyed on it (see EnsembleFile._cache_key)
    assert again.stat().st_mtime_ns == mtime
    assert again.stat().st_atime_ns >= mtime


def test_progress_is_reported(tiny_bz2):
    src, _ = tiny_bz2
    seen = []
    ingest.decompress(src, progress=lambda done, total: seen.append(done))
    assert seen and seen[-1] == len(b'CDF\x02' + b'payload' * 1000)


def test_cancel_leaves_no_partial_file(tiny_bz2):
    src, cache = tiny_bz2
    with pytest.raises(KeyboardInterrupt):
        ingest.decompress(src, cancel=lambda: True)
    assert list(cache.iterdir()) == []


def test_eviction_drops_the_least_recently_used(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    cache.mkdir()
    monkeypatch.setattr(ingest, 'cache_dir', lambda: cache)
    import os
    import time
    for i, name in enumerate(('old.nc', 'new.nc')):
        f = cache / name
        f.write_bytes(b'x' * 1000)
        os.utime(f, (time.time() - 1000 + i * 500,) * 2)
    ingest.evict(budget=1500)
    assert (cache / 'new.nc').exists() and not (cache / 'old.nc').exists()
