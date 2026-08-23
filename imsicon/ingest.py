"""Getting from a downloaded `.nc.bz2` to a memmap-able `.nc` on disk.

262 MB compressed -> 407 MB uncompressed, ~16 s (CLAUDE.md 0.3), so this runs off the UI
thread with progress and a cancel hook.
"""
import bz2
import os
import time
import shutil
import sys
from pathlib import Path

CHUNK = 1 << 22                 # 4 MiB
MIN_FREE_BYTES = 1 << 30        # refuse to decompress with under 1 GB free
DEFAULT_CACHE_BUDGET = 4 << 30  # LRU-evict above 4 GB (one full run is ~6 GB)


class DiskFull(Exception):
    pass


def cache_dir():
    """Per-user cache: %LOCALAPPDATA% on Windows, ~/Library/Caches on macOS, XDG on Linux."""
    if sys.platform.startswith('win'):
        root = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    elif sys.platform == 'darwin':
        root = Path.home() / 'Library' / 'Caches'
    else:
        root = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    path = root / 'IMSIconViewer' / 'cache'
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_compressed(path):
    return str(path).lower().endswith('.bz2')


def expected_target(path):
    return cache_dir() / Path(path).name[:-4]      # drop '.bz2'


def evict(budget=DEFAULT_CACHE_BUDGET):
    """Drop least-recently-used cached .nc files until the cache fits the budget."""
    files = sorted(cache_dir().glob('*.nc'), key=lambda f: f.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if total <= budget:
            break
        size = f.stat().st_size
        try:
            f.unlink()
            total -= size
        except OSError:
            pass


def decompress(path, progress=None, cancel=None):
    """`.nc.bz2` -> cached `.nc` path. Reuses a complete cached copy. Cancellable."""
    path = Path(path)
    target = expected_target(path)
    if target.exists() and target.stat().st_size > 0:
        if cancel is None or not cancel():
            # Touch atime only: eviction sorts on atime, while mtime is what the stats
            # sidecar is keyed on, so rewriting it would discard the cached range.
            os.utime(target, ns=(time.time_ns(), target.stat().st_mtime_ns))
            return target
    if shutil.disk_usage(target.parent).free < MIN_FREE_BYTES:
        raise DiskFull(f'less than {MIN_FREE_BYTES >> 30} GB free in {target.parent}')

    compressed = path.stat().st_size
    partial = target.with_suffix(target.suffix + '.part')
    written = 0
    try:
        with bz2.open(path, 'rb') as src, open(partial, 'wb') as dst:
            while True:
                if cancel is not None and cancel():
                    raise KeyboardInterrupt
                block = src.read(CHUNK)
                if not block:
                    break
                dst.write(block)
                written += len(block)
                if progress is not None:
                    # ~1.55x expansion measured on real files; only drives the bar
                    progress(written, int(compressed * 1.55))
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    evict()
    return target


def resolve(path, progress=None, cancel=None):
    """Accept either `.nc` or `.nc.bz2` and return a path that nc3 can memmap."""
    path = Path(path)
    return decompress(path, progress, cancel) if is_compressed(path) else path
