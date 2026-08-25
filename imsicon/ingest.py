"""Getting from a downloaded `.nc.bz2` to a memmap-able `.nc` on disk.

262 MB compressed -> 407 MB uncompressed, ~16 s (CLAUDE.md 0.3), so this runs off the UI
thread with progress and a cancel hook.
"""
import bz2
import re
import os
import time
import shutil
import sys
from pathlib import Path

CHUNK = 1 << 22                 # 4 MiB
MIN_FREE_BYTES = 1 << 30        # refuse to decompress with under 1 GB free
DEFAULT_CACHE_BUDGET = 4 << 30  # LRU-evict above 4 GB (one full run is ~6 GB)
# Downloads are the compressed originals (~262 MB each), kept so a re-open never refetches.
DEFAULT_DOWNLOAD_BUDGET = 4 << 30


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


def download_dir():
    """Where the built-in downloader puts the compressed originals it fetched.

    A sibling of the decompressed cache, not the same directory: the two hold different
    things, expire on their own budgets, and mixing them would make an eviction glob
    delete whichever the user actually wanted to keep.
    """
    path = cache_dir().parent / 'downloads'
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_compressed(path):
    return str(path).lower().endswith('.bz2')


def expected_target(path):
    return cache_dir() / Path(path).name[:-4]      # drop '.bz2'


def _evict_glob(root, pattern, budget):
    """Drop least-recently-used files matching `pattern` until `root` fits `budget`."""
    files = sorted(root.glob(pattern), key=lambda f: f.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if total <= budget:
            break
        size = f.stat().st_size
        try:
            f.unlink()
            total -= size
            # The stats sidecar is worthless once its .nc is gone, and it is keyed on that
            # file's size+mtime, so it can never be reused by anything else.
            f.with_suffix(f.suffix + '.imsstats.json').unlink(missing_ok=True)
        except OSError:
            pass


def evict(budget=DEFAULT_CACHE_BUDGET):
    """Drop least-recently-used cached .nc files until the cache fits the budget."""
    _evict_glob(cache_dir(), '*.nc', budget)


def evict_downloads(budget=DEFAULT_DOWNLOAD_BUDGET):
    """The same, for the compressed originals the downloader fetched.

    `.part` files are deliberately spared: an interrupted transfer is meant to resume.
    """
    _evict_glob(download_dir(), '*.nc.bz2', budget)


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


# ---- finding the other fields of a run --------------------------------------------------
# Same strictness as the server listing (download.NAME_RE): a file is only treated as an
# ensemble product if its name says exactly what run and field it is.
RUN_FILE_RE = re.compile(r'^ICON_ENS_(?P<run>\d{10})_(?P<field>[A-Z0-9_]+)\.nc(?P<bz2>\.bz2)?$')


def search_roots(near=None):
    """Where to look for the other fields of a run: beside the open file, then the caches."""
    roots = []
    if near is not None:
        near = Path(near)
        roots.append(near if near.is_dir() else near.parent)
    roots += [Path.cwd() / 'data', Path.cwd()]
    try:
        roots += [cache_dir(), download_dir()]
    except OSError:
        pass
    seen, unique = set(), []
    for root in roots:
        key = str(root)
        if key not in seen and root.is_dir():
            seen.add(key)
            unique.append(root)
    return unique


def scan_for_fields(roots):
    """-> {(run, field): Path} over `roots`, in priority order.

    An already-decompressed `.nc` always wins over the `.nc.bz2` it came from -- opening
    the second field of a pair should not spend 16 s re-expanding a file that is sitting
    in the cache. Earlier roots win over later ones, so the directory the user is actually
    working in beats a stale copy in the cache.
    """
    found = {}
    for root in roots:
        for path in sorted(root.iterdir() if root.is_dir() else []):
            match = RUN_FILE_RE.match(path.name)
            if not match or not path.is_file():
                continue
            key = (match.group('run'), match.group('field'))
            previous = found.get(key)
            if previous is None or (is_compressed(previous) and not match.group('bz2')):
                found[key] = path
    return found
