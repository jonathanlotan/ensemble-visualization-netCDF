"""EnsembleFile - the domain model over one IMS ICON ensemble file.

Wraps the strided memmap from nc3 with coordinates, valid times, member labels and the
ensemble aggregations the UI needs. Every accessor is a numpy slice: see CLAUDE.md 0.4.
"""
import datetime as dt
import json
import numpy as np
from pathlib import Path

from . import nc3, transform
from .transform import AGGREGATIONS      # noqa: F401  (re-exported: v1 import site)


class EnsembleFile:
    """One `ICON_ENS_<run>_<FIELD>.nc` file, read lazily."""

    def __init__(self, path):
        self.path = Path(path)
        self.hdr = nc3.parse(self.path)
        self.field_var = nc3.field_name(self.hdr)
        self._data = nc3.view(self.path, self.hdr, self.field_var)
        self.n_times, self.n_members, self.ny, self.nx = self._data.shape

        self.lat = np.asarray(nc3.view(self.path, self.hdr, 'lat'), dtype=float)
        self.lon = np.asarray(nc3.view(self.path, self.hdr, 'lon'), dtype=float)
        self.dlat = float(np.diff(self.lat).mean()) if self.ny > 1 else 0.025
        self.dlon = float(np.diff(self.lon).mean()) if self.nx > 1 else 0.025

        attrs = self.hdr['vars'][self.field_var]['attrs']
        self.units = str(attrs.get('units', '')).strip()
        self.long_name = str(attrs.get('long_name', self.field_var)).strip()
        self.field = self.field_var[:-4] if self.field_var.endswith('_eps') else self.field_var

        raw_time = np.asarray(nc3.view(self.path, self.hdr, 'time'), dtype=float)
        epoch, scale = nc3.parse_time_units(self.hdr['vars']['time']['attrs'].get('units'))
        self.run_init = epoch
        self.times = [epoch + dt.timedelta(seconds=float(v) * scale) for v in raw_time]
        self.forecast_hours = np.array(
            [(t - epoch).total_seconds() / 3600.0 for t in self.times])

        self.member_labels = nc3.member_labels(self.hdr['attrs'].get('history'),
                                               self.n_members)
        # G26: the file may hold fewer records than its header declares (interrupted
        # download). It opens and reads correctly, but the user has to be told they are
        # looking at part of a forecast.
        self.truncated = bool(self.hdr.get('truncated'))
        self.declared_times = int(self.hdr.get('declared_numrecs', self.n_times))
        self._ranges = {}

    # ---- reads -----------------------------------------------------------------
    def frame(self, t, member):
        """One member's map at time index t -> (ny, nx) float32."""
        return np.asarray(self._data[t, member], dtype=np.float32)

    def ens_frame(self, t):
        """All members' maps at time index t -> (n_members, ny, nx) float32."""
        return np.asarray(self._data[t], dtype=np.float32)

    def agg_frame(self, t, mode):
        """Map of an across-member aggregation at time index t."""
        return transform.aggregate(self.ens_frame(t), mode)

    def series(self, iy, ix):
        """All members through time at one grid point -> (n_times, n_members) float32."""
        return np.asarray(self._data[:, :, iy, ix], dtype=np.float32)

    def sub_frame(self, t, rows, cols):
        """All members at time t, on a subsample of the grid.

        -> (n_members, len(rows), len(cols)) float32. Reads only the cells asked for
        instead of copying the whole frame, which is what keeps the wind barbs cheap: they
        need a few hundred of the 42,000 grid points, and at 20 members the difference
        measures 0.16 ms against 10 ms -- a 60 fps scrub against a stutter.
        """
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        return np.asarray(self._data[t][:, rows][:, :, cols], dtype=np.float32)

    # ---- coordinates -----------------------------------------------------------
    def nearest_index(self, lat, lon):
        """Grid indices of the cell nearest a lat/lon, clamped to the domain (F3.3)."""
        iy = int(np.clip(round((lat - self.lat[0]) / self.dlat), 0, self.ny - 1))
        ix = int(np.clip(round((lon - self.lon[0]) / self.dlon), 0, self.nx - 1))
        return iy, ix

    def extent(self):
        """(lon_min, lon_max, lat_min, lat_max) of cell edges, for view limits."""
        return (self.lon[0] - self.dlon / 2, self.lon[-1] + self.dlon / 2,
                self.lat[0] - self.dlat / 2, self.lat[-1] + self.dlat / 2)

    def label_for(self, t):
        """'2026-08-27 14:00Z  (+110 h)' - the F4 time readout."""
        return (f'{self.times[t]:%Y-%m-%d %H:%M}Z  '
                f'(+{self.forecast_hours[t]:.0f} h)')

    # ---- global range, cached --------------------------------------------------
    @property
    def stats_path(self):
        return self.path.with_suffix(self.path.suffix + '.imsstats.json')

    def _cache_key(self):
        st = self.path.stat()
        return {'size': st.st_size, 'mtime': int(st.st_mtime), 'var': self.field_var}

    def _load_ranges(self):
        """Sidecar contents, or {} when absent or stale.

        Schema v2 (G19) keys ranges by transform signature, because a de-accumulated view
        has its own range and cannot reuse the raw one. A v1 blob -- a bare {min, max} --
        is read as the raw range, so no existing cache is thrown away.
        """
        try:
            blob = json.loads(self.stats_path.read_text())
        except (OSError, ValueError):
            return {}
        if blob.get('key') != self._cache_key():
            return {}
        ranges = blob.get('ranges')
        if ranges is None:
            if 'min' in blob and 'max' in blob:            # v1
                return {'raw': {'min': blob['min'], 'max': blob['max']}}
            return {}
        return ranges if isinstance(ranges, dict) else {}

    def cached_range(self, signature='raw'):
        """Cached (min, max) for one transform view, or None if absent/stale."""
        entry = self._load_ranges().get(signature)
        if not isinstance(entry, dict) or 'min' not in entry or 'max' not in entry:
            return None
        self._ranges[signature] = (entry['min'], entry['max'])
        return self._ranges[signature]

    def _store_range(self, signature, lo, hi):
        """Merge one signature into the sidecar, leaving the others intact."""
        ranges = self._load_ranges()
        ranges[signature] = {'min': lo, 'max': hi}
        try:
            self.stats_path.write_text(json.dumps(
                {'schema': 2, 'key': self._cache_key(), 'ranges': ranges}))
        except OSError:
            pass                      # read-only location: cache is an optimisation only

    def scan_range(self, progress=None, cancel=None, signature='raw', frame_source=None):
        """One full pass for the global min/max (~0.6 s for 407 MB). Caches the result.

        `frame_source` lets a transformed view be scanned in its own values while the
        result is still filed against this file's sidecar.
        """
        source = frame_source if frame_source is not None else self.ens_frame
        lo, hi = np.inf, -np.inf
        for t in range(self.n_times):
            if cancel is not None and cancel():
                return None
            block = source(t)
            if np.isfinite(block).any():
                lo = min(lo, float(np.nanmin(block)))
                hi = max(hi, float(np.nanmax(block)))
            if progress is not None:
                progress(t + 1, self.n_times)
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._ranges[signature] = (lo, hi)
        self._store_range(signature, lo, hi)
        return self._ranges[signature]

    def range_for(self, signature='raw'):
        """In-memory range for one transform view (None until cached or scanned)."""
        return self._ranges.get(signature)

    @property
    def value_range(self):
        return self._ranges.get('raw')

    @property
    def truncation_note(self):
        if not self.truncated:
            return None
        return (f'{self.path.name} holds only {self.n_times} of the '
                f'{self.declared_times} forecast steps its header declares -- the file '
                'looks incomplete (interrupted download?). Everything shown is correct, '
                'but it stops early.')

    def summary(self):
        short = (f' | INCOMPLETE: {self.n_times}/{self.declared_times} steps'
                 if self.truncated else '')
        return (f'{self.path.name} | {self.field} ({self.long_name}) [{self.units}] | '
                f'run {self.run_init:%Y-%m-%d %H:%M}Z | {self.n_members} members | '
                f'{self.n_times} steps{short} | {self.ny}x{self.nx} grid')


def member_stats(values):
    """The six F4 numbers for one time step's member values."""
    values = np.asarray(values, dtype=np.float64)
    good = values[np.isfinite(values)]
    if good.size == 0:
        return dict(mean=np.nan, max=np.nan, min=np.nan, p90=np.nan, p10=np.nan, n=0)
    p10, p90 = np.percentile(good, [10, 90])
    return dict(mean=float(good.mean()), max=float(good.max()), min=float(good.min()),
                p90=float(p90), p10=float(p10), n=int(good.size))
