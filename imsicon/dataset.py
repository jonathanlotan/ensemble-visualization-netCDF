"""EnsembleFile - the domain model over one IMS ICON ensemble file.

Wraps the strided memmap from nc3 with coordinates, valid times, member labels and the
ensemble aggregations the UI needs. Every accessor is a numpy slice: see CLAUDE.md 0.4.
"""
import datetime as dt
import json
import numpy as np
from pathlib import Path

from . import nc3

AGGREGATIONS = ('mean', 'max', 'min', 'spread', 'median')


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
        self._range = None

    # ---- reads -----------------------------------------------------------------
    def frame(self, t, member):
        """One member's map at time index t -> (ny, nx) float32."""
        return np.asarray(self._data[t, member], dtype=np.float32)

    def ens_frame(self, t):
        """All members' maps at time index t -> (n_members, ny, nx) float32."""
        return np.asarray(self._data[t], dtype=np.float32)

    def agg_frame(self, t, mode):
        """Map of an across-member aggregation at time index t."""
        stack = self.ens_frame(t)
        if mode == 'mean':
            return np.nanmean(stack, axis=0)
        if mode == 'max':
            return np.nanmax(stack, axis=0)
        if mode == 'min':
            return np.nanmin(stack, axis=0)
        if mode == 'median':
            return np.nanmedian(stack, axis=0)
        if mode == 'spread':
            return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
        raise ValueError(f'unknown aggregation {mode!r}')

    def series(self, iy, ix):
        """All members through time at one grid point -> (n_times, n_members) float32."""
        return np.asarray(self._data[:, :, iy, ix], dtype=np.float32)

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

    def cached_range(self):
        """Global (min, max) from the sidecar cache, or None if absent/stale."""
        try:
            blob = json.loads(self.stats_path.read_text())
        except (OSError, ValueError):
            return None
        if blob.get('key') != self._cache_key():
            return None
        self._range = (blob['min'], blob['max'])
        return self._range

    def scan_range(self, progress=None, cancel=None):
        """One full pass for the global min/max (~0.6 s for 407 MB). Caches the result."""
        lo, hi = np.inf, -np.inf
        for t in range(self.n_times):
            if cancel is not None and cancel():
                return None
            block = self.ens_frame(t)
            if np.isfinite(block).any():
                lo = min(lo, float(np.nanmin(block)))
                hi = max(hi, float(np.nanmax(block)))
            if progress is not None:
                progress(t + 1, self.n_times)
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._range = (lo, hi)
        try:
            self.stats_path.write_text(json.dumps(
                {'key': self._cache_key(), 'min': lo, 'max': hi}))
        except OSError:
            pass                      # read-only location: cache is an optimisation only
        return self._range

    @property
    def value_range(self):
        return self._range

    def summary(self):
        return (f'{self.path.name} | {self.field} ({self.long_name}) [{self.units}] | '
                f'run {self.run_init:%Y-%m-%d %H:%M}Z | {self.n_members} members | '
                f'{self.n_times} steps | {self.ny}x{self.nx} grid')


def member_stats(values):
    """The six F4 numbers for one time step's member values."""
    values = np.asarray(values, dtype=np.float64)
    good = values[np.isfinite(values)]
    if good.size == 0:
        return dict(mean=np.nan, max=np.nan, min=np.nan, p90=np.nan, p10=np.nan, n=0)
    p10, p90 = np.percentile(good, [10, 90])
    return dict(mean=float(good.mean()), max=float(good.max()), min=float(good.min()),
                p90=float(p90), p10=float(p10), n=int(good.size))
