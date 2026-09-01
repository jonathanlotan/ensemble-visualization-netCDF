"""What the second data axis of a file IS -- ensemble members, or pressure levels.

Every IMS ICON product this app opens is `(time, <axis>, lat, lon)`, and until now the
axis was always the 20 ensemble members of **G1**. The deterministic ICON-LAM run
(`IMS_ICON_manual.pdf`) uses the same shape for a different thing: six of its fields are
3-D on **20 pressure levels**, and the rest are plain surface fields with no axis at all.

The three cases are one value object, decided from the file:

    kind='member'    20 members, positional identity from `history`      (G1)
    kind='pressure'  20 levels, 1000..150 hPa, identity from the values
    kind='single'    a surface field: one "level", and nothing to choose

The distinction is not cosmetic, and it is the reason this is a module rather than a
boolean. **An ensemble is a set of samples of one quantity, so a mean, a spread and a
percentile across it are the whole point. A column of pressure levels is not** -- the mean
of the temperature at 1000 hPa and at 150 hPa is not a temperature anyone forecasts, and a
"P90 across levels" is meaningless in exactly the way a mean of compass directions is
(**G16**). So `aggregatable` is False for a pressure axis and the UI offers one level at a
time, rather than offering statistics that would look perfectly plausible.
"""
import re
from typing import NamedTuple

import numpy as np

from .products import MANUAL_PRESSURE_LEVELS, PRESSURE_LEVELS

# The conventional low-level chart, and what a temperature or wind map on pressure levels
# is read at first. The map title always names the level, so this is a starting point
# rather than a hidden assumption.
DEFAULT_LEVEL_HPA = 850.0

MEMBER, PRESSURE, SINGLE = 'member', 'pressure', 'single'

# Dimension names that mean "pressure" in CF-ish output. `sfc` is deliberately NOT here:
# that is the ensemble's member axis, mislabelled as a level by the cdo merge (G1).
_PRESSURE_DIMS = ('plev', 'plev_2', 'pressure', 'lev', 'level', 'levels', 'isobaric',
                  'p', 'pres')
# hPa per unit. Anything else is not a pressure, and an unrecognised string must not be
# guessed at -- labelling 85000 Pa as "85000 hPa" is worse than not labelling it.
_PRESSURE_UNITS = {'pa': 0.01, 'hpa': 1.0, 'mb': 1.0, 'mbar': 1.0, 'millibar': 1.0,
                   'millibars': 1.0, 'hectopascal': 1.0, 'hectopascals': 1.0}


def pressure_scale(units):
    """hPa per unit of `units`, or None when the string is not a pressure."""
    text = re.sub(r'\s+', '', str(units or '')).lower()
    return _PRESSURE_UNITS.get(text)


def _looks_like_pressure(values):
    """-> hPa per unit, inferred from the VALUES when the header carries no units.

    Only two answers are possible and they do not overlap: this domain's levels are
    150..1000 hPa or 15000..100000 Pa, three orders of magnitude apart. Anything outside
    both windows is refused rather than scaled by whichever factor is closer.
    """
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all() or (values <= 0).any():
        return None
    low, high = float(values.min()), float(values.max())
    if 1.0 <= low and high <= 1200.0:
        return 1.0
    if 100.0 <= low and high <= 120000.0:
        return 0.01
    return None


def format_level(hpa):
    """`850 hPa`. Whole numbers stay whole -- 850.0 hPa reads as a measurement error."""
    value = float(hpa)
    return f'{value:g} hPa'


class LevelAxis(NamedTuple):
    """The second data axis of one file: what it is, and how to name a position on it."""

    kind: str
    labels: tuple
    values: tuple = ()          # hPa, ascending or descending as the FILE stores them
    note: str = None

    # ---- what it is ------------------------------------------------------------------
    @property
    def n(self):
        return len(self.labels)

    @property
    def is_pressure(self):
        return self.kind == PRESSURE

    @property
    def aggregatable(self):
        """May the app take a mean / max / spread ACROSS this axis?

        Only for an ensemble. See the module docstring: across a column of pressure
        levels these statistics are not wrong-looking, which is what makes them dangerous.
        """
        return self.kind == MEMBER

    @property
    def noun(self):
        return {MEMBER: 'member', PRESSURE: 'pressure level'}.get(self.kind, 'level')

    @property
    def selector_label(self):
        """What the toolbar calls the control that picks one position on this axis."""
        return {MEMBER: 'Member', PRESSURE: 'Level', SINGLE: 'Level'}[self.kind]

    def describe(self):
        """`22 pressure levels` / `20 members` / `2 m (single level)`."""
        if self.kind == SINGLE:
            return f'{self.labels[0]} (single level)'
        return f'{self.n} {self.noun}{"s" if self.n != 1 else ""}'

    def label(self, index):
        if not self.labels:
            return ''
        return self.labels[int(np.clip(index, 0, self.n - 1))]

    # ---- moving along it -------------------------------------------------------------
    @property
    def upward(self):
        """Positions ordered from the bottom of the atmosphere upward.

        For a pressure axis that is DESCENDING pressure, whatever order the file stores;
        for members it is file order, since one member is not above another.
        """
        if not self.is_pressure or len(self.values) != self.n:
            return tuple(range(self.n))
        return tuple(int(i) for i in np.argsort(-np.asarray(self.values, dtype=float)))

    def step(self, index, delta):
        """Move `delta` positions UP the axis -- up the atmosphere for pressure levels.

        "Up" is the physical direction, not the index direction: pressing up at 850 hPa
        gets 825 hPa (higher, thinner air) however the file happens to have ordered its
        level coordinate. Clamped at both ends rather than wrapping, so holding the key
        stops at the top of the column instead of jumping back to the ground.
        """
        if self.n <= 1:
            return 0
        order = self.upward
        position = order.index(int(np.clip(index, 0, self.n - 1)))
        return order[int(np.clip(position + int(delta), 0, self.n - 1))]

    def nearest(self, hpa):
        """Index of the level closest to a pressure. 0 for a non-pressure axis."""
        if not self.is_pressure or len(self.values) != self.n:
            return 0
        values = np.asarray(self.values, dtype=float)
        return int(np.abs(values - float(hpa)).argmin())

    @property
    def default_index(self):
        """Where to land when a file is opened."""
        return self.nearest(DEFAULT_LEVEL_HPA) if self.is_pressure else 0


def member_axis(labels):
    return LevelAxis(MEMBER, tuple(labels))


def single_axis(label='surface'):
    return LevelAxis(SINGLE, (label,))


def _single_label(coord, units):
    """`2 m` / `10 m` for a one-plane field that carries a height, else `surface`.

    Measured: the deterministic run's `t_2m` is (time, height, lat, lon) with height = 2 m
    and `u_10m` with height = 10 m. Calling a 10 m gust "surface" is not wrong enough to
    matter, and naming it what the file says is free.
    """
    METRES = ('m', 'meter', 'metre', 'meters', 'metres')
    if str(units or '').strip().lower() not in METRES:
        return 'surface'
    try:
        value = float(np.asarray(coord, dtype=float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return 'surface'
    return f'{value:g} m' if np.isfinite(value) and value > 0 else 'surface'


def pressure_axis(hpa, note=None):
    values = tuple(float(v) for v in hpa)
    return LevelAxis(PRESSURE, tuple(format_level(v) for v in values), values, note)


def axis_for(n, *, dim=None, coord=None, units=None, attrs=None, field=None,
             family=None, history=None):
    """Decide what an axis of length `n` is, from the file first and the manual last.

    The order of the tests is the order of how much they are worth trusting:

    1. `n == 1` -- a surface field. Nothing to choose, whatever the dimension is called.
    2. **The coordinate values say it.** Distinct positive values with pressure units, or
       with no units but unmistakably in a pressure range, ARE the levels. This is the
       only branch that can name a level correctly for a file this build has never seen.
    3. **The catalogue says the field is 3-D and the count matches the manual's ladder.**
       Used only when the file's own coordinate is degenerate -- the ensemble's `sfc` axis
       is 20 zeros (G1), and a deterministic file merged the same way would be too. The
       levels are then the manual's, in the manual's order, and `note` says so out loud,
       because an assumed order that happened to be reversed would label every map wrongly
       while looking entirely normal.
    4. Otherwise the axis is the ensemble's members, named from `history` (G1).
    """
    from . import nc3               # local: nc3 imports nothing from here

    n = int(n)
    if n <= 1:
        return single_axis(_single_label(coord, units) if coord is not None else 'surface')

    values = None if coord is None else np.asarray(coord, dtype=float).ravel()
    if values is not None and values.size == n and len(set(values.tolist())) == n:
        scale = pressure_scale(units)
        if scale is None and str(dim or '').lower() in _PRESSURE_DIMS:
            scale = _looks_like_pressure(values)
        if scale is None and str((attrs or {}).get('standard_name', '')).lower() \
                in ('air_pressure', 'pressure'):
            scale = _looks_like_pressure(values)
        if scale is not None:
            return pressure_axis(values * scale)

    if family is not None and field is not None and family.has_levels(field):
        for ladder, source in ((PRESSURE_LEVELS, 'the 22 measured on the server'),
                               (MANUAL_PRESSURE_LEVELS, "the 20 in the manual's table")):
            if n == len(ladder):
                return pressure_axis(ladder, note=(
                    f'{field} is a 3-D field, but its level coordinate carries no usable '
                    f'pressures, so the levels shown are {source}, in that order. The '
                    'server and the manual disagree about both the count and the order '
                    'of that ladder, so if a file stores them differently again every '
                    'level here would be labelled upside down -- check one value against '
                    'the model before relying on it.'))

    return member_axis(nc3.member_labels(history, n))
