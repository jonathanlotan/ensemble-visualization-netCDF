"""Fields computed from more than one file: dew point, difference, and the wind map.

All three are the same shape of problem -- take two ensemble files of one run and combine
them member by member -- so they share one pairing check and one view base class.

    T_2M  ─┐
           ├─► DewPointView ──┐
 RELHUM_2M─┘                  ├─► DifferenceView  (T − Td, the dew point depression)
    T_2M  ─────────────────---┘

    U_10M ─┐
           ├─► WindView       (speed as the map, u and v kept for the barbs)
    V_10M ─┘

The views mirror `FieldView`'s attribute surface, so `MainWindow`, `MapView`, `PlotView`
and `ReadoutPanel` need no special case: a derived field is just another dataset.

**G17 -- member correspondence across two files must be verified, not assumed.** Per G1,
member identity is positional and survives only in the `history` attribute. If T and RH
were merged in different orders, combining `t[i]` with `rh[i]` mixes two *different*
members and the result looks entirely plausible. `check_pairable` compares the member
labels element for element and refuses to pair on any mismatch -- the same reason v2 gave
for wind speed, now enforced rather than documented.
"""
from typing import NamedTuple

import numpy as np

from . import barbs, isolines, products, transform
from .dataset import EnsembleFile
from .fieldview import FieldView

# Alduchov & Eskridge (1996), the modern refit of Magnus-Tetens. Stated accuracy is better
# than 0.1 degC over -40..+50 degC and 1..100 % RH, which covers every value this model
# produces over Israel. The coefficients are for degC, so the conversion in and out of
# Kelvin is part of the formula, not decoration.
MAGNUS_A = 17.625
MAGNUS_B = 243.04            # degC
ZERO_C = 273.15

# Fields the dew point may be built from, and the units each must be in. The units check is
# a guard, not a conversion: a RELHUM_2M that came back as a 0-1 fraction would give a dew
# point ~45 degC too low with no other symptom, so an unexpected units string refuses
# rather than guesses (the same policy as the v2 registry, v2.md 1.3).
DEW_POINT_INPUTS = {'temperature': ('T_2M', 'K'), 'humidity': ('RELHUM_2M', '%')}
# Both families spell the roles differently -- the ensemble publishes `T_2M`/`RELHUM_2M`,
# the deterministic run `t_2m`/`rh_2m` -- so a role is checked against every family's name
# for it rather than against one hard-coded field. The units expected for the role are the
# same either way: the formula is written for K and %, whatever the file is called.
ROLE_UNITS = {'temperature': 'K', 'humidity': '%', 'zonal': 'm s-1', 'meridional': 'm s-1'}


# A wind map can be built from the 10 m components or from the 3-D ones on pressure
# levels; both play the same role in the formula, so both are accepted for it.
_ROLE_ALIASES = {'zonal': ('zonal', 'zonal_upper'),
                 'meridional': ('meridional', 'meridional_upper')}


def role_fields(role):
    """Every family's name(s) for a role, e.g. ('T_2M', 't_2m') for 'temperature'."""
    names = _ROLE_ALIASES.get(role, (role,))
    return tuple(dict.fromkeys(
        family.roles[name] for family in products.FAMILIES for name in names
        if name in family.roles))


def role_field(family, role):
    """This family's name for a role, or None when it has no field for it."""
    return family.roles.get(role) if family is not None else None


# The wind pairs a family can offer, most-surface first: the 10 m components, and (for the
# deterministic run) the 3-D ones, which give the wind AT a pressure level.
WIND_PAIRS = (('zonal', 'meridional'), ('zonal_upper', 'meridional_upper'))


def wind_pairs_in(family, present):
    """-> [(zonal, meridional)] the family defines and `present` actually holds."""
    pairs = []
    for roles in WIND_PAIRS:
        fields = tuple(family.roles.get(role) for role in roles)
        if all(fields) and all(field in present for field in fields):
            pairs.append(fields)
    return pairs


def wind_pair_for(family, present, prefer=None, levels=None):
    """The wind pair to build from, out of what this run has on disk.

    `prefer` is a field the pair should contain -- opening `u` (on pressure levels) and
    asking for a wind map means the wind on those levels, not the 10 m wind that happens
    to be in the same run.

    `levels` picks by SHAPE instead, which is what barbs drawn over another field need:
    over an 850 hPa temperature the barbs must be the wind at 850 hPa (the 3-D pair), and
    over a surface or ensemble map they must be the 10 m pair. The catalogue answers that
    without opening anything; `check_pairable` is still the real guard.
    """
    pairs = wind_pairs_in(family, present)
    for fields in pairs:
        if prefer is not None and prefer in fields:
            return fields
    if levels is not None:
        for fields in pairs:
            if all(family.has_levels(field) for field in fields) == bool(levels):
                return fields
        return None
    return pairs[0] if pairs else None
DEW_POINT_FIELD = 'TD_2M'
# What the dew point depression is called on screen. `T_2M-TD_2M` is the machine name;
# `T-Td` is what a forecaster reads it as, and the map is for reading.
DEPRESSION_NAME = 'T-Td'
# The two operands, in order, that MAKE a difference the depression -- whichever way it
# was built: from the dialog, from `--derive depression`, or as an ad-hoc `--difference
# T_2M TD_2M` against a TD_2M file written earlier by "Save field...".
# Compared through `products.field_key`, so the deterministic run's NATIVE pair
# (`t_2m` and its published `td_2m`) is recognised as the same quantity as the ensemble's
# computed one -- same name on the map, same 0.5 degC isolines, same sort band.
DEPRESSION_OPERANDS = (products.field_key(DEW_POINT_INPUTS['temperature'][0]),
                       products.field_key(DEW_POINT_FIELD))


# ---- the "sort" scale (R5) --------------------------------------------------------------
class SortScale(NamedTuple):
    """A colour scale that only colours a band, and leaves everything above it plain.

    "Sort" is the requested word for it, and it describes what it does to the map: the
    depression is sorted into the range that decides whether there is fog or cloud at the
    surface, and everything drier than that stops competing for attention. Values above
    the top stop are not a separate colour -- they ARE the top stop, which is white, so
    the colours simply run out where the band does.

    `stops` are `(value, '#rrggbb', name)` in DISPLAY units, ascending. The name is not
    decoration: "0 is red, 1 is yellow-orange, 2 is white" is how the scale was specified
    and how the tooltip has to describe it, so it is kept beside the hex rather than
    written out a second time in the UI. Only data here -- the pyqtgraph colormap is
    built in `ui/main.py`, because `derived.py` stays free of Qt.
    """
    stops: tuple

    @property
    def levels(self):
        """(low, high) for the colorbar -- the band, not the data range."""
        return (float(self.stops[0][0]), float(self.stops[-1][0]))

    @property
    def top(self):
        return float(self.stops[-1][0])

    def positions(self):
        """-> ([0..1] positions, [colours]) -- the two arrays a colormap is built from."""
        low, high = self.levels
        span = (high - low) or 1.0
        return ([(float(stop[0]) - low) / span for stop in self.stops],
                [stop[1] for stop in self.stops])

    def described(self):
        """-> [(value, name)] -- the scale in words, for a tooltip or a title."""
        return [(float(stop[0]), stop[2]) for stop in self.stops]


# Red at 0, yellow-orange at 1, white at 2 -- stated in degC, the canonical space of a
# temperature difference, and rescaled by the units affine like any other spacing (G15),
# so the band is 2 degC whether the toolbar reads degC, K or degF. The colours are
# ColorBrewer RdYlBu's warm end, which stays legible in print and to the red-green
# colour blind (the ramp varies in lightness, not only in hue).
SORT_STOPS = ((0.0, '#d7191c', 'red'), (1.0, '#fdae61', 'yellow-orange'),
              (2.0, '#ffffff', 'white'))

# The wind map: the two components it is built from, and the units each must be in. Same
# policy as the dew point -- the units string is a guard, never a conversion, because a
# component that arrived in km h-1 would draw barbs at 3.6x the real speed and look fine.
WIND_INPUTS = {'zonal': ('U_10M', 'm s-1'), 'meridional': ('V_10M', 'm s-1')}
WIND_FIELD = 'WSPD_10M'
WIND_NAME = 'wind 10m'
# The same view built from the deterministic run's 3-D components is the wind AT A LEVEL,
# not at 10 m, and saying "10m" over an 850 hPa map would be a plain misstatement. It gets
# its own name and its own registry entry (`WSPD`), which is also what a saved file of it
# reopens as.
UPPER_WIND_FIELD = 'WSPD'
UPPER_WIND_NAME = 'wind'


class PairError(Exception):
    """Two files cannot be combined. The message says which check failed and why."""


# ---- the physics -----------------------------------------------------------------------
def dew_point_celsius(t_celsius, rh_percent, dtype=np.float32):
    """Dew point in degC from temperature in degC and relative humidity in %.

        gamma = ln(RH/100) + a*T/(b+T)
        Td    = b*gamma / (a - gamma)

    Three edge cases decide whether this is usable on real model output:

    * **RH > 100 %** happens as float noise around saturation. Left alone it produces
      Td > T, which is thermodynamically impossible. RH is clamped to 100, where the
      formula collapses exactly to Td = T.
    * **RH <= 0 %** has no dew point at all -- ln(0) is -inf -- so the result is NaN
      rather than a very large negative number that would still colour a map.
    * **T <= -243.04 degC** is the formula's pole. Not reachable in this product, masked
      anyway, because a pole that is "unreachable" is exactly what shows up in a file one
      day.

    `Td <= T` is an identity of the formula and is enforced at the end, so rounding cannot
    produce a supersaturated pixel.

    Written with in-place numpy so that a 20-member map frame (261x161x20 = 840k values)
    stays inside the 60 fps scrub budget v2 set: the obvious spelling, with `np.where` for
    each guard, measured 13.2 ms of a 16.7 ms frame, of which 8.2 ms was the `np.where`
    calls alone and only 0.5 ms was the logarithm. Every step below therefore writes into
    a buffer it already owns, and the guards are applied once at the end.
    """
    t = np.asarray(t_celsius, dtype=dtype)
    rh = np.asarray(rh_percent, dtype=dtype)
    tiny = np.finfo(dtype).tiny
    # Two buffers, allocated once and written through for the whole formula. `np.empty`
    # rather than letting each operator allocate: at 20x261x161 every temporary is 3.4 MB.
    # It also sidesteps numpy handing back an immutable scalar for a 0-d input, which no
    # in-place step would accept.
    shape = np.broadcast(t, rh).shape
    gamma = np.empty(shape, dtype=dtype)
    scratch = np.empty(shape, dtype=dtype)

    # gamma = ln(RH/100). The clip pins RH into (0, 100]: the top is the supersaturation
    # clamp, the bottom only keeps the logarithm finite -- those cells are masked out at
    # the end anyway.
    np.clip(rh, tiny, dtype(100.0), out=gamma)
    np.log(gamma, out=gamma)
    gamma -= dtype(np.log(100.0))              # ln(rh) - ln(100) == ln(rh/100)

    # gamma += a*T/(b+T)
    np.add(t, dtype(MAGNUS_B), out=scratch)
    with np.errstate(invalid='ignore', divide='ignore'):
        np.divide(t, scratch, out=scratch)
    scratch *= dtype(MAGNUS_A)
    gamma += scratch

    # Td = b*gamma / (a - gamma)
    np.subtract(dtype(MAGNUS_A), gamma, out=scratch)
    gamma *= dtype(MAGNUS_B)
    with np.errstate(invalid='ignore', divide='ignore'):
        np.divide(gamma, scratch, out=gamma)

    np.minimum(gamma, t, out=gamma)            # Td <= T, always
    unusable = ~(np.isfinite(gamma) & (rh > 0) & (t > -MAGNUS_B))
    np.copyto(gamma, np.nan, where=unusable)   # copyto, not gamma[mask]: 0-d safe
    return gamma


def dew_point_kelvin(t_kelvin, rh_percent, dtype=np.float32):
    """The same, in the Kelvin the IMS files (and this app's registry) speak."""
    t = np.asarray(t_kelvin, dtype=dtype)
    return dew_point_celsius(t - ZERO_C, rh_percent, dtype=dtype) + dtype(ZERO_C)


def humidity_oddities(rh_percent):
    """-> (n_above_100, n_non_positive). Reported once, not counted on every redraw."""
    rh = np.asarray(rh_percent)
    finite = np.isfinite(rh)
    # 1e-3 of tolerance: values sitting exactly on saturation are normal, not an oddity.
    return (int(np.count_nonzero(finite & (rh > 100.001))),
            int(np.count_nonzero(finite & (rh <= 0.0))))


# ---- pairing ---------------------------------------------------------------------------
def _describe(view):
    return f'{view.field} ({getattr(view.path, "name", view.path)})'


def check_pairable(a, b):
    """Raise PairError unless two views are the same forecast on the same grid (G17).

    Checked in the order that produces the most useful message: member order first,
    because it is the failure that otherwise goes unnoticed.
    """
    axis_a, axis_b = a.axis, b.axis
    if axis_a.kind != axis_b.kind:
        raise PairError(
            f'{_describe(a)} is on {axis_a.describe()} and {_describe(b)} is on '
            f'{axis_b.describe()}. Those are different axes -- an ensemble member is not '
            'a pressure level -- so combining them element by element would pair values '
            'that have nothing to do with each other.')
    if a.n_members != b.n_members:
        raise PairError(f'{_describe(a)} has {axis_a.describe()} and {_describe(b)} has '
                        f'{axis_b.describe()}. They are not the same {axis_a.noun} axis.')
    labels_a, labels_b = list(a.member_labels), list(b.member_labels)
    # One plane has no position to mix up, so its label is a NAME (`2 m`, `10 m`,
    # `surface`), not an identity -- and requiring the two to match would refuse a 10 m
    # wind over a surface precipitation map, or `t_2m - t_g`, for no reason. G17's danger
    # is pairing position i of one file with a different position i of another, which
    # needs there to be more than one.
    if axis_a.n > 1 and labels_a != labels_b:
        first = next(i for i, (x, y) in enumerate(zip(labels_a, labels_b)) if x != y)
        mixes = ('two different ensemble members' if axis_a.aggregatable
                 else 'two different pressure levels')
        raise PairError(
            f'{axis_a.noun.capitalize()} order differs: position {first} is '
            f'{labels_a[first]!r} in {_describe(a)} but {labels_b[first]!r} in '
            f'{_describe(b)}. Combining them would mix {mixes} at every grid point, and '
            'the result would look completely plausible (G17), so these files are '
            'refused.')
    if (a.ny, a.nx) != (b.ny, b.nx):
        raise PairError(f'{_describe(a)} is {a.ny}x{a.nx} and {_describe(b)} is '
                        f'{b.ny}x{b.nx}. Different grids cannot be combined.')
    if not (np.allclose(a.lat, b.lat) and np.allclose(a.lon, b.lon)):
        raise PairError(f'{_describe(a)} and {_describe(b)} are the same size but sit on '
                        'different coordinates.')
    if a.run_init != b.run_init:
        raise PairError(f'{_describe(a)} is the {a.run_init:%Y-%m-%d %H:%M}Z run and '
                        f'{_describe(b)} is the {b.run_init:%Y-%m-%d %H:%M}Z run. '
                        'Combining two runs needs alignment on valid time, which this '
                        'build does not do.')
    if a.n_times != b.n_times or not np.allclose(a.forecast_hours, b.forecast_hours):
        incomplete = [_describe(v) for v in (a, b) if getattr(v, 'truncated', False)]
        extra = (f' {" and ".join(incomplete)} stops early -- an interrupted download '
                 'would do that.' if incomplete else '')
        raise PairError(f'{_describe(a)} has {a.n_times} forecast steps and '
                        f'{_describe(b)} has {b.n_times}.{extra}')


def units_look_compatible(field_a, field_b):
    """Cheap check that two FIELD NAMES could be subtracted, from the registry alone.

    `DifferenceView` settles it properly once the files are open, but that costs a
    decompress of up to 262 MB each; this lets a dialog grey out `RELHUM_2M - T_2M`
    before the user waits half a minute to be told no. An unknown field is allowed
    through, because only the real check is entitled to refuse.
    """
    entry_a = transform.UNITS.get(products.field_key(field_a))
    entry_b = transform.UNITS.get(products.field_key(field_b))
    if entry_a is None or entry_b is None:
        return True
    return bool(set(entry_a.expected) & set(entry_b.expected))


def open_field(path, units_label=None):
    """`path` -> FieldView, the same object `MainWindow` builds for a single file."""
    return FieldView(EnsembleFile(path), units_label=units_label)


# ---- the view base ---------------------------------------------------------------------
class DerivedView:
    """Mirrors `FieldView`'s surface, producing values computed from two operands.

    Geometry, time axis and member labels come from the first operand by delegation --
    `check_pairable` has already established the second agrees about all of them.
    """

    is_difference_view = False       # G15: True when the values are a CHANGE in a quantity
    diverging = False                # a map that wants a symmetric scale about 0
    derived = True
    can_rate = False                 # a derived field is instantaneous
    accum_kind = None
    rate_hours = 0
    rate_note = None
    window_steps = 0
    isoline_interval = None          # R5: an `isolines.Interval` in CANONICAL units
    sort_stops = None                # R5: `SortScale` stops in canonical units, or None

    def __init__(self, operands, field, long_name, provenance, display_name=None):
        self.operands = tuple(operands)
        self.ref = self.operands[0]
        self.field = field
        # What the map title and the graph's y axis call this. `field` stays the machine
        # identity -- the settings key, the written variable name -- so a friendlier label
        # never has to be parsed back into one.
        self.display_name = display_name or field
        self.long_name = long_name
        self.provenance = provenance
        self.note = None
        self._cached = None          # (lo, hi) in this view's own canonical space
        self._cached_levels = None   # the same, per position on the second axis
        self._cache_stamp = None

    def __getattr__(self, name):
        # Only reached when normal lookup fails; guarded so an access before __init__
        # finishes raises AttributeError instead of recursing forever.
        try:
            ref = object.__getattribute__(self, 'ref')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(ref, name)

    def __repr__(self):
        return f'<{type(self).__name__} {self.field} in {self.units!r}>'

    # ---- values: subclasses produce DISPLAY units -------------------------------------
    def _display_frame(self, t, member):
        raise NotImplementedError

    def _display_ens_frame(self, t):
        raise NotImplementedError

    def _display_series(self, iy, ix):
        raise NotImplementedError

    def frame(self, t, member):
        return np.asarray(self._display_frame(t, member), dtype=np.float32)

    def ens_frame(self, t):
        return np.asarray(self._display_ens_frame(t), dtype=np.float32)

    def series(self, iy, ix):
        return np.asarray(self._display_series(iy, ix), dtype=np.float32)

    def agg_frame(self, t, mode):
        return transform.aggregate(self.ens_frame(t), mode)

    # ---- units: the shape MainWindow expects -------------------------------------------
    @property
    def unit_labels(self):
        return [choice.label for choice in self.unit_choices]

    @property
    def can_convert_units(self):
        return len(self.unit_choices) > 1

    @property
    def units_note(self):
        return self.note

    def set_units(self, label):
        raise NotImplementedError

    # ---- isolines and the sort scale ---------------------------------------------------
    # Both are spacings and thresholds rather than values, so both take the affine's scale
    # and never its offset (**G15**) -- which is what makes "every 0.5 degC" and "below
    # 2 degC" mean the same thing with the Units combo on degC, K or degF.
    @property
    def isolines(self):
        """Contour interval in DISPLAY units, or None when this view is not contoured."""
        interval = self.isoline_interval
        if interval is None:
            return None
        return interval.scaled(self.units_affine, difference=self.is_difference_view)

    @property
    def sort_scale(self):
        """-> `SortScale` in DISPLAY units, or None when this view has no sort band."""
        if not self.sort_stops:
            return None
        scale = abs(float(self.units_affine.apply_delta(1.0)))
        return SortScale(tuple((value * scale, colour, name)
                               for value, colour, name in self.sort_stops))

    # ---- rate: never, for an instantaneous derived quantity ----------------------------
    @property
    def rate_choices(self):
        return [transform.rate_label(0)]

    def set_rate(self, hours):
        return not int(hours or 0)

    # ---- range: in memory, per view ----------------------------------------------------
    # No sidecar. The sidecar is keyed on ONE file's size and mtime (EnsembleFile.
    # _cache_key), which cannot identify a value that depends on two files -- a stale
    # entry would then be indistinguishable from a fresh one. A derived scan is a few
    # seconds off the UI thread, which is a fair price for not inventing a two-file key.
    @property
    def canonical_units(self):
        """Units of `canonical_ens_frame` -- what this view is written to a file in."""
        raise NotImplementedError

    def canonical_ens_frame(self, t):
        """Values in the view's own canonical space, independent of the units combo.

        Public because it is what `ncwrite.write_canonical` saves: a dew point belongs in
        a file in Kelvin whatever the screen happens to be showing, so that reopening it
        lands on the units registry's normal treatment of a temperature.
        """
        raise NotImplementedError

    def _canonical_stamp(self):
        """What the cached range was measured under; None when it cannot go stale."""
        return None

    def _transform_range(self, lo, hi):
        raise NotImplementedError

    @property
    def value_range(self):
        if self._cached is None:
            return None
        lo, hi = self._transform_range(*self._cached)
        return (lo, hi) if lo <= hi else (hi, lo)

    def level_range(self, index):
        """One level's own range, so a 500 hPa map is not coloured against 1000 hPa."""
        index = int(index)
        if not self._cached_levels or not (0 <= index < len(self._cached_levels)):
            return None
        lo, hi = self._transform_range(*self._cached_levels[index])
        return (lo, hi) if lo <= hi else (hi, lo)

    def cached_range(self):
        return self.value_range

    def scan_range(self, progress=None, cancel=None):
        lo, hi = np.inf, -np.inf
        low = np.full(self.n_members, np.inf)
        high = np.full(self.n_members, -np.inf)
        for t in range(self.n_times):
            if cancel is not None and cancel():
                return None
            block = np.asarray(self.canonical_ens_frame(t))
            if np.isfinite(block).any():
                lo = min(lo, float(np.nanmin(block)))
                hi = max(hi, float(np.nanmax(block)))
                if block.ndim == 3 and block.shape[0] == self.n_members:
                    with np.errstate(invalid='ignore'):
                        finite = np.isfinite(block).any(axis=(1, 2))
                        low = np.where(finite, np.fmin(low, np.nanmin(block, axis=(1, 2))),
                                       low)
                        high = np.where(finite, np.fmax(high, np.nanmax(block, axis=(1, 2))),
                                        high)
            if progress is not None:
                progress(t + 1, self.n_times)
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._cached = (lo, hi)
        self._cached_levels = [(float(a), float(b) if b > a else float(a) + 1.0)
                               for a, b in zip(low, high)
                               if np.isfinite(a) and np.isfinite(b)] or None
        self._cache_stamp = self._canonical_stamp()
        return self.value_range

    # ---- labels ------------------------------------------------------------------------
    @property
    def truncation_note(self):
        notes = [view.truncation_note for view in self.operands if view.truncation_note]
        return '\n\n'.join(notes) if notes else None

    def label_for(self, t):
        return self.ref.label_for(t)

    @property
    def source_files(self):
        """Every file this view's values come from, however deeply nested.

        The depression is a difference between T_2M and a *dew point*, which is itself
        built from two files -- so walking only the direct operands credits T_2M twice and
        never mentions the humidity that half the number came from.
        """
        names = []
        for operand in self.operands:
            nested = getattr(operand, 'source_files', None)
            for name in (nested if nested is not None else [operand.path.name]):
                if name not in names:
                    names.append(name)
        return names

    def summary(self):
        sources = ' | '.join(self.source_files)
        return (f'{self.display_name} ({self.long_name}) [{self.units}] | '
                f'run {self.run_init:%Y-%m-%d %H:%M}Z | {self.axis.describe()} | '
                f'{self.n_times} steps | {self.ny}x{self.nx} grid | from {sources}')


# ---- dew point -------------------------------------------------------------------------
class DewPointView(DerivedView):
    """`TD_2M` computed from `T_2M` and `RELHUM_2M`, member by member.

    Canonical space is Kelvin, matching what IMS stores for every temperature field, so
    the v2 units registry applies unchanged and the file this view writes reopens with the
    same defaults as a real one.
    """

    def __init__(self, temperature, humidity):
        check_pairable(temperature, humidity)
        _require_field_units(temperature, role_fields('temperature'),
                             ROLE_UNITS['temperature'], 'temperature')
        _require_field_units(humidity, role_fields('humidity'), ROLE_UNITS['humidity'],
                             'humidity')
        super().__init__(
            (temperature, humidity), DEW_POINT_FIELD, 'dew point temperature in 2m',
            f'dew point from {temperature.field} and {humidity.field} '
            f'({temperature.path.name}, {humidity.path.name}) by the '
            f'Alduchov-Eskridge Magnus formula')
        self.temperature, self.humidity = temperature, humidity
        # Contoured at the same 1 degC as the temperature it is read against, from the
        # same registry entry -- a dew point map and a temperature map are compared by
        # eye, and two different intervals would make that comparison a trap.
        self.isoline_interval = isolines.interval_for(DEW_POINT_FIELD)
        # TD_2M is in the registry as a Kelvin temperature, so this yields degC / K / degF
        # with degC as the default -- the same treatment T_2M gets.
        self.unit_choices, registry_note = transform.choices_for(DEW_POINT_FIELD, 'K')
        self._units = self.unit_choices[0]
        self.note = registry_note or self._sample_note()

    def set_units(self, label):
        for choice in self.unit_choices:
            if choice.label == label:
                self._units = choice
                return True
        return False

    @property
    def units_affine(self):
        return self._units

    @property
    def units(self):
        return self._units.label

    def _sample_note(self):
        """Say once whether the humidity field holds values the formula has to fix up."""
        try:
            steps = sorted({0, self.n_times // 2, self.n_times - 1})
            sample = np.concatenate([self.humidity.raw.ens_frame(t).ravel() for t in steps])
        except Exception:
            return None
        above, non_positive = humidity_oddities(sample)
        if not (above or non_positive):
            return None
        parts = []
        if above:
            parts.append(f'{above} sampled cells report RH above 100 %, clamped to 100 '
                         '(otherwise the dew point would exceed the temperature)')
        if non_positive:
            parts.append(f'{non_positive} sampled cells report RH at or below 0 %, which '
                         'has no dew point; those are left blank')
        return f'{DEW_POINT_FIELD}: ' + '; '.join(parts) + '.'

    # ---- values ------------------------------------------------------------------------
    # The operands' RAW accessors are used deliberately: the formula is defined on K and %,
    # not on whatever the user has the temperature panel switched to.
    def _kelvin_frame(self, t, member):
        return dew_point_kelvin(self.temperature.raw.frame(t, member),
                                self.humidity.raw.frame(t, member))

    def _kelvin_ens_frame(self, t):
        return dew_point_kelvin(self.temperature.raw.ens_frame(t),
                                self.humidity.raw.ens_frame(t))

    def _kelvin_series(self, iy, ix):
        return dew_point_kelvin(self.temperature.raw.series(iy, ix),
                                self.humidity.raw.series(iy, ix))

    def _display_frame(self, t, member):
        return self._units.apply(self._kelvin_frame(t, member))

    def _display_ens_frame(self, t):
        return self._units.apply(self._kelvin_ens_frame(t))

    def _display_series(self, iy, ix):
        return self._units.apply(self._kelvin_series(iy, ix))

    # ---- range: cached in Kelvin, converted affinely like any other temperature --------
    canonical_units = 'K'

    def canonical_ens_frame(self, t):
        return self._kelvin_ens_frame(t)

    def _transform_range(self, lo, hi):
        return self._units.apply_range(lo, hi)


def _require_field_units(view, field, expected, role, what='dew point'):
    """Refuse an operand whose units are not what the formula is written for.

    A guard, never a conversion: the same policy the v2 registry follows (v2.md 1.3). A
    RELHUM_2M that arrived as a 0-1 fraction would put the dew point tens of degrees out
    with nothing on screen to show it, and a U_10M in km h-1 would draw barbs at 3.6x the
    real speed -- both entirely plausible-looking.
    """
    accepted = (field,) if isinstance(field, str) else tuple(field)
    if view.field not in accepted:
        raise PairError(f'The {role} input must be {" or ".join(accepted)}, not '
                        f'{view.field} ({view.path.name}).')
    actual = transform.normalise_units(getattr(view, 'raw', view).units)
    if actual != expected:
        raise PairError(
            f'{field} ({view.path.name}) reports units {getattr(view, "raw", view).units!r}, '
            f'which normalises to {actual!r}; the {what} needs {expected!r}. '
            f'No {what} is computed -- converting on a guess would be wrong with nothing '
            'on screen to show it.')


def _by_role(views, role):
    """The view among `views` whose field plays `role` in either family, else None."""
    accepted = role_fields(role)
    return next((view for view in views if view.field in accepted), None)


def dew_point(temperature, humidity):
    """Build a `DewPointView` from a temperature and a humidity view, in either order."""
    views = [temperature, humidity]
    picked_t, picked_rh = _by_role(views, 'temperature'), _by_role(views, 'humidity')
    if picked_t is not None and picked_rh is not None:
        temperature, humidity = picked_t, picked_rh
    return DewPointView(temperature, humidity)


# ---- difference ------------------------------------------------------------------------
class DifferenceView(DerivedView):
    """`a - b`, member by member, at every point and time.

    Composed from the operands' DISPLAY values rather than their raw ones, which is what
    makes the units come out right without a second conversion table: for a shared affine
    `y = s*x + o`, `(s*a + o) - (s*b + o) = s*(a - b)`. The offset cancels on its own, so
    G15 is satisfied by construction instead of by a special case -- and the same identity
    is why a unit change only ever RESCALES the cached range by `s'/s`.

    Both operands are therefore held to the same unit selection; a label that only one of
    them offers is not offered at all.
    """

    is_difference_view = True
    diverging = True

    def __init__(self, a, b):
        check_pairable(a, b)
        super().__init__((a, b), f'{a.field}-{b.field}',
                         f'{a.long_name} minus {b.long_name}',
                         f'difference {a.field} - {b.field} '
                         f'({a.path.name}, {b.path.name})')
        self.a, self.b = a, b
        shared = [c.label for c in getattr(b, 'unit_choices', [])]
        self.unit_choices = [c for c in getattr(a, 'unit_choices', []) if c.label in shared]
        if not self.unit_choices:
            raise PairError(
                f'{a.field} is in {a.units!r} and {b.field} is in {b.units!r}. Subtracting '
                'two different quantities produces a number with no meaning, so these '
                'fields are not offered as a difference.')
        if not self.set_units(self.unit_choices[0].label):
            # Cannot happen while the choices are the intersection of the two operands',
            # but leaving `_units` unset would surface later as an AttributeError from
            # somewhere in the paint path rather than as a reason here.
            raise PairError(f'{a.field} and {b.field} would not both accept '
                            f'{self.unit_choices[0].label!r} as their display units.')
        if a.units != b.units:
            raise PairError(f'{a.field} and {b.field} could not be put in the same units '
                            f'({a.units!r} vs {b.units!r}).')
        # A difference of two contoured fields is contoured twice as finely (R5.1): the
        # depression a forecaster reads lives in the 0-5 degC band, where the whole
        # question is answered between one temperature isoline and the next.
        if all(isolines.interval_for(view.field) for view in (a, b)):
            self.isoline_interval = isolines.DIFFERENCE
        if tuple(products.field_key(v.field) for v in (a, b)) == DEPRESSION_OPERANDS:
            # Named and coloured here rather than in `dew_point_depression`, because this
            # is the same quantity however it was arrived at -- including an ad-hoc
            # `--difference T_2M TD_2M` against a saved dew point file.
            self.display_name = DEPRESSION_NAME
            self.long_name = 'dew point depression (T - Td)'
            self.sort_stops = SORT_STOPS

    # ---- units: one selection, pushed to both operands ---------------------------------
    def set_units(self, label):
        choice = next((c for c in self.unit_choices if c.label == label), None)
        if choice is None:
            return False
        if not (self.a.set_units(label) and self.b.set_units(label)):
            return False
        self._units = choice
        return True

    @property
    def units_affine(self):
        return self._units

    @property
    def units(self):
        return self.a.units

    # ---- values ------------------------------------------------------------------------
    # G23 says to difference in float64, and it is right about the case it names: in
    # `transform.window_value` the subtraction has float64 INTERMEDIATES (`A[t]*h[t]`)
    # whose bits the upcast genuinely preserves. Here there are none -- it is one
    # elementwise subtraction of two float32 arrays, whose IEEE result is already
    # correctly rounded, and `DerivedView.ens_frame` casts back to float32 regardless, so
    # a float64 intermediate cannot survive to be seen. MEASURED on a 20x261x161 frame:
    # bit-identical output (max difference 0.0 K) for 3.16 ms against 0.56 ms. The upcast
    # would buy a sixth of the scrub budget and nothing else.
    def _display_frame(self, t, member):
        return self.a.frame(t, member) - self.b.frame(t, member)

    def _display_ens_frame(self, t):
        return self.a.ens_frame(t) - self.b.ens_frame(t)

    def _display_series(self, iy, ix):
        return self.a.series(iy, ix) - self.b.series(iy, ix)

    # ---- range: measured in display units, rescaled when the unit changes ---------------
    @property
    def canonical_units(self):
        """A difference has no unit-independent space of its own: it is whatever the two
        operands are currently showing, which is why writing one saves the screen units."""
        return self.units

    def canonical_ens_frame(self, t):
        return self._display_ens_frame(t)

    def _canonical_stamp(self):
        return float(self._units.a)

    def _transform_range(self, lo, hi):
        if not self._cache_stamp:
            return lo, hi
        factor = float(self._units.a) / self._cache_stamp
        return lo * factor, hi * factor


def difference(a, b):
    """`a - b` as a view. Refuses anything `check_pairable` or the units rule rejects."""
    return DifferenceView(a, b)


def dew_point_depression(temperature, humidity):
    """T - Td, the flagship difference: how far the air is from saturation.

    Zero means fog or cloud at the surface; a large value means dry air. It is the reason
    the dew point and the difference map arrived together -- the dew point alone tells a
    forecaster the absolute moisture, and the depression tells them whether it matters.
    """
    td = dew_point(temperature, humidity)
    # The name and the sort band come from `DifferenceView` recognising its operands, so
    # this and `--difference T_2M TD_2M` cannot drift apart. Only the provenance is
    # written here: it is the one thing that knows the humidity file this came through.
    view = difference(td.temperature, td)
    view.provenance = (f'dew point depression from {td.temperature.path.name} and '
                       f'{td.humidity.path.name}')
    return view


# ---- the wind map ----------------------------------------------------------------------
class WindView(DerivedView):
    """The wind at 10 m: **speed** as the coloured map, **u and v kept for the barbs**.

    One view carries both because they are one quantity read two ways. The image, the
    graph and the six readout statistics all need a scalar, and that scalar is the speed
    `hypot(u, v)`; the direction cannot go through any of them (**G16** -- it is circular
    data, so a linear mean, min, max or percentile of degrees is simply wrong: the linear
    mean of 350 and 10 degrees is 180, the answer is 360). Direction therefore reaches the
    screen only as barbs, which are built from the *vectors* and never from degrees.

    Canonical space is `m s-1`, so a speed field written out with "Save field..." reopens
    as an ordinary `WSPD_10M` with the registry's kt / km h-1 options -- without its
    barbs, honestly, because a speed file no longer knows which way the wind was blowing.
    """

    def __init__(self, zonal, meridional):
        check_pairable(zonal, meridional)
        _require_field_units(zonal, role_fields('zonal'), ROLE_UNITS['zonal'],
                             'zonal wind', what='wind map')
        _require_field_units(meridional, role_fields('meridional'),
                             ROLE_UNITS['meridional'], 'meridional wind', what='wind map')
        super().__init__(
            (zonal, meridional), WIND_FIELD, 'wind speed and direction in 10m',
            f'wind from {zonal.field} and {meridional.field} '
            f'({zonal.path.name}, {meridional.path.name}): speed = hypot(u, v), '
            'barbs from the component vectors',
            display_name=WIND_NAME)
        self.u, self.v = zonal, meridional
        # Named for what it is: the 10 m wind, or the wind on whatever level is shown.
        if zonal.field in (products.ICON.roles.get('zonal_upper'),):
            self.field, self.display_name = UPPER_WIND_FIELD, UPPER_WIND_NAME
            self.long_name = 'wind speed and direction on pressure levels'
        self.unit_choices, registry_note = transform.choices_for(self.field, 'm s-1')
        self._units = self.unit_choices[0]
        self.note = registry_note

    # ---- units -------------------------------------------------------------------------
    def set_units(self, label):
        for choice in self.unit_choices:
            if choice.label == label:
                self._units = choice
                return True
        return False

    @property
    def units_affine(self):
        return self._units

    @property
    def units(self):
        return self._units.label

    # ---- values: the SPEED, in the display units --------------------------------------
    # The operands' raw accessors, deliberately: the components are m s-1 in the file
    # whatever the speed happens to be shown in, and the conversion belongs at the end.
    def _speed(self, u, v):
        return self._units.apply(np.hypot(u, v))

    def _display_frame(self, t, member):
        return self._speed(self.u.raw.frame(t, member), self.v.raw.frame(t, member))

    def _display_ens_frame(self, t):
        return self._speed(self.u.raw.ens_frame(t), self.v.raw.ens_frame(t))

    def _display_series(self, iy, ix):
        return self._speed(self.u.raw.series(iy, ix), self.v.raw.series(iy, ix))

    # ---- the barbs ---------------------------------------------------------------------
    # `barb_units` and `wind_vectors` are the whole protocol the UI knows about: a view
    # that has them gets barbs, and a view that does not gets none. Nothing in MapView or
    # MainWindow tests for this class by name.
    barb_units = 'kt'

    def _components(self, t, rows, cols):
        """The two raw component stacks, over the whole grid or a subsample of it."""
        if rows is None or cols is None:
            return (self.u.raw.ens_frame(t), self.v.raw.ens_frame(t))
        return (self.u.raw.sub_frame(t, rows, cols), self.v.raw.sub_frame(t, rows, cols))

    def wind_vectors(self, t, mode, member=0, rows=None, cols=None):
        """-> (u, v) in KNOTS on the grid, aggregated to match what the map is showing.

        `rows`/`cols` restrict the read to the grid points that will actually be drawn.
        Barbs are drawn at a stride that the zoom decides, so all but a few hundred of the
        42,000 points are thrown away -- reading them anyway costs 10 ms of a 16.7 ms
        frame, and reading only what is wanted costs 0.16 ms.

        Barbs are drawn in knots whatever the colour scale is set to, because the glyph
        *is* defined in knots -- a half feather means 5 kt, not "5 of whatever unit the
        toolbar says". The colour and the barb are still the same wind; only the unit the
        feathers count in is fixed.

        The aggregation matters more than it looks. Aggregating the two components
        independently is only right for the mean:

        * `member`  -- that member's own vector.
        * `mean`    -- the mean VECTOR (**G16**). Note that its length is not the mean
          speed the colours show: |mean(V)| <= mean(|V|) by Jensen, and the gap is exactly
          the ensemble's disagreement about direction. That is a feature to read, not an
          inconsistency: barbs all pointing one way under a strong colour means the
          members agree.
        * `max` / `min` / `median` -- the vector of the member the colour under it came
          from, picked per cell by speed. Taking max(u) with max(v) would invent a wind no
          member forecast.
        * `spread` -- a max-minus-min has no member and no direction of its own, so the
          barbs fall back to the mean vector and `barb_label` says so.
        """
        u, v = self._components(t, rows, cols)
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        if mode == 'member':
            index = int(np.clip(member, 0, u.shape[0] - 1))
            picked = (u[index], v[index])
        elif mode in ('max', 'min', 'median'):
            picked = self._pick_member(u, v, mode)
        else:                                    # mean, spread, and anything unforeseen
            picked = (np.nanmean(u, axis=0), np.nanmean(v, axis=0))
        return (np.asarray(picked[0], dtype=np.float32) * barbs.KT_PER_MS,
                np.asarray(picked[1], dtype=np.float32) * barbs.KT_PER_MS)

    @staticmethod
    def _pick_member(u, v, mode):
        """The per-cell member whose speed is the max / min / median of the ensemble."""
        speed = np.hypot(u, v)
        if mode == 'max':
            index = np.where(np.isfinite(speed), speed, -np.inf).argmax(axis=0)
        elif mode == 'min':
            index = np.where(np.isfinite(speed), speed, np.inf).argmin(axis=0)
        else:
            with np.errstate(invalid='ignore'):
                middle = np.nanmedian(speed, axis=0)
            # No member sits exactly on the median of an even ensemble, so take the
            # nearest one: a real member's direction beats an interpolated non-wind.
            distance = np.abs(np.where(np.isfinite(speed), speed, np.inf) - middle)
            index = distance.argmin(axis=0)
        index = index[None, ...]
        return (np.take_along_axis(u, index, axis=0)[0],
                np.take_along_axis(v, index, axis=0)[0])

    @property
    def source_label(self):
        """`U_10M/V_10M` -- which two files the feathers are counting."""
        return f'{self.u.field}/{self.v.field}'

    def barb_label(self, mode, over=None):
        """What the barbs on screen actually are -- it is not the same for every mode.

        `over` is set when the barbs are drawn over a DIFFERENT field's map, in which case
        the label names the wind they came from: the colours and the feathers are then two
        different quantities, and a map that shows both has to say so.
        """
        head = f'barbs ({self.barb_units})'
        if over is not None:
            head += f' from {self.source_label}'
        if mode == 'member':
            if not self.axis.aggregatable:
                # There is no member to name. On a column the title beside this already
                # says which level; a one-plane field names its own height, so a 10 m
                # wind drawn over a 2 m temperature says which of the two it is.
                where = (self.axis.labels[0] if self.axis.kind == 'single'
                         else 'this level')
                return f'{head}: the wind at {where}'
            return f'{head}: one member'
        if mode == 'spread':
            return f'{head}: mean vector - a spread has no direction of its own'
        if mode in ('max', 'min', 'median'):
            return f'{head}: the {mode}-speed member at each point'
        return f'{head}: ensemble mean vector'

    def direction_at(self, t, iy, ix, mode, member=0):
        """Wind direction in degrees at one point, for the status bar (**G16** applies:
        one value read off the aggregated vector, never a statistic of degrees)."""
        u, v = self.wind_vectors(t, mode, member, rows=[iy], cols=[ix])
        return float(barbs.direction_from(u[0, 0], v[0, 0]))

    # ---- range: cached in m s-1, converted affinely like any other scaled quantity -----
    canonical_units = 'm s-1'

    def canonical_ens_frame(self, t):
        return np.hypot(np.asarray(self.u.raw.ens_frame(t), dtype=np.float32),
                        np.asarray(self.v.raw.ens_frame(t), dtype=np.float32))

    def _transform_range(self, lo, hi):
        return self._units.apply_range(lo, hi)


def wind(zonal, meridional):
    """Build a `WindView` from a U and a V view, in either order and either family."""
    views = [zonal, meridional]
    picked_u, picked_v = _by_role(views, 'zonal'), _by_role(views, 'meridional')
    if picked_u is not None and picked_v is not None:
        zonal, meridional = picked_u, picked_v
    return WindView(zonal, meridional)
