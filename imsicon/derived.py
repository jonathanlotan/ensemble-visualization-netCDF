"""Fields computed from more than one file: dew point, and the difference of two fields.

Both are the same shape of problem -- take two ensemble files of one run and combine them
member by member -- so they share one pairing check and one view base class.

    T_2M  ─┐
           ├─► DewPointView ──┐
 RELHUM_2M─┘                  ├─► DifferenceView  (T − Td, the dew point depression)
    T_2M  ─────────────────---┘

The views mirror `FieldView`'s attribute surface, so `MainWindow`, `MapView`, `PlotView`
and `ReadoutPanel` need no special case: a derived field is just another dataset.

**G17 -- member correspondence across two files must be verified, not assumed.** Per G1,
member identity is positional and survives only in the `history` attribute. If T and RH
were merged in different orders, combining `t[i]` with `rh[i]` mixes two *different*
members and the result looks entirely plausible. `check_pairable` compares the member
labels element for element and refuses to pair on any mismatch -- the same reason v2 gave
for wind speed, now enforced rather than documented.
"""
import numpy as np

from . import transform
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
DEW_POINT_FIELD = 'TD_2M'
# What the dew point depression is called on screen. `T_2M-TD_2M` is the machine name;
# `T-Td` is what a forecaster reads it as, and the map is for reading.
DEPRESSION_NAME = 'T-Td'


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
    if a.n_members != b.n_members:
        raise PairError(f'{_describe(a)} has {a.n_members} members and {_describe(b)} has '
                        f'{b.n_members}. They are not the same ensemble.')
    labels_a, labels_b = list(a.member_labels), list(b.member_labels)
    if labels_a != labels_b:
        first = next(i for i, (x, y) in enumerate(zip(labels_a, labels_b)) if x != y)
        raise PairError(
            f'Member order differs: position {first} is {labels_a[first]!r} in '
            f'{_describe(a)} but {labels_b[first]!r} in {_describe(b)}. Combining them '
            'would mix two different ensemble members at every grid point, and the result '
            'would look completely plausible (G17), so these files are refused.')
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
    entry_a, entry_b = transform.UNITS.get(field_a), transform.UNITS.get(field_b)
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

    def cached_range(self):
        return self.value_range

    def scan_range(self, progress=None, cancel=None):
        lo, hi = np.inf, -np.inf
        for t in range(self.n_times):
            if cancel is not None and cancel():
                return None
            block = self.canonical_ens_frame(t)
            if np.isfinite(block).any():
                lo = min(lo, float(np.nanmin(block)))
                hi = max(hi, float(np.nanmax(block)))
            if progress is not None:
                progress(t + 1, self.n_times)
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        self._cached = (lo, hi)
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
                f'run {self.run_init:%Y-%m-%d %H:%M}Z | {self.n_members} members | '
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
        _require_field_units(temperature, *DEW_POINT_INPUTS['temperature'], 'temperature')
        _require_field_units(humidity, *DEW_POINT_INPUTS['humidity'], 'humidity')
        super().__init__(
            (temperature, humidity), DEW_POINT_FIELD, 'dew point temperature in 2m',
            f'dew point from {temperature.field} and {humidity.field} '
            f'({temperature.path.name}, {humidity.path.name}) by the '
            f'Alduchov-Eskridge Magnus formula')
        self.temperature, self.humidity = temperature, humidity
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


def _require_field_units(view, field, expected, role):
    if view.field != field:
        raise PairError(f'The {role} input must be {field}, not {view.field} '
                        f'({view.path.name}).')
    actual = transform.normalise_units(getattr(view, 'raw', view).units)
    if actual != expected:
        raise PairError(
            f'{field} ({view.path.name}) reports units {getattr(view, "raw", view).units!r}, '
            f'which normalises to {actual!r}; the dew point formula needs {expected!r}. '
            'No dew point is computed -- converting on a guess would be wrong by tens of '
            'degrees with nothing on screen to show it.')


def dew_point(temperature, humidity):
    """Build a `DewPointView` from a T_2M and a RELHUM_2M view, in either order."""
    views = [temperature, humidity]
    by_field = {view.field: view for view in views}
    if DEW_POINT_INPUTS['temperature'][0] in by_field and \
            DEW_POINT_INPUTS['humidity'][0] in by_field:
        temperature = by_field[DEW_POINT_INPUTS['temperature'][0]]
        humidity = by_field[DEW_POINT_INPUTS['humidity'][0]]
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
    view = difference(td.temperature, td)
    view.display_name = DEPRESSION_NAME
    view.long_name = 'dew point depression (T - Td)'
    view.provenance = (f'dew point depression from {td.temperature.path.name} and '
                       f'{td.humidity.path.name}')
    return view
