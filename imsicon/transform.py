"""The one place raw file values become displayed values (v2 section 0).

Four consumers read every value -- the map, the graph, the readout and the status bar.
Converting at each call site is four chances to drift, and the failure mode is the map
reading degC while the readout reads K. So conversion lives here, once, and `FieldView`
is the only thing that applies it.

Ordering rule (v2 0.3) -- the transforms do NOT commute:

    raw --> [1] time-differencing --> [2] unit conversion --> display

and step [2] must know what step [1] produced: a window *sum* of an accumulated field is a
DIFFERENCE and takes the scale only, while an instantaneous value or a window *mean* is
ABSOLUTE and takes the full affine. A 5 K rise is a 5 degC rise, not -268.15 degC (G15).
"""
import re
from typing import NamedTuple

import numpy as np

from .products import field_key            # noqa: F401  (re-exported: one canonical key)


class Affine(NamedTuple):
    """display = a * raw + b, carrying the label that describes the result."""
    label: str
    a: float = 1.0
    b: float = 0.0

    @property
    def is_identity(self):
        return (self.a, self.b) == (1.0, 0.0)

    def apply(self, x):
        """An ABSOLUTE value: instantaneous, or the mean over a window.

        The two degenerate cases are split out because they are the common ones, not the
        rare ones: K->degC is a pure offset (a = 1) and mm->kg m-2 a pure scale, and
        `x * 1.0 + b` walks a 3.4 MB frame twice to do one addition's work.
        """
        if self.is_identity:
            return x
        x = np.asarray(x)
        if self.a == 1.0:
            return x + self.b
        if self.b == 0.0:
            return x * self.a
        return x * self.a + self.b

    def apply_delta(self, d):
        """A DIFFERENCE (G15): the offset does not apply to a change in the quantity."""
        if self.a == 1.0:
            return d
        return np.asarray(d) * self.a

    def invert(self, y):
        """Display value -> raw value. Used to push a threshold down rather than 42k values."""
        return (np.asarray(y) - self.b) / self.a

    def apply_range(self, lo, hi):
        """Transform a cached (min, max) affinely (G19); a < 0 swaps the ends."""
        a, b = self.a * lo + self.b, self.a * hi + self.b
        return (a, b) if a <= b else (b, a)

    def apply_delta_range(self, lo, hi):
        """The same for a range of DIFFERENCES: scale only (G15)."""
        a, b = self.a * lo, self.a * hi
        return (a, b) if a <= b else (b, a)


IDENTITY = Affine('', 1.0, 0.0)


# ---- G21: units strings have many spellings; compare normalised, and NEVER raise --------
_DENOM = re.compile(r'^([a-zA-Z]+)(\d*)$')

_ALIASES = {
    'k': 'K', 'kelvin': 'K', 'degk': 'K', 'deg_k': 'K', 'degreek': 'K', 'degrees_k': 'K',
    'j kg-1': 'J kg-1', 'jkg-1': 'J kg-1',
    'm s-1': 'm s-1', 'ms-1': 'm s-1', 'meter second-1': 'm s-1',
    'kg m-2': 'kg m-2', 'kgm-2': 'kg m-2',
    'w m-2': 'W m-2', 'wm-2': 'W m-2',
    '%': '%', 'percent': '%',
    '1': '1', '-': '1', '': '1', 'fraction': '1', 'dimensionless': '1',
    'm': 'm', 'meter': 'm', 'metre': 'm', 'meters': 'm',
    'mm': 'mm', 'cm': 'cm',
    'degc': '°C', 'deg_c': '°C', 'celsius': '°C', 'c': '°C',
}


def normalise_units(units):
    """Canonical token for a CF units string. TOTAL: never raises, never returns None.

    An unrecognised string comes back normalised-but-unaliased, which lands in the
    "no conversion offered" branch (v2 1.3). A units string must not be able to crash a
    file open.
    """
    try:
        text = units if isinstance(units, str) else ('' if units is None else str(units))
        text = text.strip().lower().replace('**', '').replace('^', '')
        text = re.sub(r'\s+', ' ', text)
        if '/' in text:
            head, _, tail = text.partition('/')
            tail = tail.strip()
            match = _DENOM.match(tail)
            if match:
                tail = f'{match.group(1)}-{match.group(2) or 1}'
            text = f'{head.strip()} {tail}'.strip()
        return _ALIASES.get(text, text)
    except Exception:                      # totality is the whole point of this function
        return ''


# ---- the registry (v2 1.4): field name is the key, units string is a guard --------------
_TEMPERATURE = [Affine('°C', 1.0, -273.15), Affine('K', 1.0, 0.0),
                Affine('°F', 1.8, -459.67)]
# 1 m s-1 = 3600/1852 kt exactly, by the definition of the nautical mile.
_WIND = [Affine('m s-1'), Affine('kt', 3600.0 / 1852.0, 0.0), Affine('km h-1', 3.6, 0.0)]
_CLOUD_FIELDS = ('CLCT', 'CLCL', 'CLCM', 'CLCH')
# 1 kg m-2 of water over 1 m2 is 1 mm of depth: exact, so only the label changes.
_WATER = [Affine('mm'), Affine('kg m-2')]
# Pa -> hPa is what every pressure chart is drawn in, so it leads. The file stays Pa.
_PRESSURE = [Affine('hPa', 0.01), Affine('Pa')]


class FieldUnits(NamedTuple):
    expected: tuple          # acceptable canonical units strings for this field
    choices: list            # ordered; choices[0] is the display default
    gate: str = ''           # extra runtime check before the conversion is offered


# Every `expected` string below was MEASURED against the live server on 2026-08-24
# (run 2026082400) with tools/sniff_headers.py -- all 15 fields, no guesses left.
# Two spellings differ from what CLAUDE.md 0.2 documented and are handled by G21's
# normaliser rather than by widening this table:
#     ASWDIFD_S / ASWDIR_S  ->  units='W/m**2'   normalises to 'W m-2'
#     CLCT/CLCL/CLCM/CLCH   ->  units='%', values 0..100 (NOT 0-1)
UNITS = {
    'CAPE_ML':   FieldUnits(('J kg-1',), []),
    'T_2M':      FieldUnits(('K',), _TEMPERATURE),
    'T_S':       FieldUnits(('K',), _TEMPERATURE),
    # Not an IMS product: derived.DewPointView computes it from T_2M and RELHUM_2M and
    # ncwrite can save it. Registered as a Kelvin temperature so a written TD_2M file
    # reopens with exactly the treatment T_2M gets, degC default included.
    'TD_2M':     FieldUnits(('K',), _TEMPERATURE),
    'RELHUM_2M': FieldUnits(('%',), []),
    # kg m-2 -> mm is exact and free: 1 kg m-2 of water over 1 m2 is 1 mm depth.
    'TOT_PREC':  FieldUnits(('kg m-2',), [Affine('mm'), Affine('kg m-2')]),
    'U_10M':     FieldUnits(('m s-1',), _WIND),
    'V_10M':     FieldUnits(('m s-1',), _WIND),
    # Not an IMS product either: derived.WindView computes the speed from U_10M and V_10M
    # and ncwrite can save it. Registered so a written WSPD_10M reopens with exactly the
    # treatment its components get -- m s-1, kt, km h-1.
    'WSPD_10M':  FieldUnits(('m s-1',), _WIND),
    # ...and the same view built from the deterministic run's 3-D components, which is a
    # wind at a pressure level rather than at 10 m (derived.UPPER_WIND_FIELD).
    'WSPD':      FieldUnits(('m s-1',), _WIND),
    # VMAX_10M stays in file units: guessing a forecaster wants knots is a preference.
    'VMAX_10M':  FieldUnits(('m s-1',), [Affine('m s-1'), Affine('kt', 3600.0 / 1852.0),
                                         Affine('km h-1', 3.6)]),
    'ASWDIFD_S': FieldUnits(('W m-2',), []),
    'ASWDIR_S':  FieldUnits(('W m-2',), []),
    'H_SNOW':    FieldUnits(('m',), [Affine('cm', 100.0), Affine('m'), Affine('mm', 1000.0)]),
}

# ---- the deterministic ICON-LAM catalogue (IMS_ICON_manual.pdf, Table 1) ---------------
# Keyed like everything else on `field_key`, i.e. upper-cased, which is what makes the
# deterministic run's `t_2m`, `tot_prec` and `clct` inherit the entries their ensemble
# spellings already have: they are the same quantity out of the same model.
#
# UNMEASURED, unlike the 15 above. Every `expected` string here is transcribed from the
# manual's units column, not read off a file -- the credentials for the deterministic
# folder were not available here. That is exactly the case the registry's guard is for
# (v2 1.3): a units string this build does not expect means "warn and offer no
# conversion", never "convert on a guess", so a wrong row below costs a conversion and
# cannot corrupt a reading.
UNITS.update({
    # 3-D on pressure levels.
    'TEMP':    FieldUnits(('K',), _TEMPERATURE),
    'RH':      FieldUnits(('%',), []),
    'U':       FieldUnits(('m s-1',), _WIND),
    'V':       FieldUnits(('m s-1',), _WIND),
    # Pa s-1, and NEGATIVE is rising air. No conversion: the sign convention is the thing
    # to know about omega, and hPa h-1 would invite reading it as a speed.
    'OMEGA':   FieldUnits(('pa s-1',), []),
    # Geopotential is published as m2 s-2; a chart is drawn in geopotential metres, which
    # is that divided by the standard gravity. Exact by definition of gpm, so it leads.
    'GEOPOT':  FieldUnits(('m2 s-2',), [Affine('gpm', 1.0 / 9.80665), Affine('m2 s-2')]),
    # Surface.
    'T_G':     FieldUnits(('K',), _TEMPERATURE),
    'TMAX_2M': FieldUnits(('K',), _TEMPERATURE),
    'TMIN_2M': FieldUnits(('K',), _TEMPERATURE),
    'RH_2M':   FieldUnits(('%',), []),
    'QV_S':    FieldUnits(('kg kg-1',), [Affine('g kg-1', 1000.0), Affine('kg kg-1')]),
    'PRES_MSL': FieldUnits(('pa',), _PRESSURE),
    'PRES_SFC': FieldUnits(('pa',), _PRESSURE),
    'GUST10':  FieldUnits(('m s-1',), [Affine('m s-1'), Affine('kt', 3600.0 / 1852.0),
                                       Affine('km h-1', 3.6)]),
    # Convection.
    'CAPE':    FieldUnits(('J kg-1',), []),
    'CIN_ML':  FieldUnits(('J kg-1',), []),
    'HZEROCL': FieldUnits(('m',), []),
    'HBAS_CON': FieldUnits(('m',), []),
    'HTOP_CON': FieldUnits(('m',), []),
    # Precipitation: the same exact kg m-2 = mm identity TOT_PREC gets.
    'RAIN_GSP': FieldUnits(('kg m-2',), _WATER),
    'RAIN_CON': FieldUnits(('kg m-2',), _WATER),
    'SNOW_GSP': FieldUnits(('kg m-2',), _WATER),
    'SNOW_CON': FieldUnits(('kg m-2',), _WATER),
    'GRAUPEL_GSP': FieldUnits(('kg m-2',), _WATER),
    # Column integrals stay in kg m-2: a column of ice is not a depth of rain, so the
    # mm identity that TOT_PREC earns is not offered here.
    'TQV': FieldUnits(('kg m-2',), []),
    'TQC': FieldUnits(('kg m-2',), []),
    'TQI': FieldUnits(('kg m-2',), []),
    'TQR': FieldUnits(('kg m-2',), []),
    'TQS': FieldUnits(('kg m-2',), []),
    'TQG': FieldUnits(('kg m-2',), []),
    # Radiation.
    'ASODIFD_S': FieldUnits(('W m-2',), []),
    'ASODIFU_S': FieldUnits(('W m-2',), []),
    'ASODIRD_S': FieldUnits(('W m-2',), []),
    'SODIFD_S':  FieldUnits(('W m-2',), []),
    'SOB_T':     FieldUnits(('W m-2',), []),
    'ASOB_T':    FieldUnits(('W m-2',), []),
    'ATHB_T':    FieldUnits(('W m-2',), []),
})
# MEASURED 2026-08-24 against run 2026082400: all four report units='%' and values
# spanning 0..100, i.e. PERCENT -- not the 0-1 that CLAUDE.md 0.2 documented. The gate
# stays because the encoding is a property of the data, not of the header (G22): a future
# run that switches to fractions is decided correctly without a code change.
for _f in _CLOUD_FIELDS:
    UNITS[_f] = FieldUnits(('1', '%'), [], gate='cloud')


def is_cloud_field(field):
    """G22's runtime encoding test applies to `CLCT` and to the ICON run's `clct` alike."""
    return field_key(field) in _CLOUD_FIELDS


def accumulation_kind(field):
    """'sum' | 'mean' | None -- G14, for either family's spelling of the field."""
    return ACCUMULATION.get(field_key(field))


CLOUD_FRACTION = [Affine('%', 100.0), Affine('fraction')]


def cloud_encoding(sample):
    """G22 -- decide from the DATA, because the units string cannot settle it (v2 1.5).

    -> 'percent' | 'fraction' | 'undecidable'. Refusing to decide is a real answer: a
    x100 applied to a field already in % is a silent 100x error.
    """
    if sample is None:
        return 'undecidable'
    values = np.asarray(sample, dtype=np.float64)
    good = values[np.isfinite(values)]
    if good.size == 0:
        return 'undecidable'
    hi = float(good.max())
    if hi > 1.0 + 1e-6:
        return 'percent'
    if hi < 1e-6:
        return 'undecidable'               # clear sky: 0 is 0 in both encodings
    return 'fraction'


def choices_for(field, file_units, sample=None):
    """-> (ordered [Affine], note or None). choices[0] is the default; [] is never returned.

    Policy (v2 1.3): key on the field name, guard on the units string. A mismatch means IMS
    changed something, and the safe response to a possibly-stale assumption is to stop
    converting -- not to guess. An unknown field opens normally with no conversion.
    """
    label = (file_units or '').strip()
    passthrough = [Affine(label)]
    entry = UNITS.get(field_key(field))
    if entry is None:
        return passthrough, None

    canonical = normalise_units(file_units)
    if canonical not in entry.expected:
        return passthrough, (
            f'{field}: file says units={label!r} but this build expects '
            f'{" or ".join(entry.expected)}. No unit conversion is offered -- the registry '
            f'may be stale (v2 1.3).')

    if entry.gate == 'cloud':
        verdict = cloud_encoding(sample)
        if verdict == 'percent':
            return [Affine('%')], None if canonical == '%' else (
                f'{field}: units say {label!r} but values exceed 1, so the data is stored '
                'as % -- showing it as % and offering no conversion (G22).')
        if verdict == 'fraction':
            return list(CLOUD_FRACTION), None
        return passthrough, (
            f'{field}: cloud cover is all ~0 in this file, so the encoding (fraction vs %) '
            'cannot be decided from the data. No conversion offered (G22).')

    if not entry.choices:
        # No conversion, but the units string IS recognised, so show the canonical
        # spelling: the server sends 'W/m**2' and 'W m-2' is what everything else says.
        return [Affine(canonical)], None
    return list(entry.choices), None


# ---- across-member aggregation ----------------------------------------------------------
AGGREGATIONS = ('mean', 'max', 'min', 'median', 'spread')


def aggregate(stack, mode):
    """Aggregate a (n_members, ...) stack across members.

    One copy, because three layers need it -- `EnsembleFile`, `FieldView` and the derived
    views -- and three copies of a ladder like this drift. Note that callers convert the
    member stack BEFORE aggregating, which is what makes `spread` (a max-min DIFFERENCE,
    where an affine offset must cancel -- G15) correct by construction.
    """
    stack = np.asarray(stack)
    if stack.size and np.isnan(stack).all():
        # The window edge of a rate view: an all-NaN frame is intended, so do not let
        # numpy warn about it on every redraw.
        return np.full(stack.shape[1:], np.nan, dtype=stack.dtype)
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


# ---- de-accumulation (v2 section 3 / V2.3) ---------------------------------------------
# G14: TOT_PREC accumulates; ASWDIFD_S/ASWDIR_S are AVERAGED since model start. The formula
# differs, and np.diff is WRONG for radiation -- it yields numbers that look like plausible
# W m-2 and are not. Confirmed on synthetic data 2026-08-24:
#   stored running mean [0, 100, 250, 433.33, 400, 330]
#   np.diff            -> [100, 150, 183.33, -33.33, -70]     wrong, and plausible
#   mean-kind formula  -> [100, 400, 800, 300, 50]            the true hourly signal
ACCUMULATION = {'TOT_PREC': 'sum', 'ASWDIFD_S': 'mean', 'ASWDIR_S': 'mean'}

# The deterministic run's fields, from the manual. The five radiation fields whose names
# begin with `a` are described there as "mean since model start" in as many words -- that
# is the manual's own wording, not an inference -- while `sodifd_s` and `sob_t`, which are
# not, are deliberately absent. The precipitation amounts are accumulations by the ICON
# convention and by the measured behaviour of the ensemble's TOT_PREC; if one of them ever
# turns out not to be, the G24 guard says so on screen rather than showing fake drizzle.
ACCUMULATION.update({
    'RAIN_GSP': 'sum', 'RAIN_CON': 'sum', 'SNOW_GSP': 'sum', 'SNOW_CON': 'sum',
    'GRAUPEL_GSP': 'sum',
    'ASODIFD_S': 'mean', 'ASODIFU_S': 'mean', 'ASODIRD_S': 'mean',
    'ASOB_T': 'mean', 'ATHB_T': 'mean',
})

# VMAX_10M is deliberately absent: it is already a per-interval quantity ("max over the
# previous 1 h", CLAUDE.md 0.2) and must never be differenced.

RATE_HOURS = (0, 1, 3)                      # 0 = as stored
RATE_LABELS = {0: 'as stored', 1: '1 h', 3: '3 h'}


def rate_label(hours):
    return RATE_LABELS.get(hours, f'{hours} h')


def window_steps(hours, forecast_hours):
    """Window width in TIME-STEPS. Do not hardcode 1 step = 1 hour."""
    fh = np.asarray(forecast_hours, dtype=float)
    if fh.size < 2:
        return max(1, int(round(hours)))
    step = float(np.median(np.diff(fh)))
    if not np.isfinite(step) or step <= 0:
        return max(1, int(round(hours)))
    return max(1, int(round(hours / step)))


def noise_floor(reference, amplification=1.0):
    """The float32 rounding scale at a given magnitude -- the G24 clamp threshold.

    `amplification` covers the `mean` branch, where the stored value is multiplied by the
    forecast hour before subtracting: the error in `A[t]*h[t]` is `ulp(A) * h`, so by +120 h
    the noise floor is two orders of magnitude above the raw ulp. Measured on real
    ASWDIR_S: residuals of ~1e-3 W m-2 at h=22, against a raw ulp of 3e-5.
    """
    scale = max(abs(float(reference)), 1.0)
    return 8.0 * float(np.spacing(np.float32(scale))) * max(float(amplification), 1.0)


def _clamp_noise(values, eps):
    """G24 -> (values with rounding-scale negatives zeroed, count of REAL negatives).

    Every field in ACCUMULATION is non-negative by nature -- a precipitation amount and two
    downward radiative fluxes -- so a negative window value is either float32 noise (clamp
    it) or a genuine inconsistency in the data (surface it). Clamping both would hide G14;
    clamping neither would show fake drizzle and negative sunshine.
    """
    tiny = np.isfinite(values) & (values < 0) & (values > -eps)
    values = np.where(tiny, 0.0, values)
    return values, int(np.count_nonzero(np.isfinite(values) & (values <= -eps)))


def window_value(current, previous, h_now, h_prev, kind):
    """The value over the window (h_prev, h_now], from the two stored records.

    -> (values, is_difference, n_negative)

    G23: the subtraction happens in float64 unconditionally. Differencing two large
    near-equal accumulations is catastrophic cancellation, and the upcast removes the whole
    class of concern at no measurable cost for these array sizes.
    """
    a_now = np.asarray(current, dtype=np.float64)
    a_prev = np.asarray(previous, dtype=np.float64)
    span = float(h_now) - float(h_prev)
    if span <= 0:
        return np.full(a_now.shape, np.nan), kind == 'sum', 0

    reference = np.nanmax(np.abs(a_now)) if a_now.size else 1.0
    if kind == 'mean':
        # A window MEAN is absolute, so it takes the full affine later -- note the
        # asymmetry with 'sum' (G15).
        values = (a_now * float(h_now) - a_prev * float(h_prev)) / span
        values, n_negative = _clamp_noise(values, noise_floor(reference,
                                                              abs(float(h_now)) / span))
        return values, False, n_negative

    # G24: clamp float noise to zero, but let a REAL negative through as a warning --
    # otherwise the fix for G23 would hide the G14 bug it exists to expose.
    diff, n_negative = _clamp_noise(a_now - a_prev, noise_floor(reference))
    return diff, True, n_negative


def deaccumulate(values, forecast_hours, k, kind):
    """Whole-series version, time along axis 0. -> (values, is_difference, n_negative).

    The first k steps are NaN, not a partial window: losing 3 of 121 steps is nothing, and
    a silently-partial "3-hourly" total is a misread waiting to happen.
    """
    a = np.asarray(values, dtype=np.float64)
    hours = np.asarray(forecast_hours, dtype=float)
    out = np.full(a.shape, np.nan)
    if k <= 0 or k >= a.shape[0]:
        return out, kind == 'sum', 0
    shape = (-1,) + (1,) * (a.ndim - 1)
    h_now = hours[k:].reshape(shape)
    h_prev = hours[:-k].reshape(shape)
    span = h_now - h_prev

    reference = np.nanmax(np.abs(a)) if a.size else 1.0
    if kind == 'mean':
        with np.errstate(invalid='ignore', divide='ignore'):
            values = (a[k:] * h_now - a[:-k] * h_prev) / span
        eps = noise_floor(reference, float(np.nanmax(np.abs(h_now) / span)))
        values, n_negative = _clamp_noise(values, eps)
        out[k:] = values
        return out, False, n_negative

    diff, n_negative = _clamp_noise(a[k:] - a[:-k], noise_floor(reference))
    out[k:] = diff
    return out, True, n_negative


def rate_units_label(unit_label, hours, kind):
    """The label follows the KIND, and it is not cosmetic (v2 V2.3.5).

    A window mean of W m-2 is still W m-2; a window sum of mm is mm per that window.
    """
    if not hours or kind is None:
        return unit_label
    if kind == 'mean':
        return unit_label                       # a mean over a window is still a mean
    if hours == 1:
        return f'{unit_label} h-1'
    return f'{unit_label}/{hours}h'
