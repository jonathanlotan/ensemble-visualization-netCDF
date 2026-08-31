"""FieldView -- a transparent decorator over EnsembleFile that applies the v2 transforms.

    memmap -> EnsembleFile -------------> raw physical values  (R1 code, unmodified)
                   |
                   v
              FieldView  <-- Transform (rate . units)
                   |
        +----------+----------+---------------+
        v          v          v               v
     MapView   PlotView   ReadoutPanel   status bar

`EnsembleFile` is never modified: keeping it pure as the raw-truth layer is what lets the
transforms be tested by round-tripping against it. Everything not listed below is delegated
verbatim, so MapView / PlotView / ReadoutPanel need no change at all.
"""
import numpy as np

from . import isolines, transform
from .transform import Affine


class FieldView:
    """Mirrors EnsembleFile's attribute surface, producing DISPLAY values."""

    def __init__(self, raw, units_label=None):
        self.raw = raw
        sample = self._encoding_sample() if transform.is_cloud_field(raw.field) else None
        self.unit_choices, self.units_note = transform.choices_for(
            raw.field, raw.units, sample)
        self._units = self.unit_choices[0]
        if units_label:
            self.set_units(units_label)

        # G14: only these three fields accumulate. VMAX_10M is already a per-interval
        # quantity and is deliberately not in the table.
        self.accum_kind = transform.accumulation_kind(raw.field)
        self.rate_hours = 0
        self.rate_note = None

    # ---- everything else is EnsembleFile's ------------------------------------------
    def __getattr__(self, name):
        # Only called when normal lookup fails. Guarded so an access before __init__
        # finishes raises AttributeError instead of recursing forever.
        try:
            raw = object.__getattribute__(self, 'raw')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(raw, name)

    def __repr__(self):
        return f'<FieldView {self.raw.field} in {self.units!r}>'

    # ---- units -----------------------------------------------------------------------
    def _encoding_sample(self):
        """A few frames for the G22 cloud range test -- cheap, and only for CLC* fields."""
        try:
            idx = sorted({0, self.raw.n_times // 2, self.raw.n_times - 1})
            return np.concatenate([self.raw.ens_frame(t).ravel() for t in idx])
        except Exception:
            return None

    @property
    def unit_labels(self):
        return [c.label for c in self.unit_choices]

    @property
    def units_affine(self):
        return self._units

    def set_units(self, label):
        """Select one of `unit_labels`. Unknown labels leave the current choice alone."""
        for choice in self.unit_choices:
            if choice.label == label:
                self._units = choice
                return True
        return False

    @property
    def can_convert_units(self):
        return len(self.unit_choices) > 1

    @property
    def units(self):
        return transform.rate_units_label(self._units.label, self.rate_hours,
                                          self.accum_kind)

    # ---- rate -------------------------------------------------------------------------
    @property
    def can_rate(self):
        return self.accum_kind is not None

    @property
    def rate_choices(self):
        return [transform.rate_label(h) for h in
                (transform.RATE_HOURS if self.can_rate else (0,))]

    def set_rate(self, hours):
        """Window width in HOURS; 0 restores the stored (cumulative) values."""
        hours = int(hours or 0)
        if hours and not self.can_rate:
            return False
        self.rate_hours = hours
        self.rate_note = None
        return True

    @property
    def window_steps(self):
        """The window in TIME-STEPS -- never assume 1 step is 1 hour."""
        if not self.rate_hours or not self.can_rate:
            return 0
        return transform.window_steps(self.rate_hours, self.raw.forecast_hours)

    def _note_negatives(self, count):
        # G24: a real negative in a 'sum' rate means the field is not actually
        # accumulated. Surfaced, not clamped away -- otherwise the fix for G23 hides G14.
        if not count:
            return
        why = ('the stored total decreases' if self.accum_kind == 'sum'
               else 'the implied cumulative total decreases')
        self.rate_note = (
            f'{self.field}: {count} negative value(s) in the {self.rate_hours} h rate, '
            f'because {why}. This field cannot be negative, so either the accumulation '
            f'kind is wrong (G14) or those cells are inconsistent in the file (G24).')

    def _window(self, t, reader):
        """[1] time-differencing. -> (values, is_difference), NaN before the first window."""
        k = self.window_steps
        if not k:
            return reader(t), False
        if t < k:
            # A partial window would be a misread waiting to happen: show a gap instead.
            return np.full(np.shape(reader(0)), np.nan), self.accum_kind == 'sum'
        hours = self.raw.forecast_hours
        values, is_diff, negatives = transform.window_value(
            reader(t), reader(t - k), hours[t], hours[t - k], self.accum_kind)
        self._note_negatives(negatives)
        return values, is_diff

    # ---- reads: raw -> [1] differencing -> [2] units -> display ----------------------
    def _absolute(self, values):
        return np.asarray(self._units.apply(values), dtype=np.float32)

    def _to_display(self, values, is_difference):
        """[2] unit conversion, told what kind of quantity step [1] produced (G15)."""
        converted = (self._units.apply_delta(values) if is_difference
                     else self._units.apply(values))
        return np.asarray(converted, dtype=np.float32)

    def frame(self, t, member):
        values, is_diff = self._window(t, lambda i: self.raw.frame(i, member))
        return self._to_display(values, is_diff)

    def ens_frame(self, t):
        values, is_diff = self._window(t, self.raw.ens_frame)
        return self._to_display(values, is_diff)

    def agg_frame(self, t, mode):
        """Aggregate the DISPLAY stack, not the raw one.

        Converting first makes every aggregation correct by construction, including
        `spread` (a max-min DIFFERENCE, where the affine offset must cancel -- G15) and a
        hypothetical negative scale, which would swap max and min.
        """
        return transform.aggregate(self.ens_frame(t), mode)

    def series(self, iy, ix):
        raw = self.raw.series(iy, ix)
        k = self.window_steps
        if not k:
            return self._absolute(raw)
        values, is_diff, negatives = transform.deaccumulate(
            raw, self.raw.forecast_hours, k, self.accum_kind)
        self._note_negatives(negatives)
        return self._to_display(values, is_diff)

    # ---- labels describe the TRANSFORMED quantity ------------------------------------
    @property
    def long_name(self):
        return self.raw.long_name

    @property
    def display_name(self):
        """What the map title and the y axis call this. A derived view overrides it --
        `T-Td` reads better than `T_2M-TD_2M` -- so every caller can just ask for it."""
        return self.field

    # ---- isolines --------------------------------------------------------------------
    @property
    def isolines(self):
        """Contour interval in DISPLAY units, or None when this field is not contoured.

        The registry states it in canonical units, as a spacing and an anchor, and the
        two convert differently (**G15**): the spacing takes the affine's scale, the
        anchor takes the whole affine. That is what keeps "every 1 degree Celsius" true
        when the Units combo says degF -- the SAME lines, at 32.0 and 33.8 degF, rather
        than a new set anchored on whole Fahrenheit.

        A rate view returns None: `TOT_PREC` is not contoured either way, but were a
        contoured field ever accumulated, its 1 h window would be a different quantity
        from the value the interval was chosen for.
        """
        interval = isolines.interval_for(self.field)
        if interval is None or self.rate_hours:
            return None
        return interval.scaled(self._units, difference=self.is_difference_view)

    @property
    def transform_signature(self):
        """Identifies the view a cached range belongs to (G19). Units are NOT part of it:
        a unit change is affine and transforms the cached range instead of rescanning."""
        if not self.rate_hours or not self.can_rate:
            return 'raw'
        return f'rate:{self.accum_kind}:{self.rate_hours}'

    @property
    def is_difference_view(self):
        """True when what is on screen is a change in the quantity, not the quantity."""
        return bool(self.rate_hours) and self.accum_kind == 'sum'

    def _display_range(self, cached):
        if cached is None:
            return None
        lo, hi = float(cached[0]), float(cached[1])
        return (self._units.apply_delta_range(lo, hi) if self.is_difference_view
                else self._units.apply_range(lo, hi))

    @property
    def value_range(self):
        """G19: a pure unit change is affine, so transform the cached range, never rescan.

        The cached value is in step-[1] space (differenced but not yet converted), so this
        holds for a rate view too -- switching mm to kg m-2 never costs a scan.
        """
        return self._display_range(self.raw.range_for(self.transform_signature))

    def level_range(self, index):
        """The cached range of ONE position on the second axis, in display units."""
        return self._display_range(
            self.raw.level_range(index, self.transform_signature))

    # ---- saving: for a file-backed field, what you see is what you save ---------------
    @property
    def canonical_units(self):
        """`ncwrite.write_canonical` writes a plain field exactly as displayed.

        A derived view overrides this to name a unit-independent space of its own (Kelvin
        for the dew point). A field read from a file has no such space worth preferring --
        saving a rate view back in its cumulative form would be a different quantity, not
        a different unit -- so the honest answer is the one on screen.
        """
        return self.units

    def canonical_ens_frame(self, t):
        return self.ens_frame(t)

    def label_for(self, t):
        """A rate is BACKWARD-looking; a reader who takes it as instantaneous is off by
        one interval, so the window is named in the label."""
        base = self.raw.label_for(t)
        k = self.window_steps
        if not k:
            return base
        if t < k:
            return f'{base}  [no {self.rate_hours} h window yet]'
        return (f'{base}  [{self.rate_hours} h to '
                f'{self.raw.times[t]:%H:%M}Z]')

    def summary(self):
        short = (f' | INCOMPLETE: {self.n_times}/{self.raw.declared_times} steps'
                 if self.raw.truncated else '')
        return (f'{self.path.name} | {self.field} ({self.long_name}) [{self.units}] | '
                f'run {self.run_init:%Y-%m-%d %H:%M}Z | {self.axis.describe()} | '
                f'{self.n_times} steps{short} | {self.ny}x{self.nx} grid')

    # ---- range caching, per transform view (G19) --------------------------------------
    def _window_frame(self, t):
        """Step [1] only -- what the cached range is measured in."""
        return self._window(t, self.raw.ens_frame)[0]

    def cached_range(self):
        return self.raw.cached_range(self.transform_signature)

    def scan_range(self, progress=None, cancel=None):
        signature = self.transform_signature
        source = None if signature == 'raw' else self._window_frame
        return self.raw.scan_range(progress=progress, cancel=cancel,
                                   signature=signature, frame_source=source)
