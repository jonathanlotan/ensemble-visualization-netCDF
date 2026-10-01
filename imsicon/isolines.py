"""Isolines: the contour geometry, and the interval each field is contoured at.

Both halves live here because they answer one question -- *how is this field contoured* --
and splitting the interval into the units registry while the geometry sat somewhere else
would put the two numbers a reader has to compare in two files.

    frame (display units) ──► levels_for(min, max, step) ──► contour_lines() ──► MapView

**An interval is stated in the view's CANONICAL units, as a spacing and an anchor, and
the two convert differently (**G15**).** The *spacing* is a difference and takes the
affine's scale alone -- 1 degC is 1 K and 1.8 degF. The *anchor* is a value and takes the
whole affine, which is the half that is easy to miss: contouring "every 1 degree Celsius"
means the lines sit on whole degrees Celsius, so in Kelvin they sit at 273.15, 274.15 and
in Fahrenheit at 32.0, 33.8 -- the SAME lines, relabelled. Spacing alone would have
anchored them on whole Kelvin or whole Fahrenheit instead, quietly drawing a different set
of lines each time the Units combo moved. A view whose values are themselves a difference
(T-Td) takes the scale for both, because zero difference is zero in every unit.

**Why the marching squares here and not `geo.contour_segments`.** `geo` contours one
level (`fr_land == 0.5`) once, at startup, and caches it, so a Python loop over 16k cells
was fine. A field is contoured at ~40 levels on *every* redraw of a 261x161 grid, inside
v2's 16.7 ms frame budget, so the same loop would be 1.7 million iterations per frame.
This one is vectorised, and `geo` now delegates to it -- one marching squares in the
codebase rather than two that can disagree. The cost is measured in R5.4.
"""
from typing import NamedTuple

import numpy as np

from . import products


# ---- the spacings on offer -------------------------------------------------------------
class Ladder(NamedTuple):
    """The spacings the Isolines slider offers for one kind of quantity (R5.9, R9).

    `steps` are stated in the quantity's NATURAL unit -- degrees for a temperature, gpm
    for a geopotential height -- and `scale` is how many canonical units one natural unit
    is (1 K per degC, 9.80665 m2 s-2 per gpm), so the view is always handed a canonical
    spacing and G15 keeps applying. A ladder per kind rather than one for the app, because
    "every 2" has to mean something on a 500 hPa height chart too, and 2 gpm would be a
    hatch pattern there while 2 degC is a reading.
    """
    steps: tuple
    scale: float = 1.0
    emphasis: dict = {}      # natural step -> every Nth line heavier
    unit: str = 'degrees'    # how the natural unit is called in help text

    def canonical(self, step):
        return float(step) * self.scale

    def natural(self, canonical):
        return float(canonical) / self.scale

    def nearest(self, step):
        """-> the offered NATURAL spacing closest to `step` (natural units)."""
        return min(self.steps, key=lambda choice: abs(choice - float(step)))

    def index_of(self, canonical):
        """-> slider notch for a CANONICAL spacing (the nearest one)."""
        return self.steps.index(self.nearest(self.natural(canonical)))

    def emphasis_for(self, step, default=5):
        """-> how many lines apart the heavy ones are, for a NATURAL spacing."""
        return self.emphasis.get(round(float(step), 6), default)


# The temperature ladder: canonical (i.e. Celsius) degrees, which is the whole of G15
# applied to a user's choice. Picking 2 with the Units combo on degF draws the SAME lines
# as picking 2 on degC did, labelled 3.6 degF; anything else would move every line on the
# map as a side effect of relabelling the colorbar. The emphasis puts the heavy line on a
# number a forecaster would name -- 1, 5, 10, 6 and 20 degrees -- which is a table rather
# than a formula because 3 has no round multiple near 5.
DEGREES = Ladder((0.5, 1.0, 2.0, 3.0, 4.0), 1.0,
                 {0.5: 2, 1.0: 5, 2.0: 5, 3.0: 2, 4.0: 5}, '°C')

# Standard gravity: the m2 s-2 of geopotential per geopotential metre, by definition.
G0 = 9.80665
# One foot of geopotential height, in m2 s-2: 0.3048 m exactly, times g.
FT = 0.3048 * G0
# The height ladder, in feet -- the unit the chart is read in by default (R9). 200 ft is
# 61 gpm, close to the 6 dam a 500 hPa chart is conventionally drawn at; 100 ft (30 m)
# suits 850 hPa and 1,000 ft a jet-level chart. The emphasis puts the heavy line on a
# whole thousand feet from every notch: every 10th at 100, 5th at 200, 4th at 250, 2nd
# at 500.
HEIGHT = Ladder((100.0, 200.0, 250.0, 500.0, 1000.0, 2000.0), FT,
                {100.0: 10, 200.0: 5, 250.0: 4, 500.0: 2, 1000.0: 5, 2000.0: 5}, 'ft')


# ---- the interval policy ---------------------------------------------------------------
class Interval(NamedTuple):
    """How a quantity is contoured: lines at `anchor + k * step`, every `emphasis`th heavy.

    Held in the view's canonical units by the registry, and in display units once
    `scaled` has been through it. `ladder` is what the slider offers for it.
    """
    step: float
    emphasis: int = 5        # every Nth line from the anchor is drawn heavier
    anchor: float = 0.0      # a value a line must pass through: 273.15 K, i.e. 0 degC
    ladder: Ladder = DEGREES

    def at_step(self, step):
        """-> the same contouring, re-spaced to `step` (in the interval's own units).

        The **anchor is kept**, so every spacing still puts a line on 0 degC and the map
        is re-read rather than redrawn somewhere else; only the emphasis follows the step,
        because "every 5th line" means 5 degC at a 1 degC spacing and 2.5 degC at 0.5 --
        and a heavy line that lands on a number nobody would name is no landmark at all.

        A step that is not a positive number leaves the interval alone: this is reached
        from a slider and a command line, and a bad value must cost the map its spacing
        choice, never its lines.
        """
        step = abs(float(step))
        if not np.isfinite(step) or step <= 0:
            return self
        ladder = self.ladder or DEGREES
        return Interval(step, ladder.emphasis_for(ladder.natural(step), self.emphasis),
                        self.anchor, ladder)

    def scaled(self, affine, difference=False):
        """-> the same lines expressed in DISPLAY units.

        The spacing takes the scale and the anchor takes the whole affine, so the lines
        stay where they are and only their labels change (**G15**, and see the module
        docstring). `difference=True` for a view whose values are already a difference,
        where the offset cancels and zero stays zero.

        `abs` on the spacing, because a negative scale would flip its sign while leaving
        the lines themselves exactly where they were.
        """
        anchor = (affine.apply_delta(self.anchor) if difference
                  else affine.apply(self.anchor))
        return Interval(abs(float(affine.apply_delta(self.step))), self.emphasis,
                        float(anchor), self.ladder)


# Contoured at 1 degC, with every 5th line (a round 5 degC) heavier. Keyed by field name
# and stated in Kelvin -- the canonical space every temperature in this app is held in --
# so `T_2M`, `T_S` and a computed or saved `TD_2M` are all contoured identically. The
# anchor is the freezing point, which is what makes the lines whole degrees CELSIUS
# rather than whole Kelvin: 1 K of spacing from 0 K would put them at 20.85 degC.
ZERO_CELSIUS_K = 273.15
_TEMPERATURE = Interval(1.0, 5, ZERO_CELSIUS_K)
STEPS = {
    'T_2M': _TEMPERATURE,
    'T_S': _TEMPERATURE,
    'TD_2M': _TEMPERATURE,
    # The deterministic run's temperatures, keyed the same canonical way (`field_key`):
    # `temp` on pressure levels, the 2 m temperature and dew point, the grid-mean surface
    # temperature, and the daily extremes. An 850 hPa chart is read at 1 degC exactly as a
    # 2 m one is, so they share the interval rather than getting a second table.
    'TEMP': _TEMPERATURE,
    'T_G': _TEMPERATURE,
    'TMAX_2M': _TEMPERATURE,
    'TMIN_2M': _TEMPERATURE,
    # R9: the geopotential on pressure levels, read as a HEIGHT chart. The file is m2 s-2
    # (canonical); the lines are every 200 ft (61 gpm, about the conventional 6 dam of a
    # 500 hPa chart), anchored at 0 so they sit on round feet, with every 5th (1,000 ft)
    # heavier. The HEIGHT ladder is what the slider offers instead of the degrees one.
    'GEOPOT': Interval(200.0 * FT, 5, 0.0, HEIGHT),
}

# A DIFFERENCE of two contoured fields -- the dew point depression above all, but equally
# T_2M - T_S -- is contoured twice as finely, with every 2nd line (a round 1 degC) heavier.
# The reason is the range, not symmetry: a temperature map spans 20 degC or more across
# the domain, while the depression a forecaster is reading is in the 0-5 degC band where
# half a degree is the difference between fog and no fog.
# Anchored at zero, which needs no conversion: no difference is no difference in degC,
# K and degF alike.
DIFFERENCE = Interval(0.5, 2, 0.0)

# The temperature ladder's steps, by the name R5.9 gave them. Stated as absolute spacings
# rather than as multiples of the field's own interval, because "every 2 degrees" has to
# mean the same thing on a temperature map and on a depression -- a multiplier would make
# one notch 2 degC on T_2M and 1 degC on T-Td while the control read the same.
STEP_CHOICES = DEGREES.steps


def emphasis_for(step, default=5):
    """-> how many lines apart the heavy ones are, for a chosen spacing in degrees."""
    return DEGREES.emphasis_for(step, default)


def nearest_step(step):
    """-> the offered degree spacing closest to `step`, for placing the slider on a field
    whose registry interval (0.5 on a difference, 1 on a temperature) is one of them."""
    return DEGREES.nearest(step)


def ladder_for(field):
    """-> the `Ladder` the slider should offer for a field: its interval's, else degrees."""
    interval = interval_for(field)
    return interval.ladder if interval is not None and interval.ladder else DEGREES


# Above this many lines a map is a hatch pattern rather than a reading, so the step is
# coarsened by a whole-number factor (and `levels_for` reports the step it actually used,
# because a title that names an interval the lines are not drawn at is worse than none).
MAX_LINES = 60


def interval_for(field):
    """-> `Interval` in canonical units for a field name, or None if it is not contoured.

    Keyed on `products.field_key`, so the two families' spellings of one quantity
    (`T_2M` and `t_2m`) are contoured identically rather than one of them not at all.
    """
    return STEPS.get(products.field_key(field))


def levels_for(lo, hi, step, anchor=0.0, cap=None):
    """-> (levels, step_used) for a frame spanning `lo..hi`.

    Levels sit at `anchor + k*step` for whole k, never at the frame's own minimum plus a
    multiple: the 20 degC isoline has to be at 20 degC in every frame, or scrubbing time
    would slide every line across the map as the data range breathed.
    """
    step = abs(float(step))
    lo, hi, anchor = float(lo), float(hi), float(anchor)
    cap = MAX_LINES if cap is None else cap
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(step)
            and np.isfinite(anchor)) or step <= 0:
        return np.empty(0), step
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if cap and cap > 0:
        # ceil division: the smallest whole factor that fits the cap. Whole, so every line
        # drawn at the coarser step was also a line at the finer one.
        factor = max(1, int(-(-(int(span / step) + 1) // cap)))
        step *= factor
    # The 1e-9 keeps a level sitting exactly on the frame's own min or max -- an isothermal
    # patch, or a difference that touches 0 -- from falling in or out with float noise.
    first = int(np.ceil((lo - anchor) / step - 1e-9))
    last = int(np.floor((hi - anchor) / step + 1e-9))
    if last < first:
        return np.empty(0), step
    return anchor + np.arange(first, last + 1) * step, step


def emphasis_mask(levels, step, emphasis, anchor=0.0):
    """-> a boolean per level: True for the ones drawn heavier.

    Those are every `emphasis`th line counted from the anchor -- 5 degC on a temperature
    map, 1 degC on a depression -- so the heavy lines land on round numbers rather than on
    whichever line happened to come first in this frame.
    """
    levels = np.asarray(levels, dtype=float)
    if levels.size == 0 or not emphasis or emphasis <= 1 or step <= 0:
        return np.zeros(levels.size, dtype=bool)
    return np.rint((levels - anchor) / step).astype(np.int64) % int(emphasis) == 0


def split_emphasis(levels, step, emphasis, anchor=0.0):
    """-> (ordinary, emphasised) levels, the two layers the map draws separately."""
    levels = np.asarray(levels, dtype=float)
    index = emphasis_mask(levels, step, emphasis, anchor)
    return levels[~index], levels[index]


# ---- marching squares -------------------------------------------------------------------
# Corner bits, anticlockwise from the bottom left of a cell:
#     bit 0 = a (y0, x0)   bit 1 = b (y0, x1)   bit 2 = c (y1, x1)   bit 3 = d (y1, x0)
# Edges: 0 = a-b (bottom), 1 = b-c (right), 2 = d-c (top), 3 = a-d (left).
# Each row is up to two segments as (from_edge, to_edge, from_edge, to_edge), -1 unused.
# Rows 16 and 17 are the saddle cases 5 and 10 resolved the other way -- see `_SADDLES`.
_TABLE = np.array([
    [-1, -1, -1, -1],      # 0  ....
    [3, 0, -1, -1],        # 1  a
    [0, 1, -1, -1],        # 2  b
    [3, 1, -1, -1],        # 3  ab
    [1, 2, -1, -1],        # 4  c
    [3, 0, 1, 2],          # 5  ac   saddle, centre below: a and c are separate
    [0, 2, -1, -1],        # 6  bc
    [3, 2, -1, -1],        # 7  abc
    [2, 3, -1, -1],        # 8  d
    [0, 2, -1, -1],        # 9  ad
    [0, 1, 2, 3],          # 10 bd   saddle, centre below: b and d are separate
    [1, 2, -1, -1],        # 11 abd
    [3, 1, -1, -1],        # 12 cd
    [0, 1, -1, -1],        # 13 acd
    [0, 3, -1, -1],        # 14 bcd
    [-1, -1, -1, -1],      # 15 abcd
    [0, 1, 2, 3],          # 16 case 5,  centre above: b and d are the separate ones
    [3, 0, 1, 2],          # 17 case 10, centre above: a and c are the separate ones
], dtype=np.int8)
# Cases where the four corners alternate above/below, so all four edges cross and the
# pairing is a genuine choice. Resolved by the cell's mean: whichever side of the level
# the middle of the cell is on is the side that stays connected through it.
_SADDLES = {5: 16, 10: 17}


def _crossings(z, levels):
    """Every (cell, level) pair that holds a line -> (cells, level_index).

    The whole point of doing it this way. The obvious spelling loops over levels and
    scans the grid for each one, which is 40 passes over 42,000 cells per frame even
    though a single isoline touches a few hundred of them. Here the grid is walked
    **once**: a binary search places each cell's corner range in the level ladder, which
    gives the *count* of levels it crosses, and `repeat` expands that straight into the
    (cell, level) pairs. Everything downstream then works on the crossings that exist --
    typically 13,000 of a possible 1.4 million.

    A cell holds the line of `level` when `low < level <= high`, matching the `>=`
    convention the case table is built on. Cells touching a NaN produce `low = high = NaN`
    and both searches return the end of the ladder, so they contribute nothing.

    The corner arrays stay 2-D views: `np.minimum` over them writes one contiguous result
    whose `ravel` is free, where slicing each corner out first would copy the grid four
    times before any work started.
    """
    a, b = z[:-1, :-1], z[:-1, 1:]
    c, d = z[1:, 1:], z[1:, :-1]
    low = np.minimum(np.minimum(a, b), np.minimum(c, d)).ravel()
    high = np.maximum(np.maximum(a, b), np.maximum(c, d)).ravel()

    first = np.searchsorted(levels, low, side='right')     # first level above `low`
    last = np.searchsorted(levels, high, side='right')     # one past the last <= `high`
    counts = np.maximum(last - first, 0)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    cells = np.repeat(np.arange(counts.size, dtype=np.intp), counts)
    # 0,1,2,... within each cell's run of levels, without a Python loop over the runs.
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    within = np.arange(total, dtype=np.intp) - np.repeat(starts, counts)
    return cells, np.repeat(first.astype(np.intp), counts) + within


def _segments(x, y, z, levels):
    """-> (xs, ys, level_index per segment). The shared core of the two public calls."""
    nx = z.shape[1]
    cells, at = _crossings(z, levels)
    if cells.size == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.intp)
    level = levels[at]
    rows, cols = np.divmod(cells, nx - 1)
    # Gathered for the crossing cells only -- a few thousand of the grid's forty-odd
    # thousand -- which is why the corners are not materialised in `_crossings`. Flat
    # indices into the raveled grid rather than `z[rows, cols]`: a 1-D take is markedly
    # cheaper than two-dimensional fancy indexing for the same values.
    flat = z.ravel()
    corner = rows * nx + cols
    av, bv = flat[corner], flat[corner + 1]
    cv, dv = flat[corner + nx + 1], flat[corner + nx]
    x0, x1 = x[cols], x[cols + 1]
    y0, y1 = y[rows], y[rows + 1]
    # Only the edges the case table selects are ever read, and on those one corner is >=
    # the level while the other is <, so the denominator cannot be zero. The guard covers
    # the edges that are computed and discarded.
    with np.errstate(divide='ignore', invalid='ignore'):
        f0 = (level - av) / (bv - av)
        f2 = (level - dv) / (cv - dv)
        edge_x = np.stack([x0 + f0 * (x1 - x0), x1, x0 + f2 * (x1 - x0), x0])
        f1 = (level - bv) / (cv - bv)
        f3 = (level - av) / (dv - av)
        edge_y = np.stack([y0, y0 + f1 * (y1 - y0), y1, y0 + f3 * (y1 - y0)])

    case = ((av >= level).astype(np.uint8)
            | ((bv >= level).astype(np.uint8) << 1)
            | ((cv >= level).astype(np.uint8) << 2)
            | ((dv >= level).astype(np.uint8) << 3))
    variant = case.astype(np.intp)
    if ((case == 5) | (case == 10)).any():
        above = ((av + bv + cv + dv) * 0.25) >= level
        for plain, flipped in _SADDLES.items():
            variant[(case == plain) & above] = flipped
    pairs = _TABLE[variant]

    index = np.arange(cells.size)
    out_x, out_y, out_level = [], [], []
    for first in (0, 2):                            # up to two segments per cell
        start, end = pairs[:, first], pairs[:, first + 1]
        drawn = start >= 0
        if not drawn.any():
            continue
        here = index[drawn]
        segment_x = np.empty((here.size, 3))
        segment_y = np.empty((here.size, 3))
        segment_x[:, 0] = edge_x[start[drawn], here]
        segment_x[:, 1] = edge_x[end[drawn], here]
        segment_y[:, 0] = edge_y[start[drawn], here]
        segment_y[:, 1] = edge_y[end[drawn], here]
        segment_x[:, 2] = np.nan                    # the break `connect='finite'` reads
        segment_y[:, 2] = np.nan
        out_x.append(segment_x)
        out_y.append(segment_y)
        out_level.append(at[drawn])
    if not out_x:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.intp)
    return (np.concatenate(out_x), np.concatenate(out_y), np.concatenate(out_level))


def _prepare(x, y, z, levels):
    """Validate and normalise the inputs both public entry points share.

    A float32 field is kept in float32 -- every frame this app draws is float32, and the
    comparisons and the binary search are the bulk of the cost, so halving their memory
    traffic is most of the difference between contouring inside the frame budget and not.
    The crossing positions still land in float64, because the coordinates are; float32
    would place a line to about 0.3 m on a 2.5 km grid either way. Anything else is
    contoured in float64, so a caller passing exact values keeps them.
    """
    z = np.asarray(z)
    z = z if z.dtype == np.float32 else np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != z.shape[1] or y.size != z.shape[0]:
        raise ValueError(f'coordinates {y.size}x{x.size} do not match the '
                         f'{z.shape[0]}x{z.shape[1]} field')
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    # `_crossings` binary-searches the ladder, so it must be sorted; callers pass
    # `levels_for` output, which already is, but a hand-written list need not be. The cast
    # keeps the search and the case bits deciding on the same numbers.
    levels = np.sort(levels[np.isfinite(levels)]).astype(z.dtype, copy=False)
    return (x, y, z, levels) if levels.size else None


def contour_lines(x, y, z, levels):
    """-> (xs, ys): 2-point segments separated by NaN, for every level at once.

    `x`/`y` are the cell-centre coordinates of `z`, so the lines come back in degrees and
    one `PlotDataItem` with `connect='finite'` draws the whole set. Cells touching a NaN
    are skipped, which is what a rate view's opening gap and the dew point's blanked cells
    need: they must leave a hole, not a line drawn around one.

    Segments come out unordered rather than stitched into paths. Nothing here needs a
    closed polygon -- no filling, no labels running along the line -- and stitching 40
    levels of a 261x161 grid into ordered paths every frame is exactly the per-point
    Python cost this module exists to avoid.
    """
    ready = _prepare(x, y, z, levels)
    if ready is None:
        return np.empty(0), np.empty(0)
    xs, ys, _at = _segments(*ready)
    return xs.ravel(), ys.ravel()


def contour_set(x, y, z, interval, cap=None):
    """Everything the map needs for one frame, in one call.

    -> {'ordinary': (xs, ys), 'emphasised': (xs, ys), 'levels': array, 'step': float}

    `interval.step` is already in the frame's own units -- the affine conversion happens
    on the view (`FieldView.isolines`), not here, so this function never has to know what
    a unit is. Ordinary and emphasised lines are split *after* the geometry rather than
    contoured separately, so the emphasis costs a boolean mask instead of a second pass.
    """
    z = np.asarray(z)
    good = z[np.isfinite(z)]
    blank = {'ordinary': (np.empty(0), np.empty(0)),
             'emphasised': (np.empty(0), np.empty(0)),
             'levels': np.empty(0), 'step': float(interval.step) if interval else 0.0}
    if interval is None or good.size == 0:
        return blank
    levels, step = levels_for(good.min(), good.max(), interval.step, interval.anchor,
                              cap=cap)
    if levels.size == 0:
        return dict(blank, step=step)
    ready = _prepare(x, y, z, levels)
    if ready is None:
        return dict(blank, step=step)
    xs, ys, at = _segments(*ready)
    if xs.size == 0:
        return dict(blank, levels=levels, step=step)
    heavy = emphasis_mask(levels, step, interval.emphasis, interval.anchor)[at]
    return {'ordinary': (xs[~heavy].ravel(), ys[~heavy].ravel()),
            'emphasised': (xs[heavy].ravel(), ys[heavy].ravel()),
            'levels': levels, 'step': step}
