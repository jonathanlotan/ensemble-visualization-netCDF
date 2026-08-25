"""Wind barbs: the WMO staff-and-feather glyph, as plain numpy geometry.

No Qt and no pyqtgraph in here, so the convention -- which way the staff points, which
side the feathers sit on, how a speed becomes feathers -- is testable without a window.

**The glyph.** A barb is drawn at a grid point with the staff pointing in the direction
the wind comes FROM, and the feathers at the far (upwind) end:

    half feather = 5 kt      full feather = 10 kt      pennant = 50 kt

The speed is rounded to the nearest 5 kt before it is decomposed, and a speed that rounds
to zero is drawn as an open circle (calm) rather than as a bare staff -- a staff with no
feathers would read as "10 kt of something I forgot to draw".

**Which side the feathers are on** is not decoration: it is the hemisphere convention.
With the staff drawn pointing up (a wind from the north) the feathers extend to the
*right*, which is matplotlib's `barbs` default and the northern-hemisphere convention. In
vector terms, with `s` the unit staff direction the feathers run along `r = (s_y, -s_x)`,
i.e. `s` turned 90 degrees clockwise. This domain (28-34.5 N) is entirely northern, so the
southern-hemisphere flip is deliberately not offered.

**Everything is built in PIXEL offsets and converted back to degrees at the end**, using
the data units one screen pixel spans (`px`, `py` -- pyqtgraph's `viewPixelSize`). Two
things follow, and both are the point of doing it this way:

* a barb is the same size on screen at every zoom, because its size is *defined* in
  pixels rather than in degrees;
* the angle is right regardless of the map's aspect ratio. A degree of longitude is
  `cos(lat)` of a degree of latitude on the ground, and the map is aspect-locked to
  exactly that (CLAUDE.md **G8**), so a ground direction `(u east, v north)` lands on
  screen at `(u / (cos_lat * px), v / py)` -- which reduces to being parallel to `(u, v)`
  when the lock holds, and self-corrects when it does not.
"""
import numpy as np

# 1 m s-1 = 3600/1852 kt exactly, by the definition of the nautical mile (same constant
# the units registry uses for the kt display option).
KT_PER_MS = 3600.0 / 1852.0

HALF_KT = 5.0                   # one half feather
STEPS_PER_FULL = 2              # a full feather is two half-feathers, i.e. 10 kt
STEPS_PER_PENNANT = 10          # a pennant is ten, i.e. 50 kt
# 39 five-knot steps is 195 kt: 3 pennants + 4 feathers + 1 half, the most that fits on
# the staff. Nothing this model produces comes near it; the cap exists so a corrupt value
# cannot ask for a hundred feathers running off the far end of the glyph.
MAX_STEPS = 39

# Proportions of the staff length. The staff is the only absolute size; everything else is
# a fraction of it, so one number sets how big barbs are on screen.
SHAFT_PX = 20.0
SPACING = 0.16                  # gap between successive feathers, along the staff
HEIGHT = 0.42                   # how far a full feather reaches across the staff
PENNANT_WIDTH = 0.22            # the pennant's base, along the staff
CALM_RADIUS = 0.16              # the open circle drawn where the wind rounds to zero
CALM_POINTS = 13

# How far apart barbs should sit on screen, and the grid strides that are allowed to get
# them there. Snapping to round numbers keeps the lattice stable: a stride that crept
# 7, 8, 9, 10 through a zoom would reshuffle every barb on the map at each step.
TARGET_SPACING_PX = 34.0
STRIDES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100)


def barb_counts(speed_kt):
    """-> (pennants, fulls, halves) for a speed in knots, rounded to the nearest 5 kt.

    All three come back as integer arrays shaped like the input, so a whole map is
    decomposed in one call. `pennants + fulls + halves == 0` is the calm case.
    """
    speed = np.asarray(speed_kt, dtype=float)
    usable = np.isfinite(speed)
    # floor(x + 0.5), not np.rint: rint rounds halves to even, so 12.5 kt would draw as
    # 10 and 17.5 kt as 20. Round half up is what a forecaster expects of a barb.
    steps = np.where(usable, np.floor(np.maximum(speed, 0.0) / HALF_KT + 0.5), 0.0)
    steps = np.minimum(steps, MAX_STEPS).astype(int)
    pennants = steps // STEPS_PER_PENNANT
    rest = steps - STEPS_PER_PENNANT * pennants
    fulls = rest // STEPS_PER_FULL
    return pennants, fulls, rest - STEPS_PER_FULL * fulls


def choose_stride(spacing_px, target_px=TARGET_SPACING_PX, strides=STRIDES):
    """How many grid cells to skip so neighbouring barbs land ~`target_px` apart.

    This is the whole "resolution follows the zoom" rule: `spacing_px` is what one grid
    cell measures on screen right now, so zooming in raises it and the stride falls to 1
    (a barb at every grid point), while zooming out drops it and the stride grows.
    """
    if not np.isfinite(spacing_px) or spacing_px <= 0:
        return strides[-1]
    needed = target_px / float(spacing_px)
    for stride in strides:
        if stride >= needed - 1e-9:
            return stride
    return strides[-1]


def sample_indices(n, stride, lo=None, hi=None):
    """Grid indices to draw: the lattice `0, stride, 2*stride, ...` clipped to [lo, hi].

    Anchored to index 0 rather than to the visible window, so panning slides the barbs
    across the screen instead of re-picking a different set of grid points every frame.
    """
    n = int(n)
    stride = max(1, int(stride))
    first = 0 if lo is None else max(0, int(np.floor(lo)))
    last = n - 1 if hi is None else min(n - 1, int(np.ceil(hi)))
    if last < first or n <= 0:
        return np.empty(0, dtype=int)
    start = -(-first // stride) * stride            # first lattice point >= first
    return np.arange(start, last + 1, stride, dtype=int)


def _staff_direction(u, v, px, py, cos_lat):
    """Unit vectors, in pixel space: the staff (upwind) and the feather side (its right)."""
    # Ground east/north -> degrees -> pixels. cos_lat converts a ground-east component
    # into degrees of longitude; px and py convert degrees into pixels.
    dx = u / (max(float(cos_lat), 1e-6) * px)
    dy = v / py
    norm = np.hypot(dx, dy)
    norm = np.where(norm > 0, norm, 1.0)
    sx, sy = -dx / norm, -dy / norm                 # the staff points into the wind
    return sx, sy, sy, -sx                          # feathers: the staff turned clockwise


def barb_geometry(x, y, u, v, px, py, cos_lat=1.0, length_px=SHAFT_PX):
    """Barb glyphs for N points, in DATA (lon/lat) coordinates.

    `x`, `y` are longitude and latitude; `u`, `v` the wind in KNOTS, u eastward and v
    northward; `px`, `py` the degrees of longitude and latitude one screen pixel spans.

    -> {'lines': (xs, ys), 'flags': (n, 3, 2)}

    `lines` is every stroke -- staffs, feathers and calm circles -- as one NaN-separated
    polyline pair, so a whole map of barbs is a single `PlotDataItem` rather than
    thousands of scene items (the same trick `geo.py` uses for the coastline). `flags` is
    the pennant triangles, which want filling rather than stroking.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    x, y, u, v = x[keep], y[keep], u[keep], v[keep]

    lines_x, lines_y, flags = [], [], []
    empty = (np.empty(0), np.empty(0))
    if x.size == 0 or not (np.isfinite(px) and np.isfinite(py)) or px <= 0 or py <= 0:
        return {'lines': empty, 'flags': np.empty((0, 3, 2))}

    length = float(length_px)
    spacing, height = SPACING * length, HEIGHT * length
    pennant_width = PENNANT_WIDTH * length

    pennants, fulls, halves = barb_counts(np.hypot(u, v))
    elements = pennants + fulls + halves
    sx, sy, rx, ry = _staff_direction(u, v, px, py, cos_lat)

    def add_lines(idx, ax, ay, bx, by):
        """Segments from (ax, ay) to (bx, by), given as PIXEL offsets from point `idx`."""
        if idx.size == 0:
            return
        gap = np.full(idx.size, np.nan)
        lines_x.append(np.stack([x[idx] + ax * px, x[idx] + bx * px, gap], axis=1).ravel())
        lines_y.append(np.stack([y[idx] + ay * py, y[idx] + by * py, gap], axis=1).ravel())

    # The staff, for every point that is not calm.
    staff = np.nonzero(elements > 0)[0]
    add_lines(staff, np.zeros(staff.size), np.zeros(staff.size),
              length * sx[staff], length * sy[staff])

    # A half feather standing alone is set in from the tip, so it cannot be misread as a
    # full one drawn at the end of the staff. (The same rule matplotlib follows.)
    lone_half = (pennants + fulls == 0) & (halves > 0)

    for slot in range(int(elements.max()) if elements.size else 0):
        is_pennant = slot < pennants
        is_full = (slot >= pennants) & (slot < pennants + fulls)
        is_half = (slot >= pennants + fulls) & (slot < elements)
        # Distance from the point to this element, along the staff: the tip, less what
        # the elements nearer the tip have already used up.
        used = (np.minimum(slot, pennants) * (pennant_width + spacing)
                + np.clip(slot - pennants, 0, fulls) * spacing)
        along = length - used - np.where(is_half & lone_half, 1.5 * spacing, 0.0)

        feather = np.nonzero(is_full | is_half)[0]
        if feather.size:
            reach = np.where(is_full[feather], height, 0.5 * height)
            base = along[feather]
            # Tilted toward the tip by half a pennant width, which is what gives a barb
            # its swept-back look instead of a comb of perpendicular spikes.
            tip = base + 0.5 * pennant_width
            add_lines(feather, base * sx[feather], base * sy[feather],
                      tip * sx[feather] + reach * rx[feather],
                      tip * sy[feather] + reach * ry[feather])

        pennant = np.nonzero(is_pennant)[0]
        if pennant.size:
            base = along[pennant]
            corners = []
            for offset, out in ((base, 0.0), (base - pennant_width, 0.0),
                                (base - 0.5 * pennant_width, height)):
                corners.append(np.stack(
                    [x[pennant] + (offset * sx[pennant] + out * rx[pennant]) * px,
                     y[pennant] + (offset * sy[pennant] + out * ry[pennant]) * py],
                    axis=1))
            flags.append(np.stack(corners, axis=1))

    calm = np.nonzero(elements == 0)[0]
    if calm.size:
        angle = np.linspace(0.0, 2.0 * np.pi, CALM_POINTS)
        radius = CALM_RADIUS * length
        gap = np.full((calm.size, 1), np.nan)
        ring_x = x[calm][:, None] + radius * np.cos(angle)[None, :] * px
        ring_y = y[calm][:, None] + radius * np.sin(angle)[None, :] * py
        lines_x.append(np.concatenate([ring_x, gap], axis=1).ravel())
        lines_y.append(np.concatenate([ring_y, gap], axis=1).ravel())

    return {
        'lines': ((np.concatenate(lines_x), np.concatenate(lines_y))
                  if lines_x else empty),
        'flags': np.concatenate(flags) if flags else np.empty((0, 3, 2)),
    }


def direction_from(u, v):
    """Meteorological wind direction in degrees -- the direction it comes FROM.

    N=0, E=90, S=180, W=270 (v2.md 5.2). **G16**: this is circular data. It is fine to
    read one value off it, and wrong to take a linear mean, min, max or percentile of it
    -- the ensemble's direction has to come from the mean VECTOR, which is why the barbs
    aggregate u and v and never these degrees.
    """
    return np.mod(270.0 - np.degrees(np.arctan2(np.asarray(v, dtype=float),
                                                np.asarray(u, dtype=float))), 360.0)
