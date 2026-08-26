"""The map's colour scale: transparent where the field is zero, vivid everywhere else.

Two transforms over a pyqtgraph `ColorMap`, both of which exist because of how an ICON
field is actually distributed across its scale.

**Transparent at zero.** A CAPE map at 03 UTC is zero nearly everywhere, and turbo paints
zero as a near-black navy -- so the whole domain reads as a dark rectangle with the
coastline lost somewhere underneath it, and the eye has to hunt for the one place
something is happening. Zero is not a small value of CAPE, it is the *absence* of CAPE,
and the honest way to draw an absence is to draw nothing and let the geography show
through (see `geo.load_mapdata`'s land layer, which is what shows through).

The floor fades rather than switching: a model field is not "0 or 1400", it is a floor of
exact zeros with a smooth skirt of small values around every active cell. Cutting at
exactly 0.0 would leave that skirt painted solid and the map barely changed, so alpha
rises from nothing to opaque across the bottom `FADE` of the scale. Everything above that
is untouched -- the transform can lighten a map, never a value.

**Only when zero is the bottom of the scale.** `transparent_zero` is decided by the
caller from the colorbar's low end, not applied blindly to position 0.0 of the ramp. On a
2 m temperature map the bottom of the scale is the coldest air in the domain, which is a
reading and not an absence; fading it out would hide the coldest place on the map. So the
fade appears on CAPE, precipitation, snow depth, wind speed and every `spread` map -- the
fields whose floor is a true zero -- and nowhere else.

**Vivid.** With the floor gone the rest has to carry the map on its own, so each stop is
pushed away from grey (saturation gain, brightness held) and the near-black end of the
ramp is lifted off the floor (`VALUE_FLOOR`), because a colour that is almost black reads
as "nothing here" against a pale background -- which is exactly the reading the
transparency is there to give to zero, and to nothing else.

The result is cached: `refresh_map` runs on every frame of a 121-step scrub, and building
a 256-entry lookup table to arrive at the same colours is the sort of thing that only
ever shows up as "the slider feels heavy".
"""
from functools import lru_cache

import numpy as np
import pyqtgraph as pg

# Fraction of the scale over which alpha climbs from nothing to opaque. 5 % of a CAPE
# scale is ~160 J kg-1 -- below the threshold of anything a forecaster would act on, and
# wide enough to swallow the skirt of small values that a hard cut at 0.0 would leave.
FADE = 0.05
# The alpha ramp is bent so it leaves zero quickly and arrives gently: a linear ramp makes
# the top of the fade a visible edge, which reads as a contour that is not there.
FADE_SHAPE = 0.65

SATURATION_GAIN = 1.3
# Lift the darkest end of the ramp to at least this brightness. turbo starts at a navy of
# V = 0.23 and viridis at a purple of V = 0.27; both read as black over a pale map.
VALUE_FLOOR = 0.34

LUT_POINTS = 256


def _table(cmap):
    """-> (N, 4) float RGBA in 0..1, evenly spaced over the ramp."""
    lut = cmap.getLookupTable(0.0, 1.0, LUT_POINTS, alpha=True, mode=pg.ColorMap.FLOAT)
    return np.array(lut, dtype=float, copy=True)


def _build(pos, rgba):
    """`pg.ColorMap` from 0..1 float RGBA.

    The bytes are not cosmetic: `ColorMap.__init__` runs every colour through `mkColor`,
    which reads a 4-tuple as 0-255 integers -- so handing it floats in 0..1 silently
    builds a colormap that is black from end to end.
    """
    return pg.ColorMap(np.asarray(pos, dtype=float),
                       np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.ubyte))


def vivid(rgba, saturation=SATURATION_GAIN, floor=VALUE_FLOOR):
    """Push every stop away from grey and off the black floor, in place. -> rgba."""
    rgb = rgba[:, :3]
    # Saturation about the value: V = max(r, g, b) is held and the other channels are
    # pulled further down, which is an HSV saturation multiply without the round trip.
    value = rgb.max(axis=1, keepdims=True)
    np.clip(value - (value - rgb) * saturation, 0.0, 1.0, out=rgb)
    # Then lift V itself. `floor + (1 - floor) * V >= V` for every V in 0..1, so this only
    # ever brightens, and it leaves a fully bright colour exactly where it was.
    safe = np.maximum(value, 1e-6)
    np.clip(rgb * ((floor + (1.0 - floor) * value) / safe), 0.0, 1.0, out=rgb)
    return rgba


def fade_in_from_zero(rgba, fade=FADE, shape=FADE_SHAPE):
    """Ramp alpha from 0 at the bottom of the scale to opaque at `fade`, in place."""
    position = np.linspace(0.0, 1.0, len(rgba))
    ramp = np.clip(position / max(fade, 1e-6), 0.0, 1.0) ** shape
    rgba[:, 3] *= ramp
    return rgba


@lru_cache(maxsize=32)
def map_colormap(name, transparent_zero=False, punchy=True):
    """The colour scale the map should carry for `name`, cached by its arguments.

    `name` is a pyqtgraph colormap name; an unknown one falls back to viridis rather than
    raising, for the reason `transform.normalise_units` never raises -- a colour choice
    must not be able to stop a forecast being opened.
    """
    try:
        base = pg.colormap.get(name)
    except Exception:
        base = pg.colormap.get('viridis')
    if not punchy and not transparent_zero:
        return base
    rgba = _table(base)
    if punchy:
        vivid(rgba)
    if transparent_zero:
        fade_in_from_zero(rgba)
    return _build(np.linspace(0.0, 1.0, len(rgba)), rgba)


def with_transparent_top(pos, colours):
    """A colormap whose last stop keeps its colour but stops being painted.

    The R5 sort band ends at white, on the reasoning that above the top stop "the colours
    simply run out". That was true while the map's background was plain white; now that
    the land beneath is grey, an opaque white top stop would paint over the coastline in
    exactly the dry air the band is trying to say nothing about. The stop keeps its RGB --
    the colorbar still reads white at the top -- and loses its alpha, so the colours run
    out for real.
    """
    rgba = np.array([pg.colorTuple(pg.mkColor(colour)) for colour in colours],
                    dtype=float) / 255.0
    rgba[-1, 3] = 0.0
    return _build(pos, rgba)
