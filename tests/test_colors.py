"""The map's colour scale: transparent at a true zero, vivid everywhere else (R6).

`ui/colors.py` has no Qt widgets in it, only pyqtgraph colour objects, so these are plain
unit tests of the two transforms and of the one predicate that decides whether the first
of them applies at all.
"""
import numpy as np
import pyqtgraph as pg
import pytest

from imsicon.ui import colors
from imsicon.ui.main import _zero_is_the_floor


def lut(cmap, n=256):
    """(n, 4) uint8 RGBA -- what the ImageItem actually paints with."""
    return np.asarray(cmap.getLookupTable(0.0, 1.0, n, alpha=True))


# ---- the alpha ramp ---------------------------------------------------------------------
def test_the_bottom_of_the_scale_is_not_painted_at_all():
    table = lut(colors.map_colormap('turbo', transparent_zero=True))
    assert table[0][3] == 0


def test_everything_above_the_fade_is_fully_opaque():
    table = lut(colors.map_colormap('turbo', transparent_zero=True))
    above = int(np.ceil(colors.FADE * (len(table) - 1))) + 1
    assert (table[above:, 3] == 255).all()
    # and the fade really is a fade, not a step: something in between is part way
    assert 0 < table[max(above // 2, 1)][3] < 255


def test_the_fade_only_ever_lightens_and_only_at_the_bottom():
    """A colour scale that changed a value's colour would be lying about the value."""
    opaque = lut(colors.map_colormap('turbo', transparent_zero=False))
    faded = lut(colors.map_colormap('turbo', transparent_zero=True))
    assert np.array_equal(opaque[:, :3], faded[:, :3])       # same colours throughout
    assert (faded[:, 3] <= opaque[:, 3]).all()               # only ever less painted


def test_without_the_flag_nothing_is_transparent():
    table = lut(colors.map_colormap('turbo', transparent_zero=False))
    assert (table[:, 3] == 255).all()


# ---- the vivid transform ----------------------------------------------------------------
def _saturation(rgb):
    top = rgb.max(axis=1).astype(float)
    bottom = rgb.min(axis=1).astype(float)
    return np.where(top > 0, (top - bottom) / np.maximum(top, 1e-9), 0.0)


def test_vivid_pushes_the_ramp_away_from_grey():
    stock = lut(pg.colormap.get('turbo'))
    punchy = lut(colors.map_colormap('turbo'))
    assert _saturation(punchy[:, :3]).mean() > _saturation(stock[:, :3]).mean()


def test_vivid_lifts_the_near_black_end_off_the_floor():
    """turbo starts at a navy that reads as black over a pale map -- which is the reading
    the transparency is there to give to zero, and to nothing else."""
    stock = lut(pg.colormap.get('turbo'))
    punchy = lut(colors.map_colormap('turbo'))
    assert stock[0][:3].max() < 0.30 * 255
    assert punchy[0][:3].max() > 0.34 * 255


def test_vivid_never_darkens_a_stop():
    stock = lut(pg.colormap.get('turbo'))
    punchy = lut(colors.map_colormap('turbo'))
    assert (punchy[:, :3].max(axis=1) >= stock[:, :3].max(axis=1) - 1).all()


def test_a_fully_bright_colour_is_left_where_it_was():
    """The lift is `floor + (1 - floor) * V`, which is the identity at V = 1."""
    rgba = np.array([[1.0, 1.0, 0.0, 1.0]])
    colors.vivid(rgba)
    assert rgba[0][:3].max() == pytest.approx(1.0)


def test_an_unknown_colormap_name_falls_back_instead_of_raising():
    """`transform.normalise_units`' rule: a display choice must not stop a file opening."""
    table = lut(colors.map_colormap('no such ramp'))
    assert table.shape == (256, 4)


def test_the_colormap_is_cached_so_a_scrub_does_not_rebuild_it():
    first = colors.map_colormap('turbo', transparent_zero=True)
    assert colors.map_colormap('turbo', transparent_zero=True) is first
    assert colors.map_colormap('turbo', transparent_zero=False) is not first


def test_the_stops_survive_the_trip_through_mkColor():
    """A ColorMap built from floats in 0..1 is black end to end: `ColorMap.__init__` runs
    every colour through `mkColor`, which reads a 4-tuple as 0-255 integers."""
    table = lut(colors.map_colormap('turbo'))
    assert table[:, :3].max() > 200


# ---- the sort band's top stop -----------------------------------------------------------
def test_the_sort_band_keeps_its_white_and_loses_its_paint():
    cmap = colors.with_transparent_top([0.0, 0.5, 1.0],
                                       ['#d7191c', '#fdae61', '#ffffff'])
    table = lut(cmap, 3)
    assert tuple(table[0]) == (215, 25, 28, 255)
    assert tuple(table[-1]) == (255, 255, 255, 0)


# ---- the predicate ----------------------------------------------------------------------
@pytest.mark.parametrize('lo, hi, expected', [
    (0.0, 3252.0, True),            # CAPE, dataset range
    (0.0, 0.0, True),               # H_SNOW in summer: an all-zero field
    (0.0, 12.0, True),              # spread, which is pinned to zero
    (17.4, 43.2, False),            # a temperature map: the floor is the coldest air
    (-4.0, 38.0, False),            # zero inside the scale is an ordinary reading
    (-12.0, 12.0, False),           # a symmetric difference map
])
def test_zero_is_the_floor(lo, hi, expected):
    assert _zero_is_the_floor(lo, hi) is expected


def test_a_floor_that_has_been_through_a_units_affine_still_counts():
    """0 kg m-2 of rain scaled to another unit is not exactly 0.0 any more."""
    assert _zero_is_the_floor(3e-9, 25.0)


def test_a_nan_range_is_not_a_zero_floor():
    """G20's family: an all-NaN rate frame must not reach the colormap as a decision."""
    assert not _zero_is_the_floor(float('nan'), 1.0)
    assert not _zero_is_the_floor(0.0, float('nan'))
