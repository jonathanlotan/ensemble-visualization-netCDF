"""Wind barb geometry: the convention, the arithmetic and the zoom rule (R4).

These are the tests that decide whether the map is *right* rather than merely pretty. A
barb pointing the wrong way, or feathers on the wrong side, produces a picture a
forecaster reads confidently and wrongly -- exactly the failure mode G14 and G17 are about
elsewhere in this project.
"""
import numpy as np
import pytest

from imsicon import barbs

# One screen pixel spans this much of the map. py is latitude, px longitude, and
# px = py / cos(lat) is what the aspect lock (G8) makes true.
COS = float(np.cos(np.deg2rad(31.25)))
PY = 0.00857
PX = PY / COS


def segments(geometry):
    """The NaN-separated polyline back as [(x0, y0), (x1, y1)] pairs."""
    xs, ys = geometry['lines']
    pairs, current = [], []
    for x, y in zip(xs, ys):
        if np.isnan(x):
            if len(current) >= 2:
                pairs.append(current)
            current = []
        else:
            current.append((float(x), float(y)))
    if len(current) >= 2:
        pairs.append(current)
    return pairs


def one_barb(u, v, px=PX, py=PY, length=barbs.SHAFT_PX):
    return barbs.barb_geometry([35.0], [31.25], [u], [v], px=px, py=py, cos_lat=COS,
                               length_px=length)


def pixels(point, px=PX, py=PY, origin=(35.0, 31.25)):
    """A geometry point as a pixel offset from the grid point it belongs to."""
    return np.array([(point[0] - origin[0]) / px, (point[1] - origin[1]) / py])


# ---- how a speed becomes feathers ------------------------------------------------------
@pytest.mark.parametrize('speed_kt, expected', [
    (0.0, (0, 0, 0)),            # calm
    (2.0, (0, 0, 0)),            # rounds to 0 kt: still calm
    (3.0, (0, 0, 1)),            # rounds to 5
    (7.0, (0, 0, 1)),            # rounds to 5
    (8.0, (0, 1, 0)),            # rounds to 10
    (12.5, (0, 1, 1)),           # rounds UP to 15, not down to 10 (see the rint note)
    (17.5, (0, 2, 0)),           # rounds up to 20
    (25.0, (0, 2, 1)),
    (47.0, (0, 4, 1)),           # 45 kt
    (52.0, (1, 0, 0)),           # 50 kt: one pennant, and no leftover feather
    (67.0, (1, 1, 1)),           # 65 kt
    (100.0, (2, 0, 0)),
])
def test_the_wmo_decomposition_of_a_speed(speed_kt, expected):
    assert tuple(int(np.asarray(c)) for c in barbs.barb_counts(speed_kt)) == expected


def test_counts_are_vectorised_over_a_whole_map():
    pennants, fulls, halves = barbs.barb_counts(np.array([[0.0, 10.0], [55.0, 5.0]]))
    assert pennants.shape == fulls.shape == halves.shape == (2, 2)
    assert pennants.tolist() == [[0, 0], [1, 0]]
    assert fulls.tolist() == [[0, 1], [0, 0]]
    assert halves.tolist() == [[0, 0], [1, 1]]


def test_a_nan_speed_draws_nothing_rather_than_a_wild_glyph():
    assert [int(np.asarray(c)) for c in barbs.barb_counts(np.nan)] == [0, 0, 0]


def test_an_absurd_speed_is_capped_instead_of_running_off_the_staff():
    pennants, fulls, halves = barbs.barb_counts(1e6)
    assert int(pennants) + int(fulls) + int(halves) <= 8


# ---- the convention: which way, and which side -----------------------------------------
@pytest.mark.parametrize('u, v, expected', [
    (0.0, -20.0, (0.0, 1.0)),          # wind FROM the north: staff points north
    (-20.0, 0.0, (1.0, 0.0)),          # from the east:  staff points east
    (0.0, 20.0, (0.0, -1.0)),          # from the south: staff points south
    (20.0, 0.0, (-1.0, 0.0)),          # from the west:  staff points west
])
def test_the_staff_points_into_the_wind(u, v, expected):
    """The one thing every reader of a barb assumes: the staff shows where it comes FROM."""
    staff = segments(one_barb(u, v))[0]
    direction = pixels(staff[1]) - pixels(staff[0])
    assert np.allclose(direction / np.linalg.norm(direction), expected, atol=1e-6)


def test_the_staff_is_the_same_length_in_pixels_whatever_the_zoom():
    for zoom in (0.25, 1.0, 4.0, 40.0):
        staff = segments(one_barb(0.0, -20.0, px=PX * zoom, py=PY * zoom))[0]
        length = np.linalg.norm(pixels(staff[1], PX * zoom, PY * zoom)
                                - pixels(staff[0], PX * zoom, PY * zoom))
        assert length == pytest.approx(barbs.SHAFT_PX, rel=1e-6)


def test_the_feathers_sit_on_the_right_of_the_staff_northern_hemisphere_style():
    """With the staff drawn pointing up (a wind from the north) the feathers go right.

    This is matplotlib's `barbs` default and the NH convention. Getting it backwards
    produces a map that looks entirely normal and reads as the southern hemisphere's.
    """
    staff, feather = segments(one_barb(0.0, -20.0))[:2]     # 20 kt = 2 full feathers
    along = pixels(staff[1]) - pixels(staff[0])
    across = pixels(feather[1]) - pixels(feather[0])
    # the component of the feather across the staff, positive to the staff's right
    right = np.array([along[1], -along[0]]) / np.linalg.norm(along)
    assert float(across @ right) > 0.5 * barbs.HEIGHT * barbs.SHAFT_PX


def test_a_half_feather_is_half_the_length_of_a_full_one():
    def reach(speed_kt):
        staff, feather = segments(one_barb(0.0, -speed_kt))[:2]
        along = pixels(staff[1]) - pixels(staff[0])
        right = np.array([along[1], -along[0]]) / np.linalg.norm(along)
        return float((pixels(feather[1]) - pixels(feather[0])) @ right)
    assert reach(10.0) == pytest.approx(2 * reach(5.0), rel=1e-6)


def test_a_lone_half_feather_is_set_in_from_the_tip():
    """...so 5 kt cannot be misread as 10 kt drawn at the end of the staff."""
    staff, half = segments(one_barb(0.0, -5.0))[:2]
    tip, root = pixels(staff[1]), pixels(staff[0])
    along = (tip - root) / np.linalg.norm(tip - root)
    assert float((tip - pixels(half[0])) @ along) > barbs.SPACING * barbs.SHAFT_PX


def test_a_calm_point_is_an_open_circle_and_no_staff():
    geometry = one_barb(0.4, 0.4)          # about 0.6 kt: rounds to nothing
    assert geometry['flags'].shape == (0, 3, 2)
    ring = np.array([pixels(p) for p in segments(geometry)[0]])
    radius = np.linalg.norm(ring, axis=1)
    assert radius.min() == pytest.approx(radius.max(), rel=1e-6)
    assert radius.mean() == pytest.approx(barbs.CALM_RADIUS * barbs.SHAFT_PX, rel=1e-6)
    assert len(segments(geometry)) == 1     # the ring, and nothing else


def test_a_pennant_is_a_filled_triangle_not_a_stroke():
    geometry = one_barb(0.0, -50.0)
    assert geometry['flags'].shape == (1, 3, 2)
    assert len(segments(geometry)) == 1     # only the staff is stroked


def test_the_glyph_is_square_on_screen_even_though_a_degree_is_not():
    """A degree of longitude is cos(lat) of a degree of latitude, so a barb built in
    degrees would be sheared. Built in pixels it is not: a north wind draws exactly
    vertical and an east wind exactly horizontal."""
    north = segments(one_barb(0.0, -20.0))[0]
    east = segments(one_barb(-20.0, 0.0))[0]
    assert north[0][0] == pytest.approx(north[1][0])      # no drift in longitude
    assert east[0][1] == pytest.approx(east[1][1])        # none in latitude
    length_of = lambda seg, i: abs(pixels(seg[1])[i] - pixels(seg[0])[i])
    assert length_of(north, 1) == pytest.approx(length_of(east, 0), rel=1e-6)


def test_points_with_no_wind_value_are_dropped_not_drawn_at_zero():
    geometry = barbs.barb_geometry([35.0, 35.1], [31.2, 31.3], [np.nan, 0.0],
                                   [5.0, -20.0], px=PX, py=PY, cos_lat=COS)
    for x in geometry['lines'][0][np.isfinite(geometry['lines'][0])]:
        assert x > 35.05            # everything drawn belongs to the second point


def test_a_degenerate_view_returns_nothing_instead_of_raising():
    for px, py in ((0.0, PY), (PX, np.nan), (-1.0, PY)):
        geometry = barbs.barb_geometry([35.0], [31.2], [5.0], [5.0], px=px, py=py)
        assert len(geometry['lines'][0]) == 0 and geometry['flags'].shape == (0, 3, 2)


# ---- the zoom rule ---------------------------------------------------------------------
def test_the_stride_falls_to_one_as_the_zoom_rises():
    """The requirement itself: barb resolution follows the zoom."""
    strides = [barbs.choose_stride(spacing) for spacing in (0.5, 2.0, 5.0, 12.0, 40.0)]
    assert strides == sorted(strides, reverse=True)
    assert strides[-1] == 1                        # zoomed right in: every grid point
    assert strides[0] > 10                         # zoomed right out: heavily thinned


def test_the_stride_keeps_barbs_about_a_target_distance_apart():
    for spacing in (1.0, 4.4, 9.0, 20.0, 33.0, 60.0):
        stride = barbs.choose_stride(spacing)
        drawn = stride * spacing
        assert drawn >= barbs.TARGET_SPACING_PX - spacing        # never crowded
        assert drawn <= 2.5 * barbs.TARGET_SPACING_PX            # never sparse


def test_the_coarsest_lattice_is_a_cap_not_a_target():
    """Zoomed out past the last stride the barbs simply stop thinning, rather than the
    lattice growing without limit. The domain is 161 columns; a stride of 100 is already
    two barbs across it, and the map's own pan limits stop well short of this."""
    assert barbs.choose_stride(0.05) == barbs.STRIDES[-1]


def test_a_nonsense_pixel_size_asks_for_the_coarsest_lattice_rather_than_dividing_by_zero():
    assert barbs.choose_stride(0.0) == barbs.STRIDES[-1]
    assert barbs.choose_stride(np.nan) == barbs.STRIDES[-1]


def test_the_lattice_is_anchored_so_panning_does_not_reshuffle_the_barbs():
    """Every index is a multiple of the stride, so a pan slides the same points across
    the window instead of picking a different set of grid cells each frame."""
    first = barbs.sample_indices(261, 8, 40.0, 120.0)
    second = barbs.sample_indices(261, 8, 47.0, 127.0)
    assert set(first) & set(second) == set(first[first >= 48])
    assert np.all(first % 8 == 0) and np.all(second % 8 == 0)


def test_the_sample_is_clipped_to_the_grid_and_to_the_window():
    assert barbs.sample_indices(10, 3, -50, 50).tolist() == [0, 3, 6, 9]
    assert barbs.sample_indices(10, 1, 4.2, 6.1).tolist() == [4, 5, 6, 7]
    assert barbs.sample_indices(10, 3, 8, 4).size == 0          # window off the grid
    assert barbs.sample_indices(0, 3).size == 0


# ---- direction, the one thing that must not become a statistic -------------------------
@pytest.mark.parametrize('u, v, degrees', [
    (0.0, -1.0, 0.0), (-1.0, 0.0, 90.0), (0.0, 1.0, 180.0), (1.0, 0.0, 270.0),
    (-1.0, -1.0, 45.0),
])
def test_direction_is_the_meteorological_from_direction(u, v, degrees):
    """v2.md 5.2's table, unchanged: N=0, E=90, S=180, W=270, and NE=45."""
    assert float(barbs.direction_from(u, v)) == pytest.approx(degrees)


def test_knots_are_exact_by_the_definition_of_the_nautical_mile():
    assert barbs.KT_PER_MS == pytest.approx(1.9438444924406046)
    assert 1852.0 * barbs.KT_PER_MS / 3600.0 == pytest.approx(1.0)
