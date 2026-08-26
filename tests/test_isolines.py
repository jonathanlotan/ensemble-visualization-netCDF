"""The contour geometry and the interval policy (R5).

The load-bearing claim is that `isolines.contour_lines` -- which is vectorised, walks the
grid once for the whole level set, and resolves saddles from a table -- draws exactly what
the obvious cell-by-cell loop would. `_reference` below IS that obvious loop, written
independently from the marching-squares definition, and `test_the_fast_path_matches_the_
obvious_loop` holds the two together on random fields. The same relationship
`tests/test_nc3.py` keeps between the fast reader and `netCDF4`.
"""
import numpy as np
import pytest

import synth
from imsicon import derived, isolines, ncwrite
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView
from imsicon.transform import Affine


# ---- a deliberately naive marching squares, as the oracle -------------------------------
def _reference(x, y, z, levels):
    """Cell by cell, level by level, in plain Python. -> a set of rounded segments.

    Returned as a set of unordered endpoint pairs, because nothing about the fast path
    promises an order -- only that the same lines come out.
    """
    z = np.asarray(z, dtype=float)
    out = set()
    for level in np.atleast_1d(levels):
        for i in range(z.shape[0] - 1):
            for j in range(z.shape[1] - 1):
                a, b = z[i, j], z[i, j + 1]
                c, d = z[i + 1, j + 1], z[i + 1, j]
                if not np.isfinite(a + b + c + d):
                    continue
                if min(a, b, c, d) >= level or max(a, b, c, d) < level:
                    continue
                x0, x1, y0, y1 = x[j], x[j + 1], y[i], y[i + 1]

                def on(v0, v1, p0, p1):
                    if v1 == v0:            # an edge the pairing below never selects
                        return (np.nan, np.nan)
                    f = (level - v0) / (v1 - v0)
                    return (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]))

                edge = {0: on(a, b, (x0, y0), (x1, y0)), 1: on(b, c, (x1, y0), (x1, y1)),
                        2: on(d, c, (x0, y1), (x1, y1)), 3: on(a, d, (x0, y0), (x0, y1))}
                bits = ((a >= level) | (b >= level) << 1
                        | (c >= level) << 2 | (d >= level) << 3)
                pairs = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
                         6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)],
                         11: [(1, 2)], 12: [(3, 1)], 13: [(0, 1)], 14: [(0, 3)]}
                if bits in (5, 10):
                    middle = (a + b + c + d) / 4.0 >= level
                    isolate_ac = (bits == 5) != middle
                    chosen = [(3, 0), (1, 2)] if isolate_ac else [(0, 1), (2, 3)]
                else:
                    chosen = pairs.get(bits, [])
                for first, second in chosen:
                    ends = tuple(sorted((tuple(np.round(edge[first], 9)),
                                         tuple(np.round(edge[second], 9)))))
                    out.add((round(float(level), 9),) + ends)
    return out


def _as_set(xs, ys, level):
    """The fast path's output in the same shape as the oracle's."""
    out = set()
    xs, ys = np.asarray(xs), np.asarray(ys)
    for k in range(0, xs.size, 3):
        ends = tuple(sorted(((round(float(xs[k]), 9), round(float(ys[k]), 9)),
                             (round(float(xs[k + 1]), 9), round(float(ys[k + 1]), 9)))))
        out.add((round(float(level), 9),) + ends)
    return out


@pytest.fixture
def ramp():
    """z = x: every contour of it is a vertical line at exactly that value."""
    x = np.arange(0.0, 6.0)
    y = np.arange(0.0, 5.0)
    return x, y, np.broadcast_to(x, (y.size, x.size)).astype(np.float64).copy()


# ---- the geometry ----------------------------------------------------------------------
def test_a_contour_of_a_ramp_lies_exactly_on_the_level(ramp):
    x, y, z = ramp
    xs, ys = isolines.contour_lines(x, y, z, [2.0, 3.5])
    drawn = xs[np.isfinite(xs)]
    assert drawn.size
    assert set(np.round(drawn, 9)) == {2.0, 3.5}
    # and each line spans the full height of the grid, in 2-point segments
    assert np.nanmin(ys) == pytest.approx(0.0)
    assert np.nanmax(ys) == pytest.approx(4.0)


def test_segments_are_separated_by_nan_so_one_item_draws_them_all(ramp):
    xs, ys = isolines.contour_lines(*ramp, [2.0])
    assert xs.size % 3 == 0 and xs.size == ys.size
    assert np.all(np.isnan(xs[2::3])) and np.all(np.isnan(ys[2::3]))
    assert np.isfinite(xs[0::3]).all() and np.isfinite(xs[1::3]).all()


def test_a_level_outside_the_data_draws_nothing(ramp):
    for level in (-1.0, 99.0):
        xs, ys = isolines.contour_lines(*ramp, [level])
        assert xs.size == 0 and ys.size == 0


def test_a_cell_touching_a_nan_is_left_blank_not_drawn_around():
    """The rate view's opening gap and the dew point's blanked cells must leave a hole."""
    x = y = np.arange(4.0)
    z = np.tile(np.arange(4.0), (4, 1))
    z[1, 1] = np.nan
    xs, _ys = isolines.contour_lines(x, y, z, [1.5])
    drawn = xs[np.isfinite(xs)]
    assert set(np.round(drawn, 9)) == {1.5}            # what is left is still the 1.5 line
    # Three cells span 1.5; the NaN corner is shared by two of them, and both drop out.
    assert xs.size // 3 == 1
    whole, _ = isolines.contour_lines(x, y, np.tile(np.arange(4.0), (4, 1)), [1.5])
    assert whole.size // 3 == 3


def test_an_all_nan_frame_draws_nothing_rather_than_raising():
    x = y = np.arange(4.0)
    xs, ys = isolines.contour_lines(x, y, np.full((4, 4), np.nan), [0.0, 1.0])
    assert xs.size == 0 and ys.size == 0


def test_a_saddle_is_cut_into_two_segments_that_do_not_cross():
    """Both diagonals crossing one cell would draw an X through it, which is the one
    thing a contour must never do: it would join two regions that are not connected."""
    z = np.array([[0.0, 1.0], [1.0, 0.0]])
    xs, ys = isolines.contour_lines([0.0, 1.0], [0.0, 1.0], z, [0.5])
    assert xs.size == 6                                # two segments, NaN-separated
    corners = {(round(float(a), 6), round(float(b), 6))
               for a, b in zip(xs, ys) if np.isfinite(a)}
    # each segment cuts one corner off, so every endpoint is a mid-edge point
    assert corners == {(0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0)}
    first = [(xs[0], ys[0]), (xs[1], ys[1])]
    assert {tuple(np.round(p, 6)) for p in first} in (
        {(0.0, 0.5), (0.5, 0.0)}, {(1.0, 0.5), (0.5, 1.0)})


def test_a_saddle_is_resolved_by_the_middle_of_the_cell():
    """The two pairings are both valid marching squares; which one is right depends on
    where the centre of the cell sits, and getting it backwards joins the wrong lobes."""
    low_centre = np.array([[0.0, 10.0], [10.0, 0.0]])          # mean 5 > level 1
    xs, ys = isolines.contour_lines([0.0, 1.0], [0.0, 1.0], low_centre, [1.0])
    ends = {tuple(np.round((a, b), 6)) for a, b in zip(xs[:2], ys[:2])}
    assert ends == {(0.0, 0.1), (0.1, 0.0)}     # the below-corner at the origin is cut off
    xs, ys = isolines.contour_lines([0.0, 1.0], [0.0, 1.0], low_centre, [9.0])
    ends = {tuple(np.round((a, b), 6)) for a, b in zip(xs[:2], ys[:2])}
    # The centre is now BELOW the level, so the two above-corners are the islands and the
    # cut runs bottom-to-right around one of them -- not bottom-to-left, which is what the
    # other pairing would draw and which would join the wrong pair of lobes.
    assert ends == {(0.9, 0.0), (1.0, 0.1)}


def test_a_peak_gives_a_closed_ring_around_it():
    x = y = np.linspace(-2.0, 2.0, 41)
    xx, yy = np.meshgrid(x, y)
    z = 10.0 - (xx ** 2 + yy ** 2)                     # a paraboloid peaking at the origin
    xs, ys = isolines.contour_lines(x, y, z, [9.0])    # the ring at radius 1
    radius = np.hypot(xs[np.isfinite(xs)], ys[np.isfinite(ys)])
    assert radius.size > 20
    assert np.allclose(radius, 1.0, atol=0.03)


def test_the_fast_path_matches_the_obvious_loop():
    """The whole point of the vectorised version: it is an optimisation, not a variant."""
    rng = np.random.default_rng(11)
    x = np.linspace(33.0, 34.0, 17)
    y = np.linspace(28.0, 29.5, 13)
    for trial in range(4):
        z = rng.normal(0.0, 1.0, (y.size, x.size)).round(1)   # ties on purpose
        if trial == 3:
            z[3, 4] = z[7, 9] = np.nan
        levels, _step = isolines.levels_for(np.nanmin(z), np.nanmax(z), 0.25)
        fast = set()
        for level in levels:
            fast |= _as_set(*isolines.contour_lines(x, y, z, [level]), level)
        assert fast == _reference(x, y, z, levels)


def test_one_call_for_many_levels_equals_a_call_for_each():
    rng = np.random.default_rng(5)
    x, y = np.linspace(0, 3, 13), np.linspace(0, 2, 11)
    z = rng.normal(size=(y.size, x.size))
    levels = np.arange(-1.0, 1.01, 0.25)
    together = isolines.contour_lines(x, y, z, levels)
    apart = [isolines.contour_lines(x, y, z, [level]) for level in levels]

    def key(xs, ys):
        finite = np.isfinite(xs)
        return sorted(zip(np.round(xs[finite], 9), np.round(ys[finite], 9)))
    assert key(*together) == sorted(sum((key(*part) for part in apart), []))


def test_mismatched_coordinates_are_refused_rather_than_drawn_wrong():
    with pytest.raises(ValueError):
        isolines.contour_lines(np.arange(4.0), np.arange(4.0), np.zeros((4, 5)), [0.5])


def test_a_field_with_no_cells_draws_nothing():
    for shape in ((1, 5), (5, 1), (1, 1)):
        xs, _ = isolines.contour_lines(np.arange(float(shape[1])),
                                       np.arange(float(shape[0])),
                                       np.zeros(shape), [0.0])
        assert xs.size == 0


# ---- choosing the levels ----------------------------------------------------------------
def test_levels_sit_on_the_anchor_not_on_the_frame_minimum():
    """Scrubbing time must not slide every line across the map as the range breathes."""
    warm, _ = isolines.levels_for(18.4, 23.9, 1.0)
    cool, _ = isolines.levels_for(15.2, 20.7, 1.0)
    assert list(warm) == [19.0, 20.0, 21.0, 22.0, 23.0]
    assert list(cool) == [16.0, 17.0, 18.0, 19.0, 20.0]
    shared = set(np.round(warm, 9)) & set(np.round(cool, 9))
    assert shared == {19.0, 20.0}                  # the same lines where they overlap


def test_an_anchor_puts_the_lines_on_whole_degrees_celsius_in_any_unit():
    kelvin, _ = isolines.levels_for(288.0, 292.0, 1.0, isolines.ZERO_CELSIUS_K)
    assert np.allclose(kelvin, [288.15, 289.15, 290.15, 291.15])
    # i.e. exactly the 15, 16, 17, 18 degC isotherms, which is what was asked for
    assert np.allclose(np.asarray(kelvin) - isolines.ZERO_CELSIUS_K, [15.0, 16.0, 17.0,
                                                                      18.0])


def test_levels_include_one_sitting_exactly_on_the_frame_edge():
    levels, _ = isolines.levels_for(0.0, 2.0, 0.5)
    assert list(levels) == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_a_frame_narrower_than_one_interval_can_hold_no_line():
    levels, _ = isolines.levels_for(20.2, 20.7, 1.0)
    assert levels.size == 0


def test_too_many_lines_coarsens_the_step_by_a_whole_factor():
    """A hatch pattern is not a reading. The coarser lines must be a SUBSET of the finer
    ones, so the map thins out rather than showing a different set of isotherms."""
    fine, fine_step = isolines.levels_for(0.0, 300.0, 1.0, cap=0)
    coarse, step = isolines.levels_for(0.0, 300.0, 1.0, cap=60)
    assert fine_step == 1.0 and fine.size == 301
    assert step == 6.0 and coarse.size <= 60
    assert set(np.round(coarse, 6)) <= set(np.round(fine, 6))


def test_a_broken_step_draws_nothing_rather_than_hanging():
    for step in (0.0, np.nan, np.inf):
        levels, _ = isolines.levels_for(0.0, 10.0, step)
        assert levels.size == 0
    assert isolines.levels_for(np.nan, np.nan, 1.0)[0].size == 0
    assert isolines.levels_for(0.0, 10.0, 1.0, np.nan)[0].size == 0
    # A spacing has no sign, so a negative one is its magnitude, not an empty map.
    assert list(isolines.levels_for(0.0, 3.0, -1.0)[0]) == [0.0, 1.0, 2.0, 3.0]


def test_the_heavy_lines_fall_on_round_numbers():
    levels, step = isolines.levels_for(0.0, 10.0, 1.0)
    ordinary, heavy = isolines.split_emphasis(levels, step, 5)
    assert list(heavy) == [0.0, 5.0, 10.0]
    assert list(ordinary) == [1, 2, 3, 4, 6, 7, 8, 9]

    levels, step = isolines.levels_for(0.0, 3.0, 0.5)
    _ordinary, heavy = isolines.split_emphasis(levels, step, 2)
    assert list(heavy) == [0.0, 1.0, 2.0, 3.0]          # every whole degree on a T-Td map


def test_contour_set_covers_every_level_across_its_two_layers():
    rng = np.random.default_rng(2)
    x, y = np.linspace(0, 4, 21), np.linspace(0, 3, 17)
    z = rng.normal(0, 3, (y.size, x.size))
    drawn = isolines.contour_set(x, y, z, isolines.Interval(1.0, 5))
    both = np.concatenate([drawn['ordinary'][0], drawn['emphasised'][0]])
    single = isolines.contour_lines(x, y, z, drawn['levels'])[0]
    assert both.size == single.size
    assert drawn['ordinary'][0].size and drawn['emphasised'][0].size


def test_contour_set_of_an_empty_or_unset_field_is_blank_not_an_error():
    x, y = np.linspace(0, 1, 4), np.linspace(0, 1, 4)
    blank = isolines.contour_set(x, y, np.full((4, 4), np.nan), isolines.Interval(1.0))
    assert blank['ordinary'][0].size == 0 and blank['levels'].size == 0
    assert isolines.contour_set(x, y, np.zeros((4, 4)), None)['levels'].size == 0


# ---- the interval policy ---------------------------------------------------------------
CELSIUS, KELVIN, FAHRENHEIT = (Affine('°C', 1.0, -273.15), Affine('K'),
                               Affine('°F', 1.8, -459.67))


def test_a_spacing_scales_but_an_anchor_converts_wholly():
    interval = isolines.Interval(1.0, 5, isolines.ZERO_CELSIUS_K)
    assert interval.scaled(CELSIUS) == isolines.Interval(1.0, 5, 0.0)
    assert interval.scaled(KELVIN) == isolines.Interval(1.0, 5, 273.15)
    fahrenheit = interval.scaled(FAHRENHEIT)
    assert fahrenheit.step == pytest.approx(1.8)
    assert fahrenheit.anchor == pytest.approx(32.0)


def test_the_same_lines_come_out_in_every_unit():
    """G15's whole point: 1 degC of spacing is 1.8 degF, and the 20 degC line is the
    68 degF line -- not some new line anchored on whole Fahrenheit."""
    interval = isolines.Interval(1.0, 5, isolines.ZERO_CELSIUS_K)
    celsius, _ = isolines.levels_for(18.0, 22.0, *interval.scaled(CELSIUS)[::2])
    fahrenheit, _ = isolines.levels_for(64.4, 71.6, *interval.scaled(FAHRENHEIT)[::2])
    assert np.allclose(celsius * 1.8 + 32.0, fahrenheit)


def test_a_difference_takes_the_scale_for_the_anchor_too():
    """Zero difference is zero in every unit, so the offset must not reach the anchor."""
    scaled = isolines.DIFFERENCE.scaled(CELSIUS, difference=True)
    assert (scaled.step, scaled.anchor) == (0.5, 0.0)
    scaled = isolines.DIFFERENCE.scaled(FAHRENHEIT, difference=True)
    assert scaled.step == pytest.approx(0.9) and scaled.anchor == 0.0


def test_only_the_temperatures_are_contoured():
    assert isolines.interval_for('T_2M').step == 1.0
    assert isolines.interval_for('T_S').step == 1.0
    assert isolines.interval_for('TD_2M').step == 1.0
    for field in ('CAPE_ML', 'TOT_PREC', 'RELHUM_2M', 'U_10M', 'WSPD_10M', 'CLCT'):
        assert isolines.interval_for(field) is None


# ---- what the views offer ---------------------------------------------------------------
@pytest.fixture
def pair(tmp_path):
    temperature, humidity = synth.pair(tmp_path)
    return (FieldView(EnsembleFile(temperature)), FieldView(EnsembleFile(humidity)))


def test_a_temperature_field_is_contoured_every_degree(pair):
    temperature, _humidity = pair
    assert temperature.units == '°C'
    assert temperature.isolines == isolines.Interval(1.0, 5, 0.0)
    temperature.set_units('°F')
    assert temperature.isolines.step == pytest.approx(1.8)
    assert temperature.isolines.anchor == pytest.approx(32.0)


def test_a_field_with_no_interval_offers_none(pair):
    _temperature, humidity = pair
    assert humidity.isolines is None


def test_the_dew_point_is_contoured_like_the_temperature_it_is_read_against(pair):
    view = derived.dew_point(*pair)
    assert view.isolines == isolines.Interval(1.0, 5, 0.0)


def test_the_depression_is_contoured_twice_as_finely(pair):
    view = derived.dew_point_depression(*pair)
    assert view.isolines == isolines.Interval(0.5, 2, 0.0)
    view.set_units('°F')
    assert view.isolines.step == pytest.approx(0.9)
    assert view.isolines.anchor == 0.0             # a difference of nothing is nothing


def test_a_difference_of_two_uncontoured_fields_has_no_interval(tmp_path):
    a = FieldView(EnsembleFile(synth.cloud(tmp_path / 'ICON_ENS_2026082300_CLCT.nc')))
    b = FieldView(EnsembleFile(synth.cloud(tmp_path / 'ICON_ENS_2026082300_CLCL.nc')))
    b.raw.field = 'CLCL'                            # a second cloud field to subtract
    assert derived.difference(a, b).isolines is None


def test_the_wind_map_is_not_contoured(tmp_path):
    u, v = synth.wind_pair(tmp_path)
    view = derived.wind(FieldView(EnsembleFile(u)), FieldView(EnsembleFile(v)))
    assert view.isolines is None
    assert view.sort_scale is None


# ---- the sort band ----------------------------------------------------------------------
def test_only_the_depression_has_a_sort_band(pair):
    temperature, humidity = pair
    assert getattr(temperature, 'sort_scale', None) is None
    assert derived.dew_point(temperature, humidity).sort_scale is None
    assert derived.dew_point_depression(temperature, humidity).sort_scale is not None


def test_the_band_is_red_at_zero_and_white_at_two(pair):
    scale = derived.dew_point_depression(*pair).sort_scale
    assert scale.levels == (0.0, 2.0) and scale.top == 2.0
    positions, colours = scale.positions()
    assert positions == [0.0, 0.5, 1.0]
    assert colours == ['#d7191c', '#fdae61', '#ffffff']
    assert scale.described() == [(0.0, 'red'), (1.0, 'yellow-orange'), (2.0, 'white')]


def test_the_band_is_two_degrees_celsius_whatever_the_units_say(pair):
    view = derived.dew_point_depression(*pair)
    assert view.sort_scale.levels == (0.0, 2.0)
    view.set_units('K')
    assert view.sort_scale.levels == (0.0, 2.0)         # 2 degC of difference is 2 K
    view.set_units('°F')
    assert view.sort_scale.levels[1] == pytest.approx(3.6)
    assert view.sort_scale.described()[1][0] == pytest.approx(1.8)


def test_a_difference_that_is_not_a_depression_gets_no_band(tmp_path):
    """T_2M - T_S is a temperature difference and IS contoured, but it is signed: red at
    zero and white above 2 would hide which side of zero a cell is on."""
    a = FieldView(EnsembleFile(synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')))
    surface = synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_S.nc')
    b = FieldView(EnsembleFile(surface))
    b.raw.field = 'T_S'
    view = derived.difference(a, b)
    assert view.isolines == isolines.Interval(0.5, 2, 0.0)
    assert view.sort_scale is None
    assert view.display_name == 'T_2M-T_S'


def test_a_depression_built_from_a_saved_dew_point_file_is_the_same_map(tmp_path):
    """`--difference T_2M TD_2M` against a TD_2M written by "Save field..." is the same
    quantity as `--derive depression`, so it gets the same name, interval and band."""
    temperature, humidity = synth.pair(tmp_path)
    computed = derived.dew_point(FieldView(EnsembleFile(temperature)),
                                 FieldView(EnsembleFile(humidity)))
    written = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', computed)

    view = derived.difference(FieldView(EnsembleFile(temperature)),
                              FieldView(EnsembleFile(written)))
    assert view.display_name == derived.DEPRESSION_NAME
    assert view.long_name == 'dew point depression (T - Td)'
    assert view.isolines == isolines.Interval(0.5, 2, 0.0)
    assert view.sort_scale.levels == (0.0, 2.0)
    # and it really is the same numbers as the live computation
    live = derived.dew_point_depression(FieldView(EnsembleFile(temperature)),
                                        FieldView(EnsembleFile(humidity)))
    assert np.allclose(view.agg_frame(1, 'mean'), live.agg_frame(1, 'mean'), atol=1e-5)
