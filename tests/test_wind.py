"""The wind map: speed for the colours, vectors for the barbs, and the G16 line between.

`test_barbs.py` covers the glyph. This covers the view that feeds it -- which member's
wind each aggregation draws, what the units do, and the two refusals (**G17** member
order, and a component whose units are not what the formula expects).
"""
import numpy as np
import pytest

import synth
from imsicon import barbs, derived, ncwrite, transform
from imsicon.dataset import EnsembleFile

KT = barbs.KT_PER_MS


@pytest.fixture
def wind(tmp_path):
    synth.wind_pair(tmp_path)
    return build(tmp_path)


def build(directory, run='2026082300'):
    return derived.wind(
        derived.open_field(directory / f'ICON_ENS_{run}_U_10M.nc'),
        derived.open_field(directory / f'ICON_ENS_{run}_V_10M.nc'))


def components(view, t=0):
    return (view.u.raw.ens_frame(t), view.v.raw.ens_frame(t))


# ---- what the view is ------------------------------------------------------------------
def test_the_wind_map_is_the_speed_of_the_two_components(wind):
    u, v = components(wind)
    assert np.allclose(wind.ens_frame(0), np.hypot(u, v), atol=1e-5)
    assert wind.field == 'WSPD_10M'
    assert wind.display_name == derived.WIND_NAME
    assert wind.units == 'm s-1'


def test_the_components_are_found_in_either_order(tmp_path):
    synth.wind_pair(tmp_path)
    u = derived.open_field(tmp_path / 'ICON_ENS_2026082300_U_10M.nc')
    v = derived.open_field(tmp_path / 'ICON_ENS_2026082300_V_10M.nc')
    assert derived.wind(v, u).u.field == 'U_10M'


def test_the_graph_series_is_the_speed_at_that_point(wind):
    series = wind.series(2, 3)
    assert series.shape == (wind.n_times, wind.n_members)
    u, v = wind.u.raw.series(2, 3), wind.v.raw.series(2, 3)
    assert np.allclose(series, np.hypot(u, v), atol=1e-5)


def test_it_is_neither_a_difference_nor_a_rate(wind):
    """A wind speed is an instantaneous, non-negative quantity: no diverging map, and the
    Rate control has nothing to offer it."""
    assert wind.is_difference_view is False
    assert wind.diverging is False
    assert wind.can_rate is False
    assert wind.set_rate(0) and not wind.set_rate(1)


# ---- units ------------------------------------------------------------------------------
def test_the_speed_takes_the_wind_units_the_registry_offers(wind):
    assert wind.unit_labels == ['m s-1', 'kt', 'km h-1']
    reference = wind.ens_frame(0).copy()
    assert wind.set_units('kt')
    assert np.allclose(wind.ens_frame(0), reference * KT, rtol=1e-6)
    assert wind.units == 'kt'
    assert wind.set_units('km h-1')
    assert np.allclose(wind.ens_frame(0), reference * 3.6, rtol=1e-6)


def test_a_unit_change_rescales_the_cached_range_instead_of_rescanning(wind):
    lo, hi = wind.scan_range()
    wind.set_units('kt')
    assert wind.value_range == pytest.approx((lo * KT, hi * KT), rel=1e-6)


def test_the_barbs_stay_in_knots_whatever_the_colour_scale_says(wind):
    """A half feather means 5 kt by definition. It cannot mean 5 of whatever the units
    combo happens to be set to."""
    in_ms = wind.wind_vectors(0, 'mean')
    wind.set_units('km h-1')
    assert np.allclose(wind.wind_vectors(0, 'mean'), in_ms)
    assert wind.barb_units == 'kt'
    u, _v = in_ms
    assert np.allclose(u, wind.u.raw.ens_frame(0).mean(axis=0) * KT, rtol=1e-5)


# ---- which wind the barbs show ----------------------------------------------------------
def test_a_member_map_draws_that_member_s_own_vector(wind):
    u, v = components(wind)
    drawn = wind.wind_vectors(0, 'member', member=2)
    assert np.allclose(drawn[0], u[2] * KT, rtol=1e-5)
    assert np.allclose(drawn[1], v[2] * KT, rtol=1e-5)


def test_the_mean_map_draws_the_mean_vector_not_the_mean_of_the_directions(tmp_path):
    """**G16**: two members at 350 and 10 degrees average to 360, not to 180. A linear
    mean of the degrees is the wrong answer, and it is a *plausible* wrong answer."""
    from_degrees = np.deg2rad([350.0, 10.0])       # the two members' wind directions
    for field, values in (('U_10M', -np.sin(from_degrees)),
                          ('V_10M', -np.cos(from_degrees))):
        # a wind FROM d blows TOWARDS d+180, so its vector is -(sin d, cos d)
        stack = np.asarray(values, dtype=float)[None, :, None, None] * np.ones((3, 2, 2, 2))
        synth.write_nc3(tmp_path / f'ICON_ENS_2026082300_{field}.nc', field, 'm s-1', stack,
                        history=synth.HISTORY_TEMPLATE.format(field=field))
    view = build(tmp_path)
    drawn = view.direction_at(0, 0, 0, 'mean')
    assert min(drawn, 360.0 - drawn) == pytest.approx(0.0, abs=0.01)     # due north
    assert abs(drawn - np.mean([350.0, 10.0])) > 100.0                   # not due south


def test_the_mean_barb_is_no_longer_than_the_mean_speed_under_it(wind):
    """|mean(V)| <= mean(|V|) by Jensen, and the gap is the members disagreeing about
    direction. The barb and the colour are allowed to differ; what is not allowed is
    pretending they are the same number."""
    u, v = wind.wind_vectors(0, 'mean')
    assert np.all(np.hypot(u, v) <= wind.agg_frame(0, 'mean') * KT + 1e-4)


@pytest.mark.parametrize('mode, pick', [('max', np.argmax), ('min', np.argmin)])
def test_the_extreme_maps_draw_the_vector_of_the_member_the_colour_came_from(wind, mode, pick):
    """max(u) paired with max(v) would invent a wind no member forecast; the barb has to
    be a member's own vector, chosen per cell by the same speed the colour shows."""
    u, v = components(wind)
    index = pick(np.hypot(u, v), axis=0)
    drawn_u, drawn_v = wind.wind_vectors(0, mode)
    for iy in range(wind.ny):
        for ix in range(wind.nx):
            member = index[iy, ix]
            assert drawn_u[iy, ix] == pytest.approx(u[member, iy, ix] * KT, rel=1e-5)
            assert drawn_v[iy, ix] == pytest.approx(v[member, iy, ix] * KT, rel=1e-5)
    assert np.allclose(np.hypot(drawn_u, drawn_v), wind.agg_frame(0, mode) * KT, rtol=1e-5)


def test_the_median_map_draws_a_real_member_not_an_interpolated_non_wind(wind):
    u, v = components(wind)
    drawn_u, drawn_v = wind.wind_vectors(0, 'median')
    for iy in range(wind.ny):
        for ix in range(wind.nx):
            assert np.any(np.isclose(drawn_u[iy, ix], u[:, iy, ix] * KT, rtol=1e-5)
                          & np.isclose(drawn_v[iy, ix], v[:, iy, ix] * KT, rtol=1e-5))


def test_a_spread_map_says_the_barbs_are_the_mean_vector(wind):
    """A max-minus-min has no direction of its own, so the barbs fall back to the mean --
    and the label has to say so, or the map claims something it is not showing."""
    assert np.allclose(wind.wind_vectors(0, 'spread'), wind.wind_vectors(0, 'mean'))
    assert 'mean vector' in wind.barb_label('spread')
    assert 'no direction of its own' in wind.barb_label('spread')
    for mode in ('mean', 'max', 'min', 'median', 'member'):
        assert wind.barb_units in wind.barb_label(mode)


def test_sampling_a_subgrid_gives_exactly_the_same_vectors_as_the_whole_one(wind):
    """The barbs read only the points they draw. That has to be an optimisation, not a
    different answer."""
    rows, cols = [0, 2, 3], [1, 4]
    for mode in ('mean', 'max', 'min', 'median', 'member', 'spread'):
        full = wind.wind_vectors(0, mode, member=1)
        part = wind.wind_vectors(0, mode, member=1, rows=rows, cols=cols)
        assert np.allclose(part[0], full[0][np.ix_(rows, cols)], rtol=1e-6)
        assert np.allclose(part[1], full[1][np.ix_(rows, cols)], rtol=1e-6)


def test_the_direction_at_a_point_is_read_off_the_aggregated_vector(wind):
    u, v = wind.wind_vectors(0, 'mean')
    assert wind.direction_at(0, 1, 2, 'mean') == pytest.approx(
        float(barbs.direction_from(u[1, 2], v[1, 2])))


# ---- the two refusals -------------------------------------------------------------------
def test_components_from_differently_ordered_files_are_refused(tmp_path):
    """**G17**: member identity is positional. Pairing u[i] with a different member's v[i]
    gives a wind that never existed and looks completely ordinary."""
    synth.wind_pair(tmp_path)
    scrambled = tmp_path / 'ICON_ENS_2026082300_V_10M.nc'
    history = synth.HISTORY_TEMPLATE.format(field='V_10M').replace('_01_', '_09_')
    synth.write_nc3(scrambled, 'V_10M', 'm s-1', np.zeros((6, 3, 4, 5)), history=history)
    with pytest.raises(derived.PairError, match='Member order differs'):
        build(tmp_path)


def test_a_component_in_the_wrong_units_is_refused_rather_than_converted(tmp_path):
    synth.wind_pair(tmp_path, units='km h-1')
    with pytest.raises(derived.PairError, match="needs 'm s-1'"):
        build(tmp_path)


def test_the_wrong_field_is_refused_with_a_message_that_says_which(tmp_path):
    synth.pair(tmp_path)
    synth.wind_pair(tmp_path)
    with pytest.raises(derived.PairError, match='must be U_10M'):
        derived.wind(derived.open_field(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'),
                     derived.open_field(tmp_path / 'ICON_ENS_2026082300_V_10M.nc'))


def test_components_of_two_different_runs_are_refused(tmp_path):
    """Aligning two runs needs valid time rather than forecast hour, which this build
    does not do (v3 R3.6). The refusal is `check_pairable`'s, and it applies here too."""
    synth.wind_pair(tmp_path)
    synth.write_nc3(tmp_path / 'yesterday_V_10M.nc', 'V_10M', 'm s-1',
                    np.zeros((6, 3, 4, 5)),
                    history=synth.HISTORY_TEMPLATE.format(field='V_10M'),
                    time_units='minutes since 2026-8-22 00:00:00')
    with pytest.raises(derived.PairError, match='run'):
        derived.wind(derived.open_field(tmp_path / 'ICON_ENS_2026082300_U_10M.nc'),
                     derived.open_field(tmp_path / 'yesterday_V_10M.nc'))


# ---- saving -----------------------------------------------------------------------------
def test_the_speed_is_written_in_m_s_whatever_the_screen_shows(wind, tmp_path):
    """The same rule the dew point follows: a file gets the view's canonical units, so it
    reopens with the registry's ordinary treatment rather than a stale-units warning."""
    wind.set_units('kt')
    written = ncwrite.write_canonical(tmp_path / 'WSPD_saved.nc', wind)
    assert wind.canonical_units == 'm s-1'
    reopened = EnsembleFile(written)
    assert reopened.field == 'WSPD_10M' and reopened.units == 'm s-1'
    assert np.array_equal(reopened.ens_frame(0), wind.canonical_ens_frame(0))
    assert list(reopened.member_labels) == list(wind.member_labels)      # G1
    assert transform.choices_for('WSPD_10M', reopened.units)[0][0].label == 'm s-1'


def test_a_written_speed_field_has_no_barbs_and_does_not_pretend_to(wind, tmp_path):
    """A speed file no longer knows which way the wind was blowing, and the UI decides
    whether to draw barbs by asking the view for vectors."""
    written = ncwrite.write_canonical(tmp_path / 'WSPD.nc', wind)
    assert getattr(derived.open_field(written), 'wind_vectors', None) is None
    assert getattr(wind, 'wind_vectors', None) is not None


def test_the_provenance_names_both_files_it_came_from(wind):
    assert sorted(wind.source_files) == ['ICON_ENS_2026082300_U_10M.nc',
                                         'ICON_ENS_2026082300_V_10M.nc']
    assert 'U_10M' in wind.provenance and 'V_10M' in wind.provenance
