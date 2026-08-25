"""Dew point and difference views: the physics, the guards, and the arithmetic.

The physics is checked against published dew point values rather than against itself, and
every refusal is checked for the *message* as well as the exception -- a guard whose
message does not say which file failed is a guard the user cannot act on.
"""
import numpy as np
import pytest

import synth
from imsicon import derived, ncwrite, transform


@pytest.fixture
def pair(tmp_path):
    t_path, rh_path = synth.pair(tmp_path)
    return derived.open_field(t_path), derived.open_field(rh_path)


@pytest.fixture
def dewpoint(pair):
    return derived.dew_point(*pair)


# ---- the formula -------------------------------------------------------------------
# Published Magnus/Alduchov-Eskridge values; the formula is stated to 0.1 degC, so that is
# the tolerance. Checking against a hand-rolled reimplementation would prove nothing.
@pytest.mark.parametrize('t_c, rh, expected', [
    (20.0, 50.0, 9.3),
    (30.0, 60.0, 21.4),
    (-10.0, 80.0, -12.8),
    (35.0, 20.0, 8.7),
    (5.0, 90.0, 3.5),
])
def test_dew_point_matches_published_values(t_c, rh, expected):
    got = float(derived.dew_point_celsius(np.float32(t_c), np.float32(rh)))
    assert got == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize('t_c', [-20.0, 0.0, 15.0, 40.0])
def test_saturated_air_has_its_dew_point_at_the_temperature(t_c):
    """RH = 100 % collapses the formula to Td = T exactly, not approximately."""
    got = float(derived.dew_point_celsius(np.float32(t_c), np.float32(100.0)))
    assert got == pytest.approx(t_c, abs=1e-4)


def test_dew_point_never_exceeds_the_temperature():
    """Td > T is thermodynamically impossible, so it must be impossible here too."""
    rng = np.random.default_rng(3)
    t = rng.uniform(-35.0, 48.0, 50_000).astype(np.float32)
    rh = rng.uniform(0.5, 100.0, 50_000).astype(np.float32)
    td = derived.dew_point_celsius(t, rh)
    assert np.all(td <= t + 1e-4)
    assert np.isfinite(td).all()


def test_supersaturation_is_clamped_rather_than_producing_td_above_t():
    td = derived.dew_point_celsius(np.float32(18.0), np.float32(104.0))
    assert float(td) == pytest.approx(18.0, abs=1e-4)


def test_zero_humidity_has_no_dew_point_and_says_so_with_nan():
    """ln(0) is -inf; a huge negative number would still colour a map convincingly."""
    assert np.isnan(derived.dew_point_celsius(np.float32(20.0), np.float32(0.0)))
    assert np.isnan(derived.dew_point_celsius(np.float32(20.0), np.float32(-5.0)))


def test_a_nan_temperature_stays_nan():
    assert np.isnan(derived.dew_point_celsius(np.float32(np.nan), np.float32(50.0)))


def test_kelvin_and_celsius_forms_agree():
    t_c = np.linspace(-30, 45, 200).astype(np.float32)
    rh = np.linspace(5, 100, 200).astype(np.float32)
    assert np.allclose(derived.dew_point_kelvin(t_c + 273.15, rh),
                       derived.dew_point_celsius(t_c, rh) + 273.15, atol=1e-3)


def test_humidity_oddities_counts_both_kinds():
    assert derived.humidity_oddities(np.array([0.0, 50.0, 100.0, 100.4, -1.0])) == (1, 2)
    assert derived.humidity_oddities(np.array([10.0, 100.0])) == (0, 0)


# ---- the dew point view ------------------------------------------------------------
def test_the_dew_point_is_a_kelvin_temperature_shown_in_celsius(dewpoint):
    assert dewpoint.field == 'TD_2M'
    assert dewpoint.canonical_units == 'K'
    assert dewpoint.units == '°C' and dewpoint.unit_labels == ['°C', 'K', '°F']
    assert dewpoint.can_convert_units and not dewpoint.can_rate


def test_the_view_matches_the_formula_applied_to_the_two_files(dewpoint, pair):
    t_view, rh_view = pair
    for t in range(t_view.n_times):
        expected = derived.dew_point_celsius(
            t_view.raw.ens_frame(t) - 273.15, rh_view.raw.ens_frame(t))
        assert np.allclose(dewpoint.ens_frame(t), expected, equal_nan=True, atol=1e-4)


def test_the_dew_point_borrows_the_grid_and_time_axis_of_its_inputs(dewpoint, pair):
    t_view, _rh = pair
    for name in ('ny', 'nx', 'n_times', 'n_members', 'dlat', 'dlon', 'run_init'):
        assert getattr(dewpoint, name) == getattr(t_view, name)
    assert np.array_equal(dewpoint.lat, t_view.lat)
    assert dewpoint.member_labels == t_view.member_labels
    assert dewpoint.label_for(2) == t_view.label_for(2)


def test_switching_units_rescales_the_range_without_a_second_scan(dewpoint):
    celsius = dewpoint.scan_range()
    assert dewpoint.set_units('K')
    kelvin = dewpoint.value_range
    assert kelvin == pytest.approx(tuple(v + 273.15 for v in celsius))
    assert dewpoint.set_units('°F')
    assert dewpoint.value_range == pytest.approx(
        tuple(v * 1.8 + 32.0 for v in celsius), abs=1e-3)


def test_a_series_and_a_frame_agree_at_the_same_point(dewpoint):
    series = dewpoint.series(2, 3)
    for t in range(dewpoint.n_times):
        assert np.allclose(series[t], dewpoint.ens_frame(t)[:, 2, 3], atol=1e-4)


def test_aggregation_happens_after_conversion(dewpoint):
    """`spread` is a difference, so an affine offset must cancel (G15)."""
    dewpoint.set_units('°C')
    spread_c = dewpoint.agg_frame(3, 'spread')
    dewpoint.set_units('K')
    assert np.allclose(spread_c, dewpoint.agg_frame(3, 'spread'), atol=1e-4)
    dewpoint.set_units('°F')
    assert np.allclose(spread_c * 1.8, dewpoint.agg_frame(3, 'spread'), atol=1e-3)


def test_the_humidity_note_reports_clamped_cells_once(tmp_path):
    t_path, rh_path = synth.pair(tmp_path, humidity_encoding='supersaturated')
    view = derived.dew_point(derived.open_field(t_path), derived.open_field(rh_path))
    assert view.note and 'above 100' in view.note
    assert view.units_note == view.note      # what MainWindow shows as the units warning


def test_a_dry_cell_is_reported_and_left_blank(tmp_path):
    t_path, rh_path = synth.pair(tmp_path, humidity_encoding='dry')
    view = derived.dew_point(derived.open_field(t_path), derived.open_field(rh_path))
    assert view.note and 'below 0' in view.note
    assert np.isnan(view.ens_frame(0)[0, 0, 0])


# ---- the guards --------------------------------------------------------------------
def test_a_shuffled_member_order_is_refused(tmp_path, pair):
    """G17: pairing member 01 of T with member 09 of RH looks entirely plausible."""
    t_view, _rh = pair
    shuffled = tmp_path / 'ICON_ENS_2026082300_RELHUM_2M_shuffled.nc'
    ncwrite.write_nc3(shuffled, 'RELHUM_2M', '%',
                      np.full((6, 3, 4, 5), 50.0, dtype=np.float32),
                      history=synth.HISTORY_TEMPLATE.format(
                          field='RELHUM_2M').replace('_01_', '_09_'))
    with pytest.raises(derived.PairError) as excinfo:
        derived.dew_point(t_view, derived.open_field(shuffled))
    message = str(excinfo.value)
    assert 'Member order differs' in message and 'member 09' in message


def test_humidity_in_the_wrong_units_is_refused_rather_than_converted(tmp_path, pair):
    """A 0-1 fraction read as a percentage is ~45 degC of error and nothing on screen."""
    t_view, _rh = pair
    fraction = tmp_path / 'ICON_ENS_2026082300_RELHUM_2M_frac.nc'
    ncwrite.write_nc3(fraction, 'RELHUM_2M', '1',
                      np.full((6, 3, 4, 5), 0.5, dtype=np.float32),
                      history=synth.HISTORY_TEMPLATE.format(field='RELHUM_2M'))
    with pytest.raises(derived.PairError, match="needs '%'"):
        derived.dew_point(t_view, derived.open_field(fraction))


def test_a_different_grid_is_refused(tmp_path, pair):
    t_view, _rh = pair
    small = synth.humidity(tmp_path / 'ICON_ENS_2026082300_RELHUM_2M_small.nc', ny=3)
    with pytest.raises(derived.PairError, match='Different grids'):
        derived.dew_point(t_view, derived.open_field(small))


def test_a_different_number_of_steps_names_the_incomplete_file(tmp_path, pair):
    t_view, _rh = pair
    short = synth.humidity(tmp_path / 'ICON_ENS_2026082300_RELHUM_2M_short.nc', n_times=4)
    with pytest.raises(derived.PairError, match='forecast steps'):
        derived.dew_point(t_view, derived.open_field(short))


def test_the_temperature_input_must_be_t_2m(tmp_path, pair):
    _t, rh_view = pair
    surface = synth.write_nc3(tmp_path / 'ICON_ENS_2026082300_T_S.nc', 'T_S', 'K',
                              np.full((6, 3, 4, 5), 295.0, dtype=np.float32),
                              history=synth.HISTORY_TEMPLATE.format(field='T_S'))
    with pytest.raises(derived.PairError, match='must be T_2M'):
        derived.dew_point(derived.open_field(surface), rh_view)


def test_the_inputs_may_be_given_in_either_order(pair):
    t_view, rh_view = pair
    assert derived.dew_point(rh_view, t_view).field == 'TD_2M'


# ---- the difference view -----------------------------------------------------------
def test_the_depression_is_temperature_minus_dew_point(pair):
    t_view, rh_view = pair
    depression = derived.dew_point_depression(t_view, rh_view)
    dewpoint = derived.dew_point(t_view, rh_view)
    assert depression.field == 'T_2M-TD_2M'
    assert 'depression' in depression.long_name
    for t in (0, 3, 5):
        assert np.allclose(depression.ens_frame(t),
                           t_view.ens_frame(t) - dewpoint.ens_frame(t), atol=1e-4)


def test_the_depression_is_never_negative(pair):
    depression = derived.dew_point_depression(*pair)
    for t in range(depression.n_times):
        assert np.nanmin(depression.ens_frame(t)) >= -1e-4


def test_a_difference_is_taken_member_by_member_not_between_aggregates(pair):
    """The classic error: mean(a) - mean(b) happens to equal mean(a-b), but
    max(a) - max(b) does NOT equal max(a-b), and only the latter is the question asked."""
    t_view, rh_view = pair
    depression = derived.dew_point_depression(t_view, rh_view)
    dewpoint = derived.dew_point(t_view, rh_view)
    per_member = np.nanmax(t_view.ens_frame(2) - dewpoint.ens_frame(2), axis=0)
    assert np.allclose(depression.agg_frame(2, 'max'), per_member, atol=1e-4)


def test_a_temperature_difference_is_the_same_number_in_c_and_k(pair):
    """G15: the affine OFFSET must cancel on a difference. 5 K of drop is 5 degC of drop."""
    depression = derived.dew_point_depression(*pair)
    celsius = depression.ens_frame(2).copy()
    assert depression.set_units('K')
    assert np.allclose(celsius, depression.ens_frame(2), atol=1e-4)
    assert depression.set_units('°F')
    assert np.allclose(celsius * 1.8, depression.ens_frame(2), atol=1e-3)


def test_a_difference_range_rescales_with_the_units(pair):
    depression = derived.dew_point_depression(*pair)
    celsius = depression.scan_range()
    depression.set_units('K')
    assert depression.value_range == pytest.approx(celsius)          # scale 1, no offset
    depression.set_units('°F')
    assert depression.value_range == pytest.approx(
        tuple(v * 1.8 for v in celsius), abs=1e-3)


def test_a_difference_asks_for_a_diverging_map(pair):
    depression = derived.dew_point_depression(*pair)
    assert depression.diverging and depression.is_difference_view
    assert not derived.dew_point(*pair).diverging


def test_subtracting_unlike_quantities_is_refused(tmp_path, pair):
    t_view, _rh = pair
    cape = synth.write_nc3(tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc', 'CAPE_ML',
                           'J kg-1', np.full((6, 3, 4, 5), 900.0, dtype=np.float32),
                           history=synth.HISTORY_TEMPLATE.format(field='CAPE_ML'))
    with pytest.raises(derived.PairError, match='no meaning'):
        derived.difference(t_view, derived.open_field(cape))


def test_the_registry_pre_check_agrees_with_the_real_one():
    """The dialog greys out a pair before spending 16 s finding out it cannot work."""
    assert derived.units_look_compatible('T_2M', 'TD_2M')
    assert derived.units_look_compatible('CLCT', 'CLCL')
    assert not derived.units_look_compatible('T_2M', 'RELHUM_2M')
    assert not derived.units_look_compatible('CAPE_ML', 'TOT_PREC')
    assert derived.units_look_compatible('UNKNOWN_FIELD', 'T_2M')   # only the real check refuses


def test_td_2m_is_registered_as_a_kelvin_temperature():
    """A written dew point must reopen with the same treatment T_2M gets."""
    choices, note = transform.choices_for('TD_2M', 'K')
    assert note is None and [c.label for c in choices] == ['°C', 'K', '°F']


def test_the_summary_names_both_source_files(pair):
    summary = derived.dew_point(*pair).summary()
    assert 'TD_2M' in summary and 'T_2M.nc' in summary and 'RELHUM_2M.nc' in summary
    assert '3 members' in summary


def test_a_nested_derived_view_credits_every_source_file(pair):
    """The depression is T_2M minus a dew point, which is itself two files."""
    depression = derived.dew_point_depression(*pair)
    assert sorted(depression.source_files) == ['ICON_ENS_2026082300_RELHUM_2M.nc',
                                               'ICON_ENS_2026082300_T_2M.nc']
    summary = depression.summary()
    assert 'RELHUM_2M.nc' in summary and 'T_2M.nc' in summary


# ---- performance: the 60 fps scrub budget v2 set is a requirement, not a hope ---------
def test_a_full_size_dew_point_frame_stays_inside_the_scrub_budget():
    """MEASURED at the real 20 x 261 x 161 map size (CLAUDE.md 0.3).

    The threshold is deliberately loose -- 3x the 16.7 ms frame -- because this runs on
    whatever CI machine is free. It is here to catch a rewrite that reintroduces the
    per-guard `np.where` version, which measured 13.2 ms of kernel alone.
    """
    import time
    rng = np.random.default_rng(0)
    t = (293.15 + rng.normal(0, 4, (20, 261, 161))).astype(np.float32)
    rh = np.clip(60 + rng.normal(0, 15, (20, 261, 161)), 1, 100).astype(np.float32)
    derived.dew_point_kelvin(t, rh)                       # warm up
    start = time.perf_counter()
    for _ in range(5):
        derived.dew_point_kelvin(t, rh)
    assert (time.perf_counter() - start) / 5 < 0.050
