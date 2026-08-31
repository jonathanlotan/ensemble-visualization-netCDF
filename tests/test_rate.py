"""V2.3 / G14, G15, G23, G24: de-accumulation.

Cumulative precipitation cannot answer "will it rain at 14Z". Turning it into an hourly
rate is only useful if the arithmetic is right, and the two accumulation kinds need
DIFFERENT arithmetic -- applying the sum formula to radiation yields numbers that look like
plausible W m-2 and are not.
"""
import numpy as np
import pytest

import synth
from imsicon import transform, products
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView


def view_of(path, **kw):
    return FieldView(EnsembleFile(path), **kw)


# ---- the sum kind: TOT_PREC ------------------------------------------------------------
@pytest.fixture
def precip(tmp_path):
    path, hourly = synth.accumulated_precip(tmp_path / 'ICON_ENS_2026082300_TOT_PREC.nc')
    return view_of(path), hourly


def test_cumsum_of_the_hourly_rate_reproduces_the_stored_accumulation(precip):
    """The round-trip that proves nothing was invented or lost."""
    view, hourly = precip
    view.set_rate(1)
    rates = view.series(1, 1)[:, 0]
    assert np.isnan(rates[0])                       # no window before the first step
    stored = EnsembleFile(view.path).series(1, 1)[:, 0]
    assert np.allclose(np.nancumsum(np.nan_to_num(rates)), stored, atol=1e-5)
    assert np.allclose(rates[1:], hourly[1:], atol=1e-5)


def test_the_three_hourly_rate_is_exactly_a_minus_a_minus_three(precip):
    view, _hourly = precip
    stored = EnsembleFile(view.path).series(2, 2)[:, 0].astype(np.float64)
    view.set_rate(3)
    rates = view.series(2, 2)[:, 0]
    for t in range(3, len(stored)):
        assert rates[t] == pytest.approx(stored[t] - stored[t - 3], abs=1e-5)


def test_the_first_k_steps_are_nan_not_a_partial_window(precip):
    view, _ = precip
    for hours in (1, 3):
        view.set_rate(hours)
        k = view.window_steps
        series = view.series(0, 0)
        assert np.all(np.isnan(series[:k]))
        assert np.all(np.isfinite(series[k:]))
        assert np.all(np.isnan(view.frame(k - 1, 0)))
        assert np.all(np.isfinite(view.frame(k, 0)))


def test_frames_and_series_agree_in_rate_mode(precip):
    """The four consumers must not be able to disagree -- one transform, one answer."""
    view, _ = precip
    view.set_rate(1)
    series = view.series(1, 2)
    for t in (1, 3, 5):
        assert np.allclose(view.ens_frame(t)[:, 1, 2], series[t], atol=1e-5, equal_nan=True)
        assert view.frame(t, 0)[1, 2] == pytest.approx(series[t, 0], abs=1e-5)


def test_units_label_follows_the_window(precip):
    view, _ = precip
    assert view.units == 'mm'
    view.set_rate(1)
    assert view.units == 'mm h-1'
    view.set_rate(3)
    assert view.units == 'mm/3h'
    view.set_units('kg m-2')
    assert view.units == 'kg m-2/3h'


def test_the_label_names_the_backward_looking_window(precip):
    view, _ = precip
    view.set_rate(3)
    assert '3 h to' in view.label_for(4)
    assert 'no 3 h window yet' in view.label_for(1)


# ---- the mean kind: radiation (this is the one np.diff gets wrong) ---------------------
@pytest.fixture
def radiation(tmp_path):
    path, hourly = synth.averaged_radiation(tmp_path / 'ICON_ENS_2026082300_ASWDIR_S.nc')
    return view_of(path), hourly


def test_the_mean_kind_formula_recovers_the_known_hourly_signal(radiation):
    """G14: A[t] is a running MEAN, so the window value is
    (A[t]h[t] - A[t-k]h[t-k]) / (h[t] - h[t-k]), never a plain difference."""
    view, hourly = radiation
    view.set_rate(1)
    rates = view.series(1, 1)[:, 0]
    assert np.allclose(rates[1:], hourly[1:], atol=1e-3)


def test_np_diff_would_have_given_a_different_and_wrong_answer(radiation):
    """The regression guard: if someone 'simplifies' the mean branch to np.diff."""
    view, hourly = radiation
    stored = EnsembleFile(view.path).series(1, 1)[:, 0].astype(np.float64)
    view.set_rate(1)
    rates = view.series(1, 1)[:, 0]
    naive = np.diff(stored)
    assert not np.allclose(naive, hourly[1:], atol=1.0)       # np.diff is wrong here
    assert np.allclose(rates[1:], hourly[1:], atol=1e-3)      # and we do not use it


def test_radiation_stays_in_w_m2_because_a_window_mean_is_still_a_mean(radiation):
    view, _ = radiation
    assert view.units == 'W m-2'
    view.set_rate(1)
    assert view.units == 'W m-2'
    view.set_rate(3)
    assert view.units == 'W m-2'


def test_hourly_means_land_in_the_physically_plausible_band(radiation):
    """The sanity check that catches an np.diff applied to an averaged field."""
    view, _ = radiation
    view.set_rate(1)
    values = view.series(2, 2)[:, 0]
    good = values[np.isfinite(values)]
    assert good.min() >= 0.0
    assert good.max() <= 1100.0            # downward SW at the surface cannot exceed this


def test_the_three_hourly_mean_averages_the_hourly_means(radiation):
    view, hourly = radiation
    view.set_rate(3)
    rates = view.series(0, 0)[:, 0]
    assert rates[3] == pytest.approx(np.mean(hourly[1:4]), abs=1e-3)


# ---- what must never be differenced -----------------------------------------------------
def test_vmax_is_refused_a_rate(tmp_path):
    """It is already a max over the previous hour (CLAUDE.md 0.2)."""
    path = synth.write_nc3(tmp_path / 'v.nc', 'VMAX_10M', 'm s-1',
                           np.random.default_rng(1).random(synth.SHAPE) * 20)
    view = view_of(path)
    assert not view.can_rate
    assert view.rate_choices == ['as stored']
    assert view.set_rate(1) is False
    assert view.window_steps == 0
    assert view.units == 'm s-1'


def test_instantaneous_fields_are_refused_a_rate(tmp_path):
    view = view_of(synth.temperature(tmp_path / 't.nc'))
    assert not view.can_rate and view.set_rate(3) is False


def test_only_the_three_documented_ensemble_fields_accumulate():
    """Scoped to the ensemble catalogue: v7 added the deterministic run's own accumulated
    fields (tests/test_pressure_levels.py), and this test is about the 15 measured ones."""
    ensemble = {field: kind for field, kind in transform.ACCUMULATION.items()
                if field in products.ENSEMBLE.by_field}
    assert ensemble == {'TOT_PREC': 'sum', 'ASWDIFD_S': 'mean', 'ASWDIR_S': 'mean'}
    assert 'VMAX_10M' not in transform.ACCUMULATION


# ---- ordering, precision, noise ---------------------------------------------------------
def test_a_sum_rate_takes_the_scale_but_not_the_offset(tmp_path):
    """G15 end to end: the ordering rule enforced by the transform, not by call sites."""
    path, hourly = synth.accumulated_precip(tmp_path / 'p.nc')
    view = view_of(path)
    view.set_rate(1)
    view.set_units('mm')                                  # identity: (1, 0)
    mm = view.series(0, 0)[:, 0]
    assert np.allclose(mm[1:], hourly[1:], atol=1e-5)
    # A hypothetical offset unit must not shift a difference.
    view._units = transform.Affine('mm+1000', 1.0, 1000.0)
    shifted = view.series(0, 0)[:, 0]
    assert np.allclose(shifted[1:], mm[1:], atol=1e-5)    # unchanged: offset dropped
    view._units = transform.Affine('tenths', 10.0, 1000.0)
    scaled = view.series(0, 0)[:, 0]
    assert np.allclose(scaled[1:], mm[1:] * 10.0, atol=1e-4)


def test_a_mean_rate_takes_the_full_affine(tmp_path):
    """The asymmetry with the sum branch: a window mean is ABSOLUTE."""
    path, _ = synth.averaged_radiation(tmp_path / 'r.nc')
    view = view_of(path)
    view.set_rate(1)
    base = view.series(0, 0)[:, 0]
    view._units = transform.Affine('offset', 1.0, 50.0)
    assert np.allclose(view.series(0, 0)[1:, 0], base[1:] + 50.0, atol=1e-3)


def test_the_subtraction_happens_in_float64(tmp_path):
    """G23: catastrophic cancellation between two large near-equal accumulations."""
    big = 3.0e5
    hourly = np.array([0.0, 0.05, 0.05, 0.05, 0.05, 0.05])
    stored = big + np.cumsum(hourly)
    values, is_diff, _ = transform.window_value(stored[3], stored[2], 3.0, 2.0, 'sum')
    assert is_diff
    assert np.asarray(values).dtype == np.float64
    assert float(values) == pytest.approx(0.05, rel=1e-6)
    # the same subtraction done in float32 loses most of the signal
    f32 = np.float32(stored[3]) - np.float32(stored[2])
    assert abs(float(f32) - 0.05) > abs(float(values) - 0.05)


def test_float_noise_is_clamped_but_a_real_negative_warns():
    """G24: clamping everything would hide G14; clamping nothing would show fake drizzle."""
    a = np.array([100.0, 100.0])
    tiny = a[0] - transform.noise_floor(100.0) / 2.0
    values, _, negatives = transform.window_value(np.array([tiny]), np.array([a[0]]),
                                                  1.0, 0.0, 'sum')
    assert values[0] == 0.0 and negatives == 0            # noise: clamped silently

    values, _, negatives = transform.window_value(np.array([90.0]), np.array([100.0]),
                                                  1.0, 0.0, 'sum')
    assert values[0] == pytest.approx(-10.0) and negatives == 1   # real: surfaced


def test_a_real_negative_reaches_the_user_as_a_note(tmp_path):
    falling = np.linspace(50.0, 0.0, 6)[:, None, None] * np.ones((1, 4, 5))
    path = synth.write_nc3(tmp_path / 'bad.nc', 'TOT_PREC', 'kg m-2',
                           np.stack([falling] * 3, axis=1))
    view = view_of(path)
    view.set_rate(1)
    view.series(0, 0)
    assert view.rate_note and 'decreases' in view.rate_note


# ---- the window is measured in steps, not assumed to be hours --------------------------
def test_window_steps_uses_the_actual_time_spacing():
    hourly = np.arange(0, 121, 1.0)
    assert transform.window_steps(1, hourly) == 1
    assert transform.window_steps(3, hourly) == 3
    three_hourly = np.arange(0, 121, 3.0)
    assert transform.window_steps(3, three_hourly) == 1     # 1 step IS 3 hours here
    assert transform.window_steps(1, three_hourly) == 1     # never below one step
    half_hourly = np.arange(0, 24, 0.5)
    assert transform.window_steps(3, half_hourly) == 6


def test_the_signature_separates_rate_views_but_not_unit_views(tmp_path):
    """G19: what a cached range may and may not be reused for."""
    path, _ = synth.accumulated_precip(tmp_path / 'p.nc')
    view = view_of(path)
    assert view.transform_signature == 'raw'
    view.set_units('kg m-2')
    assert view.transform_signature == 'raw'                # units never force a rescan
    view.set_rate(1)
    assert view.transform_signature == 'rate:sum:1'
    view.set_rate(3)
    assert view.transform_signature == 'rate:sum:3'
