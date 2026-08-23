"""Domain model: coordinates, times, and the six statistics the readout shows."""
import datetime as dt

import numpy as np
import pytest

from imsicon.dataset import EnsembleFile, member_stats


@pytest.fixture(scope='module')
def ds(cape_path):
    return EnsembleFile(cape_path)


def test_shape_and_metadata(ds):
    assert (ds.n_times, ds.n_members, ds.ny, ds.nx) == (121, 20, 261, 161)
    assert ds.field == 'CAPE_ML' and ds.units == 'J kg-1'
    assert ds.run_init == dt.datetime(2026, 8, 23, tzinfo=dt.timezone.utc)


def test_times_are_hourly_out_to_120h(ds):
    assert ds.forecast_hours[0] == 0 and ds.forecast_hours[-1] == 120
    assert np.allclose(np.diff(ds.forecast_hours), 1.0)


def test_zulu_label(ds):
    assert ds.label_for(110) == '2026-08-27 14:00Z  (+110 h)'


def test_nearest_index_round_trip(ds):
    for iy, ix in ((0, 0), (130, 80), (260, 160)):
        assert ds.nearest_index(ds.lat[iy], ds.lon[ix]) == (iy, ix)


def test_nearest_index_clamps_outside_the_domain(ds):
    """F3.3: a click past the edge snaps to the edge cell instead of being dropped."""
    assert ds.nearest_index(99.0, 99.0) == (ds.ny - 1, ds.nx - 1)
    assert ds.nearest_index(-99.0, -99.0) == (0, 0)


def test_series_matches_frames(ds):
    iy, ix = 243, 112
    series = ds.series(iy, ix)
    assert series.shape == (121, 20)
    for t in (0, 60, 120):
        assert np.array_equal(series[t], ds.ens_frame(t)[:, iy, ix])


def test_aggregations_match_numpy(ds):
    stack = ds.ens_frame(48)
    assert np.allclose(ds.agg_frame(48, 'mean'), stack.mean(axis=0))
    assert np.allclose(ds.agg_frame(48, 'max'), stack.max(axis=0))
    assert np.allclose(ds.agg_frame(48, 'spread'), stack.max(axis=0) - stack.min(axis=0))


def test_member_stats_are_the_six_readout_numbers():
    values = np.arange(20, dtype=float)
    stats = member_stats(values)
    assert stats['mean'] == 9.5 and stats['min'] == 0 and stats['max'] == 19
    assert stats['p10'] == pytest.approx(np.percentile(values, 10))
    assert stats['p90'] == pytest.approx(np.percentile(values, 90))
    assert stats['n'] == 20


def test_member_stats_ignore_nan():
    stats = member_stats([1.0, np.nan, 3.0])
    assert stats['n'] == 2 and stats['mean'] == 2.0


def test_member_stats_all_nan_is_not_a_crash():
    assert member_stats([np.nan, np.nan])['n'] == 0


def test_known_hotspot_values(ds):
    """The proof-of-concept point: 34.075N 35.800E at +110 h."""
    iy, ix = ds.nearest_index(34.075, 35.800)
    stats = member_stats(ds.series(iy, ix)[110])
    assert stats['max'] == pytest.approx(3252.16, abs=0.01)
    assert stats['mean'] == pytest.approx(1400.6, abs=0.1)
    assert stats['p90'] == pytest.approx(2076.8, abs=0.1)


def test_stats_cache_round_trip(ds, tmp_path):
    lo, hi = ds.scan_range()
    assert (lo, hi) == pytest.approx((0.0, 3252.156), abs=0.01)
    assert ds.cached_range() == (lo, hi)
