"""V2.2: FieldView must be transparent -- same surface, transformed values.

The raw layer stays pure, which is what lets these tests round-trip against it.
"""
import numpy as np
import pytest

import synth
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView


@pytest.fixture
def kelvin(tmp_path):
    path = synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    return EnsembleFile(path), FieldView(EnsembleFile(path))


def test_pass_through_attributes_are_the_raw_ones(kelvin):
    raw, view = kelvin
    for name in ('field', 'ny', 'nx', 'n_times', 'n_members', 'dlat', 'dlon',
                 'run_init', 'member_labels', 'forecast_hours', 'times', 'path'):
        assert np.all(getattr(view, name) == getattr(raw, name)), name
    assert view.nearest_index(28.05, 33.05) == raw.nearest_index(28.05, 33.05)
    assert view.extent() == raw.extent()


def test_missing_attributes_still_raise_attributeerror(kelvin):
    _raw, view = kelvin
    with pytest.raises(AttributeError):
        view.no_such_attribute


def test_every_read_path_converts_consistently(kelvin):
    """The four consumers cannot disagree, because there is one place values are made."""
    raw, view = kelvin
    assert view.units == '°C'
    assert np.allclose(view.frame(1, 0), raw.frame(1, 0) - 273.15, atol=1e-4)
    assert np.allclose(view.ens_frame(1), raw.ens_frame(1) - 273.15, atol=1e-4)
    assert np.allclose(view.series(2, 3), raw.series(2, 3) - 273.15, atol=1e-4)
    assert np.allclose(view.agg_frame(1, 'mean'), raw.agg_frame(1, 'mean') - 273.15,
                       atol=1e-4)


def test_spread_is_a_difference_and_loses_the_offset(kelvin):
    """G15 in the aggregation path: a spread of 5 K is a spread of 5 degC."""
    raw, view = kelvin
    assert np.allclose(view.agg_frame(3, 'spread'), raw.agg_frame(3, 'spread'), atol=1e-4)
    assert view.agg_frame(3, 'spread').max() < 100.0        # not -273-ish


def test_min_and_max_do_not_swap_under_a_positive_scale(kelvin):
    raw, view = kelvin
    assert np.all(view.agg_frame(2, 'max') >= view.agg_frame(2, 'min'))
    assert np.allclose(view.agg_frame(2, 'max'), raw.agg_frame(2, 'max') - 273.15, atol=1e-4)


def test_switching_units_moves_every_read_together(kelvin):
    raw, view = kelvin
    view.set_units('K')
    assert view.units == 'K'
    assert np.array_equal(view.frame(1, 0), raw.frame(1, 0))
    assert np.array_equal(view.series(2, 3), raw.series(2, 3))
    view.set_units('°F')
    assert np.allclose(view.frame(1, 0), raw.frame(1, 0) * 1.8 - 459.67, atol=1e-3)


def test_an_unknown_unit_label_is_ignored_not_a_crash(kelvin):
    _raw, view = kelvin
    assert view.set_units('parsecs') is False
    assert view.units == '°C'


def test_the_cached_range_is_transformed_not_rescanned(kelvin, tmp_path):
    """G19: a pure unit change is affine, so it must never trigger a scan."""
    raw, view = kelvin
    raw_range = view.raw.scan_range()
    calls = []
    view.raw.scan_range = lambda *a, **k: calls.append(1)
    assert np.allclose(view.value_range, (raw_range[0] - 273.15, raw_range[1] - 273.15))
    view.set_units('K')
    assert np.allclose(view.value_range, raw_range)
    assert calls == []


def test_summary_and_labels_describe_the_transformed_quantity(kelvin):
    _raw, view = kelvin
    assert '[°C]' in view.summary()
    view.set_units('K')
    assert '[K]' in view.summary()


def test_a_saved_unit_preference_is_applied_at_construction(tmp_path):
    path = synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    assert FieldView(EnsembleFile(path), units_label='K').units == 'K'
    assert FieldView(EnsembleFile(path), units_label='nonsense').units == '°C'


def test_the_real_cape_file_is_passed_through_untouched(cape_path):
    """CAPE has no sensible conversion, so FieldView must be a literal no-op on it."""
    raw = EnsembleFile(cape_path)
    view = FieldView(EnsembleFile(cape_path))
    assert view.units == 'J kg-1' and not view.can_convert_units
    assert np.array_equal(view.frame(0, 0), raw.frame(0, 0))
    assert np.array_equal(view.series(150, 100), raw.series(150, 100))
    assert np.array_equal(view.agg_frame(60, 'mean'), raw.agg_frame(60, 'mean'))
