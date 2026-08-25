"""V2.4 / G19: the .imsstats.json sidecar holds the RAW range, and a rate view cannot
reuse it -- but a unit change is affine and must never cost a scan."""
import json

import numpy as np
import pytest

import synth
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView


@pytest.fixture
def precip(tmp_path):
    path, _hourly = synth.accumulated_precip(tmp_path / 'ICON_ENS_2026082300_TOT_PREC.nc')
    return path


def counting(view):
    """Wrap the raw scan so a test can count how many passes actually happen."""
    calls = []
    original = view.raw.scan_range

    def spy(*args, **kwargs):
        calls.append(kwargs.get('signature', 'raw'))
        return original(*args, **kwargs)

    view.raw.scan_range = spy
    return calls


def test_a_v1_sidecar_still_loads(tmp_path):
    """No user loses their cached range to the schema change."""
    path = synth.temperature(tmp_path / 't.nc')
    raw = EnsembleFile(path)
    raw.stats_path.write_text(json.dumps(
        {'key': raw._cache_key(), 'min': 250.0, 'max': 310.0}))
    fresh = EnsembleFile(path)
    assert fresh.cached_range() == (250.0, 310.0)
    assert fresh.cached_range('rate:sum:1') is None       # v1 knew only the raw view


def test_a_stale_key_invalidates_every_signature(tmp_path):
    path = synth.temperature(tmp_path / 't.nc')
    raw = EnsembleFile(path)
    raw.stats_path.write_text(json.dumps(
        {'schema': 2, 'key': {'size': 1, 'mtime': 1, 'var': 'nope'},
         'ranges': {'raw': {'min': 0, 'max': 1}, 'rate:sum:1': {'min': 0, 'max': 2}}}))
    fresh = EnsembleFile(path)
    assert fresh.cached_range() is None
    assert fresh.cached_range('rate:sum:1') is None


def test_a_unit_change_reuses_the_cached_range(precip):
    view = FieldView(EnsembleFile(precip))
    view.scan_range()
    calls = counting(view)
    before = view.value_range
    view.set_units('kg m-2')
    assert view.cached_range() is not None
    assert view.value_range == before        # mm and kg m-2 are the same number
    assert calls == []                       # and no scan happened


def test_a_unit_change_with_a_real_scale_transforms_the_range(tmp_path):
    view = FieldView(EnsembleFile(synth.temperature(tmp_path / 't.nc')))
    view.scan_range()
    lo_c, hi_c = view.value_range
    calls = counting(view)
    view.set_units('K')
    lo_k, hi_k = view.value_range
    assert (lo_k, hi_k) == pytest.approx((lo_c + 273.15, hi_c + 273.15))
    assert calls == []


def test_a_rate_change_forces_exactly_one_rescan(precip):
    view = FieldView(EnsembleFile(precip))
    view.scan_range()
    calls = counting(view)

    view.set_rate(1)
    assert view.cached_range() is None                   # the raw range does not apply
    view.scan_range()
    assert calls == ['rate:sum:1']

    view.set_rate(3)
    assert view.cached_range() is None
    view.scan_range()
    assert calls == ['rate:sum:1', 'rate:sum:3']

    # everything already scanned is now instant, in either direction
    view.set_rate(1)
    assert view.cached_range() is not None
    view.set_rate(0)
    assert view.cached_range() is not None
    assert len(calls) == 2


def test_each_signature_is_stored_side_by_side(precip):
    view = FieldView(EnsembleFile(precip))
    view.scan_range()
    view.set_rate(1)
    view.scan_range()
    blob = json.loads(EnsembleFile(precip).stats_path.read_text())
    assert blob['schema'] == 2
    assert set(blob['ranges']) == {'raw', 'rate:sum:1'}
    assert blob['ranges']['raw']['max'] > blob['ranges']['rate:sum:1']['max']


def test_the_rate_range_is_cached_before_units_so_it_can_be_reconverted(precip):
    """The cached number is in step-[1] space; the unit affine is applied on read."""
    view = FieldView(EnsembleFile(precip))
    view.set_rate(1)
    view.scan_range()
    blob = json.loads(EnsembleFile(precip).stats_path.read_text())
    stored = blob['ranges']['rate:sum:1']
    assert view.value_range == pytest.approx((stored['min'], stored['max']))  # mm: identity
    view._units = view._units._replace(label='tenths', a=10.0)
    assert view.value_range == pytest.approx((stored['min'] * 10, stored['max'] * 10))


def test_a_rate_scan_ignores_the_all_nan_window_edge(precip):
    view = FieldView(EnsembleFile(precip))
    view.set_rate(3)
    lo, hi = view.scan_range()
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo >= 0.0


def test_an_unwritable_sidecar_is_not_fatal(precip, monkeypatch):
    """The cache is an optimisation; a read-only data directory must still open."""
    view = FieldView(EnsembleFile(precip))
    monkeypatch.setattr(type(view.raw.stats_path), 'write_text',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('read-only')))
    assert view.scan_range() is not None
