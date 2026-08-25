"""V2.1: the prefix sniff (v2.md 1.2) -- exercised against local files, never the network."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import sniff_headers as sniff                                        # noqa: E402

from imsicon import nc3                                              # noqa: E402

CAPE_BZ2 = ROOT / 'data' / 'ICON_ENS_2026082300_CAPE_ML.nc.bz2'
SNOW_BZ2 = ROOT / 'data' / 'H_SNOW.nc.bz2'


@pytest.fixture(scope='module')
def cape_bz2():
    if not CAPE_BZ2.exists():
        pytest.skip(f'missing {CAPE_BZ2}')
    return CAPE_BZ2


def test_four_mib_prefix_settles_the_units_string(cape_bz2):
    report = sniff.sniff_local(cape_bz2, sniff.PREFIX_SHALLOW)
    assert report['units'] == 'J kg-1'                 # the v2.md 1.1 measured value
    assert report['long_name'] == 'cape of mean surface layer parcel'
    assert report['recsize'] == 3_361_688               # G2 holds in the prefix too
    assert report['begin'] == 6372
    assert report['steps'] >= 2                         # 4 MiB -> 2 full time records


def test_prefix_frame_is_identical_to_the_full_file(cape_bz2):
    """The claim the whole sniffer rests on: prefix bytes are the real file's bytes."""
    full = ROOT / 'data' / 'ICON_ENS_2026082300_CAPE_ML.nc'
    if not full.exists():
        pytest.skip('uncompressed reference file missing')
    raw = sniff.decompress_prefix(cape_bz2.read_bytes()[:sniff.PREFIX_SHALLOW])
    tmp = Path(sniff.tempfile.mkdtemp()) / 'prefix.nc'
    tmp.write_bytes(raw)
    hdr = nc3.parse(tmp)
    steps = sniff.clamp_to_available(hdr, len(raw))
    prefix_frame = np.asarray(nc3.view(tmp, hdr, 'CAPE_ML_eps')[0], dtype=np.float32)

    fhdr = nc3.parse(full)
    full_frame = np.asarray(nc3.view(full, fhdr, 'CAPE_ML_eps')[0], dtype=np.float32)
    assert steps >= 1
    assert np.array_equal(prefix_frame, full_frame)


def test_clamping_stops_a_strided_view_reading_past_the_prefix(cape_bz2):
    """G26: as_strided does not bounds-check, so an unclamped numrecs is a segfault."""
    raw = sniff.decompress_prefix(cape_bz2.read_bytes()[:sniff.PREFIX_SHALLOW])
    tmp = Path(sniff.tempfile.mkdtemp()) / 'prefix.nc'
    tmp.write_bytes(raw)
    hdr = nc3.parse(tmp)
    assert hdr['declared_numrecs'] == 121               # what the header claims
    assert hdr['truncated'] is True
    steps = hdr['numrecs']                              # what the bytes actually support
    assert 0 < steps < 121
    assert hdr['vars']['CAPE_ML_eps']['shape'][0] == steps
    record_start = min(v['begin'] for v in hdr['vars'].values() if v['record'])
    assert record_start + steps * hdr['recsize'] <= len(raw)
    # the decisive part: every element of the view is inside the mapping
    view = nc3.view(tmp, hdr, 'CAPE_ML_eps')
    assert float(np.asarray(view[steps - 1], dtype=np.float32).max()) >= 0.0


def test_a_stream_shorter_than_the_prefix_is_success_not_an_error():
    """v2.md 1.2 caveat: H_SNOW.nc.bz2 is 125,788 B in total (all-zero summer snow)."""
    if not SNOW_BZ2.exists():
        pytest.skip(f'missing {SNOW_BZ2}')
    assert SNOW_BZ2.stat().st_size < sniff.PREFIX_SHALLOW
    report = sniff.sniff_local(SNOW_BZ2, sniff.PREFIX_SHALLOW)
    assert report['units'] == 'm'
    assert report['steps'] == 121                       # the whole file fits in the prefix


def test_a_non_bz2_prefix_reports_instead_of_crashing(tmp_path):
    junk = tmp_path / 'not_really.nc.bz2'
    junk.write_bytes(b'this is not a bzip2 stream at all')
    with pytest.raises(sniff.SniffError):
        sniff.sniff_local(junk)


def test_cloud_verdict_has_exactly_three_outcomes():
    """G22: refusing to decide is a valid answer, and must not be silently 'fraction'."""
    assert sniff.cloud_verdict(np.array([0.0, 0.4, 0.99])).startswith('FRACTION')
    assert sniff.cloud_verdict(np.array([0.0, 40.0, 99.0])).startswith('PERCENT')
    assert sniff.cloud_verdict(np.zeros(10)).startswith('UNDECIDABLE')


def test_monotonicity_separates_accumulation_from_the_rest():
    """G14's detection signal."""
    rising = np.cumsum(np.ones((6, 1, 1, 1)), axis=0)
    assert sniff.monotonicity(rising).startswith('non-decreasing')
    wobbly = np.array([0.0, 5.0, 2.0, 9.0, 1.0, 3.0]).reshape(6, 1, 1, 1)
    assert sniff.monotonicity(wobbly).startswith('not monotone')
    assert sniff.monotonicity(np.zeros((2, 1, 1, 1))) is None       # too short to tell


def test_credentials_are_never_read_from_a_hardcoded_default(monkeypatch):
    monkeypatch.delenv('IMS_USER', raising=False)
    monkeypatch.delenv('IMS_PASS', raising=False)
    monkeypatch.setitem(sys.modules, 'keyring', None)
    with pytest.raises(sniff.SniffError) as err:
        sniff.credentials()
    assert 'IMS_USER' in str(err.value)
