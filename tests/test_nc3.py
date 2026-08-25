"""The byte-math contract: if these fail, every number the app shows is suspect."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from imsicon import nc3

ROOT = Path(__file__).resolve().parent.parent


def test_record_stride_includes_the_time_variable(cape_path):
    """G2: time is also a record var, so the stride is 3_361_688, not 3_361_680."""
    hdr = nc3.parse(cape_path)
    assert hdr['recsize'] == 3_361_688
    assert hdr['numrecs'] == 121


def test_field_is_big_endian_float32(cape_path):
    """G3: classic NetCDF is big-endian; native float32 would read garbage."""
    hdr = nc3.parse(cape_path)
    assert hdr['vars'][nc3.field_name(hdr)]['dtype'] == '>f4'


def test_matches_netcdf4(cape_path):
    nc = pytest.importorskip('netCDF4')
    hdr = nc3.parse(cape_path)
    name = nc3.field_name(hdr)
    ours = nc3.view(cape_path, hdr, name)
    with nc.Dataset(cape_path) as ref:
        theirs = ref[name]
        for t in (0, 1, 47, 120):
            assert np.array_equal(np.asarray(ours[t]), np.asarray(theirs[t]))
        rng = np.random.default_rng(0)
        for _ in range(200):
            t, m, y, x = (rng.integers(121), rng.integers(20),
                          rng.integers(261), rng.integers(161))
            assert float(ours[t, m, y, x]) == float(theirs[t, m, y, x])


@pytest.mark.parametrize('units,expected_scale,expected_month', [
    ('minutes since 2026-8-23 00:00:00', 60, 8),          # G4: single-digit month
    ('minutes since 2026-08-23 00:00:00', 60, 8),
    ('hours since 2026-12-01 06:00:00', 3600, 12),
    ('seconds since 2026-1-1 00:00', 1, 1),
])
def test_tolerant_time_units(units, expected_scale, expected_month):
    epoch, scale = nc3.parse_time_units(units)
    assert scale == expected_scale
    assert epoch.month == expected_month


def test_iso_parser_would_have_failed():
    """Documents why parse_time_units exists at all."""
    import datetime as dt
    with pytest.raises(ValueError):
        dt.datetime.fromisoformat('2026-8-23 00:00:00')


def test_member_labels_from_history(cape_path):
    """G1: sfc is all zeros, so member identity comes from the cdo merge command."""
    hdr = nc3.parse(cape_path)
    labels = nc3.member_labels(hdr['attrs']['history'], 20)
    assert labels[0] == 'member 01' and labels[-1] == 'member 20'
    assert len(labels) == len(set(labels)) == 20
    sfc = np.asarray(nc3.view(cape_path, hdr, 'sfc'))
    assert np.all(sfc == 0)


def test_member_labels_fall_back_to_position():
    assert nc3.member_labels('no member tokens here', 3) == \
        ['member 01', 'member 02', 'member 03']


def test_hdf5_is_rejected_clearly(tmp_path):
    fake = tmp_path / 'fake.nc'
    fake.write_bytes(b'\x89HDF\r\n\x1a\n' + b'\0' * 64)
    with pytest.raises(nc3.UnsupportedFormat, match='HDF5'):
        nc3.parse(fake)


def test_prototype_cli_runs_without_arguments():
    """B1: `python prototype/nc3.py` used to die with IndexError on sys.argv[1]."""
    # `Path.glob` returns a GENERATOR, which is always truthy -- so this guard used to let
    # the test run (and fail on exit code 2) on any checkout without a `data/` directory.
    if not any((ROOT / 'data').glob('*.nc')):
        pytest.skip('no data files: the CLI has nothing to auto-discover')
    done = subprocess.run([sys.executable, str(ROOT / 'prototype' / 'nc3.py')],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr
    assert 'recsize' in done.stdout


def test_prototype_cli_rejects_a_missing_file():
    done = subprocess.run([sys.executable, str(ROOT / 'prototype' / 'nc3.py'), '/nope.nc'],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 2
    assert 'no such file' in done.stderr
