"""V2.0: the synthetic NetCDF-3 writer must produce genuinely valid files.

The decisive test is netCDF4 reading them: a writer built from the spec agreeing with a
reader built from the same spec could be two matching misreadings, but the reference C
library agreeing with both cannot be.
"""
import numpy as np
import pytest

from imsicon import nc3
import synth


@pytest.fixture
def temp_file(tmp_path):
    return synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')


def test_our_reader_round_trips_the_writer(temp_file):
    hdr = nc3.parse(temp_file)
    assert hdr['version'] == 2                      # 64-bit offset, like the real product
    assert nc3.field_name(hdr) == 'T_2M_eps'
    view = nc3.view(temp_file, hdr, 'T_2M_eps')
    assert view.shape == synth.SHAPE
    assert view.dtype == np.dtype('>f4')            # G3: big-endian, never native


def test_record_stride_follows_the_two_record_variable_rule(temp_file):
    """G2: `time` is a record variable too, so it contributes to the stride."""
    hdr = nc3.parse(temp_file)
    _t, m, ny, nx = synth.SHAPE
    assert hdr['recsize'] == 8 + m * ny * nx * 4
    assert hdr['vars']['T_2M_eps']['begin'] - hdr['vars']['time']['begin'] == 8


def test_netcdf4_reads_it_element_for_element(temp_file):
    """The independent-implementation check that makes this fixture trustworthy."""
    netCDF4 = pytest.importorskip('netCDF4')
    hdr = nc3.parse(temp_file)
    ours = np.asarray(nc3.view(temp_file, hdr, 'T_2M_eps'), dtype=np.float32)
    with netCDF4.Dataset(temp_file) as ds:
        assert ds.data_model in ('NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT')
        theirs = np.asarray(ds.variables['T_2M_eps'][:], dtype=np.float32)
        assert ds.variables['T_2M_eps'].units == 'K'
        assert ds.variables['T_2M_eps'].dimensions == ('time', 'sfc', 'lat', 'lon')
        assert np.array_equal(np.asarray(ds.variables['lat'][:]), _lat(hdr, temp_file))
    assert np.array_equal(ours, theirs)


def _lat(hdr, path):
    return np.asarray(nc3.view(path, hdr, 'lat'), dtype=float)


def test_coordinates_and_time_survive(temp_file):
    hdr = nc3.parse(temp_file)
    lat = np.asarray(nc3.view(temp_file, hdr, 'lat'), dtype=float)
    lon = np.asarray(nc3.view(temp_file, hdr, 'lon'), dtype=float)
    assert np.allclose(np.diff(lat), 0.025) and np.allclose(np.diff(lon), 0.025)
    epoch, scale = nc3.parse_time_units(hdr['vars']['time']['attrs']['units'])
    assert (epoch.year, epoch.month, epoch.day) == (2026, 8, 23)   # G4: unpadded month
    assert scale == 60
    minutes = np.asarray(nc3.view(temp_file, hdr, 'time'), dtype=float)
    assert np.array_equal(minutes, np.arange(synth.SHAPE[0]) * 60.0)


def test_member_labels_come_from_the_history_attribute(temp_file):
    """G1: the fixtures carry a real IMS history line so member identity is exercised."""
    hdr = nc3.parse(temp_file)
    labels = nc3.member_labels(hdr['attrs']['history'], 3)
    assert labels == ['member 01', 'member 02', 'member 03']


def test_sfc_is_all_zeros_like_the_real_product(temp_file):
    hdr = nc3.parse(temp_file)
    assert np.all(np.asarray(nc3.view(temp_file, hdr, 'sfc'), dtype=float) == 0.0)


def test_accumulation_fixture_is_monotone(tmp_path):
    path, hourly = synth.accumulated_precip(tmp_path / 'p.nc')
    hdr = nc3.parse(path)
    data = np.asarray(nc3.view(path, hdr, 'TOT_PREC_eps'), dtype=np.float64)
    assert np.all(np.diff(data[:, 0, 0, 0]) >= 0)                # G14 detection signal
    assert np.allclose(data[:, 0, 0, 0], np.cumsum(hourly))


def test_radiation_fixture_is_a_running_mean_not_a_sum(tmp_path):
    path, hourly = synth.averaged_radiation(tmp_path / 'r.nc')
    hdr = nc3.parse(path)
    data = np.asarray(nc3.view(path, hdr, 'ASWDIR_S_eps'), dtype=np.float64)[:, 0, 0, 0]
    hours = np.arange(len(hourly), dtype=float)
    assert np.allclose(data[1:], np.cumsum(hourly)[1:] / hours[1:])
    assert not np.all(np.diff(data) >= 0)          # a mean is NOT monotone -- that is G14


@pytest.mark.parametrize('encoding,hi', [('fraction', 1.0), ('percent', 100.0),
                                         ('zero', 0.0)])
def test_cloud_fixtures_cover_the_three_encodings(tmp_path, encoding, hi):
    path = synth.cloud(tmp_path / f'c_{encoding}.nc', encoding=encoding)
    hdr = nc3.parse(path)
    data = np.asarray(nc3.view(path, hdr, 'CLCT_eps'), dtype=np.float64)
    assert data.max() <= hi + 1e-6
    if encoding != 'zero':
        assert data.max() > hi * 0.5


def test_a_truncated_file_is_clamped_instead_of_segfaulting(tmp_path):
    """G26: an interrupted download leaves a .nc whose header over-claims its records.

    Without the clamp, nc3.view builds an as_strided window past the end of the mapping
    and touching it crashes the interpreter -- there is no exception to catch.
    """
    path = synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    full = nc3.parse(path)
    assert full['numrecs'] == 6 and full['truncated'] is False

    raw = path.read_bytes()
    record_start = min(v['begin'] for v in full['vars'].values() if v['record'])
    cut = tmp_path / 'cut.nc'
    cut.write_bytes(raw[:record_start + 2 * full['recsize'] + 17])   # 2 records + a stub

    hdr = nc3.parse(cut)
    assert hdr['declared_numrecs'] == 6
    assert hdr['numrecs'] == 2 and hdr['truncated'] is True
    assert hdr['vars']['T_2M_eps']['shape'][0] == 2

    view = nc3.view(cut, hdr, 'T_2M_eps')
    assert view.shape[0] == 2
    whole = np.asarray(nc3.view(path, full, 'T_2M_eps')[:2], dtype=np.float32)
    assert np.array_equal(np.asarray(view, dtype=np.float32), whole)


def test_an_empty_record_section_does_not_explode(tmp_path):
    path = synth.temperature(tmp_path / 't.nc')
    full = nc3.parse(path)
    record_start = min(v['begin'] for v in full['vars'].values() if v['record'])
    cut = tmp_path / 'header_only.nc'
    cut.write_bytes(path.read_bytes()[:record_start])
    hdr = nc3.parse(cut)
    assert hdr['numrecs'] == 0 and hdr['truncated'] is True
    assert nc3.view(cut, hdr, 'T_2M_eps').shape[0] == 0


def test_an_incomplete_file_says_so_in_the_summary(tmp_path):
    """G26: it opens and reads correctly, but the user must know it stops early."""
    from imsicon.dataset import EnsembleFile
    path = synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    full = nc3.parse(path)
    record_start = min(v['begin'] for v in full['vars'].values() if v['record'])
    cut = tmp_path / 'ICON_ENS_2026082300_T_2M_cut.nc'
    cut.write_bytes(path.read_bytes()[:record_start + 3 * full['recsize']])

    whole, part = EnsembleFile(path), EnsembleFile(cut)
    assert whole.truncated is False and whole.truncation_note is None
    assert part.truncated is True and part.n_times == 3 and part.declared_times == 6
    assert 'INCOMPLETE: 3/6 steps' in part.summary()
    assert 'interrupted download' in part.truncation_note
    assert np.array_equal(part.series(1, 1), whole.series(1, 1)[:3])
