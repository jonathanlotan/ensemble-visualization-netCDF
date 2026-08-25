"""`imsicon.ncwrite`: writing a derived field back out as a real NetCDF-3 file.

The point of these tests is that "writes the dew point" means a file other tools can read,
not a private blob -- so `netCDF4` is the oracle wherever it is installed, exactly as it is
for the reader.
"""
import numpy as np
import pytest

import synth
from imsicon import derived, nc3, ncwrite
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView

netCDF4 = pytest.importorskip('netCDF4', reason='dev-only oracle')


@pytest.fixture
def dewpoint(tmp_path):
    t_path, rh_path = synth.pair(tmp_path)
    return derived.dew_point(derived.open_field(t_path), derived.open_field(rh_path))


# ---- the round trip -----------------------------------------------------------------
def test_a_written_dew_point_reopens_with_identical_values(tmp_path, dewpoint):
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    back = FieldView(EnsembleFile(out))
    for t in range(dewpoint.n_times):
        assert np.allclose(back.ens_frame(t), dewpoint.ens_frame(t), equal_nan=True)


def test_it_is_written_in_kelvin_however_the_screen_is_set(tmp_path, dewpoint):
    """Otherwise the file is a temperature the units registry does not recognise, and
    reopening it warns instead of just showing degC."""
    dewpoint.set_units('°F')
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    back = FieldView(EnsembleFile(out))
    assert back.raw.units == 'K'
    assert back.units == '°C' and back.units_note is None
    assert 250.0 < float(np.nanmin(back.raw.ens_frame(0))) < 320.0


def test_the_written_file_carries_grid_time_and_member_identity(tmp_path, dewpoint):
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    back = EnsembleFile(out)
    assert np.allclose(back.lat, dewpoint.lat) and np.allclose(back.lon, dewpoint.lon)
    assert np.allclose(back.forecast_hours, dewpoint.forecast_hours)
    assert back.run_init == dewpoint.run_init
    # G1: member identity lives only in `history`, so a writer that drops it silently
    # replaces "which member" with "which position".
    assert back.member_labels == dewpoint.member_labels
    assert not back.truncated


def test_the_provenance_of_a_derived_file_is_recorded(tmp_path, dewpoint):
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    attrs = nc3.parse(out)['attrs']
    assert 'T_2M' in attrs['imsicon_provenance']
    assert 'RELHUM_2M' in attrs['imsicon_provenance']
    assert 'Magnus' in attrs['imsicon_provenance']


def test_a_written_dew_point_pairs_with_the_original_temperature(tmp_path, dewpoint):
    """The whole reason to write it: T - Td from files must equal the live computation."""
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    from_file = derived.difference(dewpoint.temperature, FieldView(EnsembleFile(out)))
    live = derived.difference(dewpoint.temperature, dewpoint)
    for t in (0, 2, 5):
        assert np.allclose(from_file.ens_frame(t), live.ens_frame(t), equal_nan=True,
                           atol=1e-4)


# ---- netCDF4 as the oracle -----------------------------------------------------------
def test_netcdf4_reads_what_we_wrote(tmp_path, dewpoint):
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    with netCDF4.Dataset(out) as ds:
        assert ds.file_format == 'NETCDF3_64BIT_OFFSET'
        assert set(ds.variables) == {'time', 'lon', 'lat', 'sfc', 'TD_2M_eps'}
        var = ds.variables['TD_2M_eps']
        assert var.units == 'K'
        assert var.shape == (dewpoint.n_times, dewpoint.n_members,
                             dewpoint.ny, dewpoint.nx)
        for t in range(dewpoint.n_times):
            assert np.allclose(var[t][:], dewpoint.canonical_ens_frame(t), equal_nan=True)


def test_the_record_stride_rule_survives_the_writer(tmp_path, dewpoint):
    """G2: `time` is a record variable too, so it contributes to the stride."""
    out = ncwrite.write_canonical(tmp_path / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    hdr = nc3.parse(out)
    field = hdr['vars']['TD_2M_eps']
    assert hdr['recsize'] == hdr['vars']['time']['vsize'] + field['vsize']


# ---- a difference can be written too --------------------------------------------------
def test_a_difference_writes_with_a_legal_variable_name(tmp_path, dewpoint):
    """`T_2M-TD_2M` is not a NetCDF variable name; the hyphen has to go somewhere."""
    depression = derived.difference(dewpoint.temperature, dewpoint)
    out = ncwrite.write_canonical(tmp_path / 'depression.nc', depression)
    hdr = nc3.parse(out)
    assert nc3.field_name(hdr) == 'T_2M_TD_2M_eps'
    with netCDF4.Dataset(out) as ds:
        assert 'T_2M_TD_2M_eps' in ds.variables
    back = EnsembleFile(out)
    assert np.allclose(back.ens_frame(2), depression.ens_frame(2), equal_nan=True)


def test_a_difference_is_written_in_the_units_on_screen(tmp_path, dewpoint):
    depression = derived.difference(dewpoint.temperature, dewpoint)
    depression.set_units('°F')
    out = ncwrite.write_canonical(tmp_path / 'depression.nc', depression)
    assert nc3.parse(out)['vars']['T_2M_TD_2M_eps']['attrs']['units'] == '°F'


@pytest.mark.parametrize('field, expected', [
    ('T_2M-TD_2M', 'T_2M_TD_2M'), ('TD_2M', 'TD_2M'), ('a b', 'a_b'), ('2M', 'F_2M'),
])
def test_variable_names_are_sanitised(field, expected):
    assert ncwrite.nc_variable_name(field) == expected


# ---- streaming and failure ------------------------------------------------------------
def test_frames_are_pulled_one_at_a_time_not_materialised(tmp_path):
    """A real field is 407 MB; holding the whole derived stack to write it is needless."""
    asked = []
    source = ncwrite.FrameSource(
        (5, 2, 3, 4),
        lambda t: (asked.append(t), np.full((2, 3, 4), float(t), np.float32))[1])
    ncwrite.write_nc3(tmp_path / 'x.nc', 'X', 'K', source)
    assert asked == [0, 1, 2, 3, 4]
    back = EnsembleFile(tmp_path / 'x.nc')
    assert [float(back.ens_frame(t)[0, 0, 0]) for t in range(5)] == [0, 1, 2, 3, 4]


def test_cancelling_a_write_leaves_no_file_at_all(tmp_path, dewpoint):
    """A half-written .nc parses as a short forecast (G26) rather than failing to open."""
    out = tmp_path / 'ICON_ENS_2026082300_TD_2M.nc'
    with pytest.raises(ncwrite.Cancelled):
        ncwrite.write_canonical(out, dewpoint, cancel=lambda: True)
    assert not out.exists()
    assert not out.with_name(out.name + '.part').exists()


def test_a_failing_frame_source_leaves_no_file(tmp_path):
    def boom(t):
        if t == 2:
            raise RuntimeError('disk went away')
        return np.zeros((2, 3, 4), np.float32)

    out = tmp_path / 'x.nc'
    with pytest.raises(RuntimeError):
        ncwrite.write_nc3(out, 'X', 'K', ncwrite.FrameSource((5, 2, 3, 4), boom))
    assert not out.exists() and not out.with_name('x.nc.part').exists()


def test_progress_counts_every_time_step(tmp_path, dewpoint):
    seen = []
    ncwrite.write_canonical(tmp_path / 'x.nc', dewpoint,
                            progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (dewpoint.n_times, dewpoint.n_times)


def test_a_wrongly_shaped_frame_is_rejected(tmp_path):
    source = ncwrite.FrameSource((2, 2, 3, 4), lambda t: np.zeros((2, 3, 9), np.float32))
    with pytest.raises(ValueError, match='expected'):
        ncwrite.write_nc3(tmp_path / 'x.nc', 'X', 'K', source)


def test_history_round_trips_member_numbers():
    history = ncwrite.history_for_members(['member 03', 'member 07'], '2026082300', 'TD_2M')
    assert nc3.member_labels(history, 2) == ['member 03', 'member 07']


def test_history_falls_back_to_position_for_an_unlabelled_member():
    history = ncwrite.history_for_members(['odd', 'names'], '2026082300', 'TD_2M')
    assert nc3.member_labels(history, 2) == ['member 01', 'member 02']


# ---- a plain field saves as what is on screen -------------------------------------------
def test_saving_a_plain_field_writes_the_displayed_units(tmp_path):
    view = FieldView(EnsembleFile(synth.temperature(tmp_path / 'T.nc')))
    assert view.units == '°C'                      # the v2 default for a temperature
    out = ncwrite.write_canonical(tmp_path / 'copy.nc', view)
    assert nc3.parse(out)['vars']['T_2M_eps']['attrs']['units'] == '°C'
    assert np.allclose(EnsembleFile(out).ens_frame(1), view.ens_frame(1))
