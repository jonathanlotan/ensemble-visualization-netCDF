"""v7: the second axis of a file, when it is pressure levels rather than members.

The deterministic ICON-LAM product (`IMS_ICON_manual.pdf`) has the same shape as the
ensemble and means something different by it, so every test here is about the difference:
what the axis is read as, what may be computed across it, and what a position on it is
called.
"""
import numpy as np
import pytest

from imsicon import derived, download, ingest, levels, nc3, ncwrite, products, transform
from imsicon.dataset import EnsembleFile, level_stats
from imsicon.fieldview import FieldView

import synth


# ---- reading the axis out of a file ----------------------------------------------------
def test_a_plev_coordinate_is_read_as_the_pressure_levels_it_holds(tmp_path):
    ds = EnsembleFile(synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc'))
    assert ds.axis.kind == 'pressure'
    assert ds.member_labels == ['1000 hPa', '925 hPa', '850 hPa', '700 hPa', '500 hPa']
    assert ds.axis.describe() == '5 pressure levels'


def test_levels_stored_in_pascals_are_shown_in_hectopascals(tmp_path):
    """The file may be in Pa; a chart is read in hPa, and 85000 hPa is not a pressure."""
    ds = EnsembleFile(synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc',
                                           level_units='Pa'))
    assert ds.member_labels[0] == '1000 hPa'
    assert ds.axis.values[2] == pytest.approx(850.0)


def test_the_ensembles_sfc_axis_is_still_members(tmp_path):
    """G1 regression: 20 zeros on an axis called `sfc` are members, not levels."""
    ds = EnsembleFile(synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
    assert ds.axis.kind == 'member'
    assert ds.axis.aggregatable
    assert ds.member_labels[0].startswith('member')


def test_a_surface_field_with_no_vertical_dimension_opens_with_one_level(tmp_path):
    """The deterministic run's 2-D fields are (time, lat, lon): no axis to choose on."""
    path = synth.surface_field(tmp_path / 'IE_2026083100_t_2m.nc')
    assert nc3.parse(path)['vars']['t_2m']['dims'] == ['time', 'lat', 'lon']
    ds = EnsembleFile(path)
    assert ds.axis.kind == 'single' and ds.n_members == 1
    assert ds.frame(0, 0).shape == (ds.ny, ds.nx)
    assert ds.series(1, 1).shape == (ds.n_times, 1)


def test_a_degenerate_level_axis_falls_back_to_the_manual_and_says_so(tmp_path):
    """A 3-D field merged the way the ensemble was would carry 20 zeros for its levels.

    The manual's ladder is then the best available answer -- and the note is not optional:
    an assumed order that happened to be reversed would label every map wrongly while
    looking entirely normal.
    """
    data = np.zeros((2, len(products.PRESSURE_LEVELS), 3, 4), dtype=np.float32)
    path = ncwrite.write_nc3(tmp_path / 'IE_2026083100_temp.nc', 'temp', 'K', data,
                             variable='temp')          # levels=None -> the `sfc` axis
    ds = EnsembleFile(path)
    assert ds.axis.kind == 'pressure'
    assert ds.member_labels[0] == '150 hPa' and ds.member_labels[-1] == '1000 hPa'
    assert 'measured on the server' in ds.axis_note and 'upside down' in ds.axis_note


def test_the_manuals_shorter_ladder_is_still_recognised(tmp_path):
    """The manual lists 20 levels and the server publishes 22, so a file with either
    count gets a ladder -- and a note that says which one it fell back to."""
    data = np.zeros((2, len(products.MANUAL_PRESSURE_LEVELS), 3, 4), dtype=np.float32)
    path = ncwrite.write_nc3(tmp_path / 'IE_2026083100_rh.nc', 'rh', '%', data,
                             variable='rh')
    ds = EnsembleFile(path)
    assert ds.axis.kind == 'pressure' and ds.n_members == 20
    assert ds.member_labels[0] == '1000 hPa'
    assert "manual's table" in ds.axis_note


def test_a_field_the_catalogue_does_not_call_3d_is_not_given_levels(tmp_path):
    """The fallback is for the six 3-D fields, not for anything 20 planes deep."""
    data = np.zeros((2, len(products.PRESSURE_LEVELS), 3, 4), dtype=np.float32)
    path = ncwrite.write_nc3(tmp_path / 'IE_2026083100_clct.nc', 'clct', '%', data,
                             variable='clct')
    assert EnsembleFile(path).axis.kind == 'member'


def test_distinct_values_that_are_not_pressures_are_not_read_as_pressures():
    axis = levels.axis_for(20, dim='sfc', coord=np.arange(20.0), units=None,
                           history='')
    assert axis.kind == 'member'
    assert levels.pressure_scale('kg m-2') is None
    assert levels.pressure_scale(None) is None
    assert levels.pressure_scale('  HPA ') == 1.0


# ---- moving along the axis --------------------------------------------------------------
def test_up_is_up_the_atmosphere_whichever_way_the_file_stores_the_levels():
    """The load-bearing rule of the up/down buttons: `up` is lower pressure, not `+1`."""
    descending = levels.pressure_axis([1000, 925, 850, 700, 500])
    ascending = levels.pressure_axis([500, 700, 850, 1000])
    # 850 hPa is index 2 in one file and index 2 in the other, and up is 700 hPa in both.
    assert descending.labels[descending.step(2, 1)] == '700 hPa'
    assert ascending.labels[ascending.step(2, 1)] == '700 hPa'
    assert descending.labels[descending.step(2, -1)] == '925 hPa'
    assert ascending.labels[ascending.step(2, -1)] == '1000 hPa'


def test_stepping_stops_at_the_ends_instead_of_wrapping():
    axis = levels.pressure_axis([1000, 925, 850])
    assert axis.labels[axis.step(0, -1)] == '1000 hPa'      # already at the bottom
    assert axis.labels[axis.step(2, +5)] == '850 hPa'       # already at the top
    assert levels.single_axis().step(0, 1) == 0


def test_a_member_axis_steps_by_index():
    axis = levels.member_axis(['member 01', 'member 02', 'member 03'])
    assert axis.step(0, 1) == 1 and axis.step(2, -1) == 1
    assert axis.step(2, 1) == 2


def test_a_file_opens_on_the_conventional_low_level_chart():
    assert levels.pressure_axis(products.PRESSURE_LEVELS).default_index == \
        list(products.PRESSURE_LEVELS).index(850)
    assert levels.member_axis(['a', 'b']).default_index == 0


# ---- what may be computed across it ------------------------------------------------------
def test_a_column_of_levels_is_not_aggregatable_and_an_ensemble_is():
    assert not levels.pressure_axis([1000, 850]).aggregatable
    assert not levels.single_axis().aggregatable
    assert levels.member_axis(['member 01']).aggregatable


def test_the_readout_for_a_column_names_the_level_each_number_came_from():
    stats = level_stats([5.0, 9.0, 1.0], ['1000 hPa', '850 hPa', '500 hPa'], index=1)
    assert stats['value'] == 9.0 and stats['level'] == '850 hPa'
    assert (stats['max'], stats['max_level']) == (9.0, '850 hPa')
    assert (stats['min'], stats['min_level']) == (1.0, '500 hPa')
    assert 'mean' not in stats and 'p90' not in stats


def test_an_all_nan_column_reports_nothing_rather_than_a_number():
    stats = level_stats([np.nan, np.nan], ['1000 hPa', '850 hPa'], index=0)
    assert np.isnan(stats['value']) and np.isnan(stats['max'])
    assert stats['max_level'] == '' and stats['n'] == 0


# ---- the values themselves ---------------------------------------------------------------
def test_every_level_reads_back_the_value_it_was_written_with(tmp_path):
    values = np.random.default_rng(3).normal(250, 20, (4, 5, 4, 5))
    path = synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc', values=values)
    ds = EnsembleFile(path)
    for level in range(5):
        assert np.allclose(ds.frame(2, level), values[2, level], atol=1e-3)
    assert np.allclose(ds.series(1, 2), values[:, :, 1, 2], atol=1e-3)


def test_a_temperature_on_pressure_levels_gets_the_registrys_celsius(tmp_path):
    """`temp` and `T_2M` are one quantity spelled twice, so `field_key` folds them."""
    view = FieldView(EnsembleFile(synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc')))
    assert view.units == '°C'
    assert view.unit_labels == ['°C', 'K', '°F']
    assert view.frame(0, 0).mean() == pytest.approx(
        view.raw.frame(0, 0).mean() - 273.15, abs=1e-3)
    assert view.isolines is not None and view.isolines.step == pytest.approx(1.0)


def test_the_deterministic_spellings_inherit_the_ensembles_registry():
    for icon, ensemble in (('t_2m', 'T_2M'), ('tot_prec', 'TOT_PREC'), ('clct', 'CLCT')):
        assert transform.UNITS[products.field_key(icon)] is transform.UNITS[ensemble]
    assert transform.choices_for('pres_msl', 'Pa')[0][0].label == 'hPa'
    assert transform.choices_for('geopot', 'm2 s-2')[0][0].label == 'kft'


def test_the_manuals_accumulated_fields_are_de_accumulable_and_the_others_are_not():
    assert transform.accumulation_kind('tot_prec') == 'sum'
    assert transform.accumulation_kind('rain_gsp') == 'sum'
    # "mean since model start" in as many words in the manual...
    assert transform.accumulation_kind('asodird_s') == 'mean'
    # ...and these two are not described that way, so they are never differenced (G14).
    assert transform.accumulation_kind('sodifd_s') is None
    assert transform.accumulation_kind('sob_t') is None


# ---- naming, discovery and the catalogue -------------------------------------------------
def test_the_two_families_never_collide_on_one_run(tmp_path):
    synth.temperature(tmp_path / 'ICON_ENS_2026083100_T_2M.nc')
    synth.surface_field(tmp_path / 'IE_2026083100_t_2m.nc')
    found = ingest.scan_for_fields([tmp_path])
    assert set(found) == {('ens', '2026083100', 'T_2M'), ('icon', '2026083100', 't_2m')}
    assert list(ingest.fields_of_run(found, products.ICON, '2026083100')) == ['t_2m']
    assert list(ingest.fields_of_run(found, products.ENSEMBLE, '2026083100')) == ['T_2M']


def test_an_ie_listing_is_parsed_and_the_local_name_is_rebuilt():
    html = ('<a href="IE_2026083100_temp.nc.bz2">IE_2026083100_temp.nc.bz2</a>  99000\n'
            '<a href="IE_2026083112_t_2m.nc.bz2">IE_2026083112_t_2m.nc.bz2</a>  4200\n'
            '<a href="ICON_ENS_2026083100_T_2M.nc.bz2">not this family</a>  1\n')
    found = download.parse_listing(html, products.ICON)
    assert [f.field for f in found] == ['t_2m', 'temp']
    assert found[1].local_name() == 'IE_2026083100_temp.nc.bz2'
    assert found[1].levels and not found[0].levels
    assert found[1].url.endswith('/IMS_ICON/IE_2026083100_temp.nc.bz2')


def test_a_traversal_in_the_listing_cannot_choose_a_path():
    """G27, for the second family: the name on disk is rebuilt from the captures."""
    html = '<a href="../../IE_2026083100_temp.nc.bz2">../../evil.nc.bz2</a> 1'
    for entry in download.parse_listing(html, products.ICON):
        assert entry.local_name() == 'IE_2026083100_temp.nc.bz2'


def test_the_deterministic_folder_can_be_pointed_elsewhere(monkeypatch):
    """It is inferred from the server layout, not measured, so it must be overridable."""
    assert download.base_url(products.ICON).endswith('/IMS_ICON/')
    monkeypatch.setenv(download.ICON_URL_ENV, 'https://example.org/other')
    assert download.base_url(products.ICON) == 'https://example.org/other/'
    assert download.base_url(products.ENSEMBLE) == products.ENSEMBLE.base_url


def test_the_catalogue_matches_what_the_server_publishes():
    """MEASURED against run 2026083012: 50 fields, two of which the manual omits."""
    fields = [p.field for p in products.ICON.products]
    assert len(fields) == 50 and len(set(fields)) == 50
    assert {'sob_s', 'sou_s'} <= set(fields)          # on the server, not in the manual
    assert [p.field for p in products.ICON.products if p.levels] == \
        ['temp', 'rh', 'u', 'v', 'omega', 'geopot']
    # 22 levels, stored ascending in pressure: 150 hPa first, 1000 hPa last. The manual
    # says 20, the other way up -- which is exactly why the axis is read from the file.
    assert len(products.PRESSURE_LEVELS) == 22
    assert products.PRESSURE_LEVELS[0] == 150 and products.PRESSURE_LEVELS[-1] == 1000
    assert len(products.MANUAL_PRESSURE_LEVELS) == 20
    assert products.MANUAL_PRESSURE_LEVELS[0] == 1000


def test_a_one_plane_field_is_named_by_the_height_it_carries():
    """Measured: t_2m is (time, height, lat, lon) with height = 2 m, u_10m with 10 m.
    Calling a 10 m gust "surface" is not wrong enough to matter, and naming it what the
    file says is free."""
    assert levels.axis_for(1, dim='height', coord=np.array([2.0]),
                           units='m').labels == ('2 m',)
    assert levels.axis_for(1, dim='height', coord=np.array([10.0]),
                           units='m').describe() == '10 m (single level)'
    assert levels.axis_for(1, dim='height', coord=np.array([0.0]),
                           units='m').labels == ('surface',)
    assert levels.axis_for(1, dim='sfc', coord=np.array([0.0]),
                           units=None).labels == ('surface',)


# ---- pairing across families and axes ----------------------------------------------------
def _view(path):
    return FieldView(EnsembleFile(path))


def test_an_ensemble_and_a_pressure_file_are_refused_as_a_pair(tmp_path):
    ensemble = _view(synth.temperature(tmp_path / 'ICON_ENS_2026083100_T_2M.nc',
                                       n_members=5, n_times=4))
    column = _view(synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc'))
    with pytest.raises(derived.PairError) as excinfo:
        derived.check_pairable(ensemble, column)
    assert 'pressure level' in str(excinfo.value) and 'member' in str(excinfo.value)


def test_two_files_with_different_level_ladders_are_refused(tmp_path):
    a = _view(synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc'))
    b = _view(synth.pressure_field(tmp_path / 'IE_2026083100_rh.nc',
                                   levels=[1000, 925, 850, 700, 400]))
    with pytest.raises(derived.PairError) as excinfo:
        derived.check_pairable(a, b)
    assert '500 hPa' in str(excinfo.value) and '400 hPa' in str(excinfo.value)


def test_the_wind_map_works_on_pressure_levels(tmp_path):
    """u and v of the deterministic run are 3-D, so the barbs are the wind AT a level."""
    rng = np.random.default_rng(5)
    shape = (4, 5, 4, 5)
    u = _view(synth.pressure_field(tmp_path / 'IE_2026083100_u.nc', field='u',
                                   units='m s-1', values=rng.normal(0, 8, shape)))
    v = _view(synth.pressure_field(tmp_path / 'IE_2026083100_v.nc', field='v',
                                   units='m s-1', values=rng.normal(0, 8, shape)))
    view = derived.wind(u, v)
    assert view.axis.kind == 'pressure'
    assert not view.axis.aggregatable
    assert np.allclose(view.ens_frame(1),
                       np.hypot(u.raw.ens_frame(1), v.raw.ens_frame(1)), atol=1e-4)
    # One level's barbs are that level's vector, not a blend of the column.
    bu, bv = view.wind_vectors(1, 'member', 2)
    assert np.allclose(bu, u.raw.frame(1, 2) * 3600.0 / 1852.0, atol=1e-3)
    # ...and the title says so without claiming an ensemble member that does not exist.
    assert view.barb_label('member') == 'barbs (kt): the wind at this level'
    assert view.barb_label('member', over='temp').startswith(
        'barbs (kt) from u/v:')
    assert view.field == derived.UPPER_WIND_FIELD and view.display_name == 'wind'


def test_the_two_wind_pairs_are_told_apart(tmp_path):
    """A deterministic run can offer both a 10 m wind map and one on pressure levels."""
    present = {'u_10m': 1, 'v_10m': 1, 'u': 1, 'v': 1}
    assert derived.wind_pairs_in(products.ICON, present) == [('u_10m', 'v_10m'), ('u', 'v')]
    assert derived.wind_pair_for(products.ICON, present, prefer='u') == ('u', 'v')
    assert derived.wind_pair_for(products.ICON, present) == ('u_10m', 'v_10m')
    assert derived.wind_pairs_in(products.ICON, {'u': 1, 'v': 1}) == [('u', 'v')]
    assert derived.wind_pairs_in(products.ENSEMBLE, {'U_10M': 1, 'V_10M': 1}) == \
        [('U_10M', 'V_10M')]


def test_the_native_dew_point_pair_is_recognised_as_the_depression(tmp_path):
    """The deterministic run publishes td_2m, so T-Td is a difference of two files.

    It must land on the same name, interval and sort band as the ensemble's computed one,
    which is why `DifferenceView` compares its operands through `products.field_key`.
    """
    t = _view(synth.surface_field(tmp_path / 'IE_2026083100_t_2m.nc'))
    td = _view(synth.surface_field(tmp_path / 'IE_2026083100_td_2m.nc', field='td_2m',
                                   values=np.full((4, 4, 5), 285.0)))
    view = derived.difference(t, td)
    assert view.display_name == derived.DEPRESSION_NAME
    assert view.sort_scale is not None and view.sort_scale.top == pytest.approx(2.0)
    assert view.isolines.step == pytest.approx(0.5)


# ---- writing one back out ----------------------------------------------------------------
def test_a_pressure_field_written_and_reopened_keeps_its_levels(tmp_path):
    view = _view(synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc'))
    out = ncwrite.write_canonical(tmp_path / 'IE_2026083100_temp2.nc', view)
    again = EnsembleFile(out)
    assert again.member_labels == view.member_labels
    assert again.axis.kind == 'pressure' and again.axis_note is None
    assert np.allclose(again.frame(1, 3), view.frame(1, 3), atol=1e-4)
    # No fabricated member history: this file has levels, not members (G1).
    assert 'ICON_ENS' not in again.hdr['attrs'].get('history', '')


def test_a_surface_field_is_written_back_as_a_surface_field(tmp_path):
    view = _view(synth.surface_field(tmp_path / 'IE_2026083100_t_2m.nc'))
    out = ncwrite.write_canonical(tmp_path / 'IE_2026083100_t_2m_copy.nc', view)
    assert nc3.parse(out)['vars']['t_2m']['dims'] == ['time', 'lat', 'lon']
    assert EnsembleFile(out).axis.kind == 'single'


def test_two_one_plane_fields_at_different_heights_still_pair(tmp_path):
    """A 10 m wind belongs over a surface precipitation map, and `t_2m - t_g` is an R5
    example -- so the height LABEL of a single-level field must not be read as an identity
    the way a member number is (G17 needs more than one position to be in danger)."""
    from imsicon import ncwrite
    surface = ncwrite.write_nc3(tmp_path / 'IE_2026083100_tot_prec.nc', 'tot_prec',
                                'kg m-2', np.zeros((4, 1, 4, 5), dtype=np.float32),
                                levels=False, variable='tot_prec')
    at_10m = ncwrite.write_nc3(tmp_path / 'IE_2026083100_u_10m.nc', 'u_10m', 'm s-1',
                               np.ones((4, 1, 4, 5), dtype=np.float32),
                               levels=[10.0], level_units='m', variable='u_10m')
    a, b = _view(surface), _view(at_10m)
    assert a.axis.labels == ('surface',) and b.axis.labels == ('10 m',)
    derived.check_pairable(a, b)            # must not raise
