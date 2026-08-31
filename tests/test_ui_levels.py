"""v7 at the window level: the up/down control walks pressure levels, and says which.

The request this release answers is "when downloading IMS ICON maps, rather than
ensembles, the up and down button will move through pressure levels, and instead of single
members, it will say the correct pressure level" -- so these tests are on the real widgets:
what the control lists, what the keys do, what the title and the readout say, and that an
ensemble file is left exactly as it was.
"""
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

import synth
from imsicon import ingest, levels, products
from imsicon.ui import downloaddialog
from imsicon.ui.main import MainWindow

ERRORS = []


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """G30 and G38: an unpatched modal dialog wedges the offscreen run with no failure to
    point at, and `MainWindow._error` is reached from a worker signal. Collecting the text
    turns a wedged run into an assertion that names the cause."""
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(downloaddialog.download, 'stored_credentials', lambda: None)
    monkeypatch.setattr(downloaddialog.download, 'keyring_module', lambda: None)
    ERRORS.clear()
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical',
                        staticmethod(lambda parent, title, text, *a, **k:
                                     (ERRORS.append(text),
                                      QtWidgets.QMessageBox.StandardButton.Ok)[1]))


@pytest.fixture(autouse=True)
def only_this_test_s_files(monkeypatch):
    """G25's reasoning: discovery must not see the developer's own ./data or caches."""
    def beside_the_file(near=None):
        if near is None:
            return []
        near = Path(near)
        return [near if near.is_dir() else near.parent]

    monkeypatch.setattr(ingest, 'search_roots', beside_the_file)


def settle(app, ms=30, rounds=5):
    for _ in range(rounds):
        app.processEvents()
        QTest.qWait(ms)
        app.processEvents()


def open_window(app, path):
    window = MainWindow(str(path))
    window.show()
    settle(app)
    if window.scan is not None and window.scan.isRunning():
        window.scan.wait(5000)
        window._on_scan_done(window.ds.value_range)
    settle(app)
    assert window.ds is not None
    assert not ERRORS, ERRORS[0]
    return window


@pytest.fixture
def column(qapp, tmp_path):
    """A temperature on the five fixture levels: 1000, 925, 850, 700, 500 hPa."""
    synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc')
    window = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    yield window
    window.close()


@pytest.fixture
def ensemble(qapp, tmp_path):
    synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    window = open_window(qapp, tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    yield window
    window.close()


def labels_in(combo):
    return [combo.itemText(i) for i in range(combo.count())]


# ---- what the control says ---------------------------------------------------------------
def test_the_picker_names_the_pressure_levels_not_the_members(column):
    assert labels_in(column.level_combo) == ['500 hPa', '700 hPa', '850 hPa',
                                             '925 hPa', '1000 hPa']
    assert column.level_title.text().strip() == 'Level:'
    assert column.level_combo.currentText() == '850 hPa'


def test_the_list_reads_like_the_atmosphere_top_first(column):
    """So that "up" means up the list AND up the column, instead of fighting the combo."""
    shown = labels_in(column.level_combo)
    pressures = [float(text.split()[0]) for text in shown]
    assert pressures == sorted(pressures)          # 500 first, 1000 last


def test_the_map_title_says_which_level_it_is_showing(column):
    title = column.map.plot.titleLabel.text
    assert '850 hPa' in title and 'member' not in title.lower()


def test_the_status_bar_counts_levels_not_members(column):
    assert '5 pressure levels' in column.status_left.text()
    assert '5 pressure levels' in column.readout.subtitle.text()


# ---- what the up and down buttons do ------------------------------------------------------
def test_up_goes_higher_into_the_atmosphere_and_down_goes_lower(column, qapp):
    column.step_level(1)
    settle(qapp)
    assert column.level_combo.currentText() == '700 hPa'
    assert '700 hPa' in column.map.plot.titleLabel.text
    column.step_level(-1)
    column.step_level(-1)
    settle(qapp)
    assert column.level_combo.currentText() == '925 hPa'


def test_the_two_toolbar_buttons_do_the_same_thing_as_the_keys(column, qapp):
    QTest.mouseClick(column.level_up, QtCore.Qt.MouseButton.LeftButton)
    settle(qapp)
    assert column.level_combo.currentText() == '700 hPa'
    QTest.mouseClick(column.level_down, QtCore.Qt.MouseButton.LeftButton)
    settle(qapp)
    assert column.level_combo.currentText() == '850 hPa'


def test_the_map_actually_changes_to_that_levels_data(column, qapp):
    at_850 = np.array(column.map.img.image, copy=True)
    column.step_level(1)
    settle(qapp)
    at_700 = np.array(column.map.img.image, copy=True)
    assert not np.allclose(at_850, at_700)
    expected = column.ds.frame(column.t, column.ds.axis.nearest(700))
    assert np.allclose(at_700, expected, atol=1e-4)


def test_stepping_stops_at_the_top_and_the_bottom(column, qapp):
    for _ in range(10):
        column.step_level(1)
    settle(qapp)
    assert column.level_combo.currentText() == '500 hPa'
    for _ in range(10):
        column.step_level(-1)
    settle(qapp)
    assert column.level_combo.currentText() == '1000 hPa'


def test_choosing_from_the_list_moves_the_map_too(column, qapp):
    column.level_combo.setCurrentIndex(labels_in(column.level_combo).index('1000 hPa'))
    settle(qapp)
    assert column.level == column.ds.axis.nearest(1000)
    assert '1000 hPa' in column.map.plot.titleLabel.text


# ---- what may not be computed across a column ----------------------------------------------
def test_a_column_is_not_offered_an_ensemble_mean(column):
    keys = [column.agg_combo.itemData(i) for i in range(column.agg_combo.count())]
    assert keys == ['member']
    assert not column.agg_combo.isEnabled()
    assert 'pressure level' in column.agg_combo.toolTip()


def test_the_graph_drops_the_mean_and_the_envelope_for_a_column(column):
    assert not column.plot.mean_curve.isVisible()
    assert not column.plot.envelope.isVisible()
    assert len(column.plot._curves) == column.ds.n_members


def test_the_selected_level_is_the_heavy_curve(column, qapp):
    def widths():
        return [curve.opts['pen'].widthF() for curve in column.plot._curves]

    heavy = int(np.argmax(widths()))
    assert heavy == column.level
    column.step_level(1)
    settle(qapp)
    assert int(np.argmax(widths())) == column.level


def test_the_readout_reports_the_level_rather_than_ensemble_statistics(column, qapp):
    column.select_point(1, 2)
    settle(qapp)
    assert column.readout.mode == 'level'
    assert set(column.readout.values) == {'time', 'level', 'value', 'max', 'min'}
    assert column.readout.values['level'].text() == '850 hPa'
    at_level = column.ds.series(1, 2)[column.t, column.level]
    assert f'{at_level:,.2f}' in column.readout.values['value'].text()
    # The extremes of the column say where they are, which is the point of a profile.
    assert 'hPa' in column.readout.values['max'].text()


def test_the_readout_follows_the_level_as_it_moves(column, qapp):
    column.select_point(1, 2)
    settle(qapp)
    before = column.readout.values['value'].text()
    column.step_level(1)
    settle(qapp)
    assert column.readout.values['level'].text() == '700 hPa'
    assert column.readout.values['value'].text() != before


# ---- a surface field of the same product ---------------------------------------------------
def test_a_surface_field_has_nothing_to_choose_and_says_so(qapp, tmp_path):
    synth.surface_field(tmp_path / 'IE_2026083100_t_2m.nc')
    window = open_window(qapp, tmp_path / 'IE_2026083100_t_2m.nc')
    try:
        assert labels_in(window.level_combo) == ['surface']
        assert not window.level_combo.isEnabled()
        assert not window.level_up.isEnabled()
        window.step_level(1)             # must not move, and must not raise
        settle(qapp)
        assert window.level == 0
        assert 'surface (single level)' in window.status_left.text()
        # One level: "highest in column" could only repeat the value above it.
        window.select_point(1, 2)
        settle(qapp)
        assert window.readout.values['value'].isVisible()
        assert not window.readout.values['max'].isVisible()
    finally:
        window.close()


def test_the_deterministic_run_lists_its_own_maps(qapp, tmp_path):
    synth.icon_run(tmp_path, fields=('temp', 't_2m', 'rh_2m'))
    synth.temperature(tmp_path / 'ICON_ENS_2026083100_T_2M.nc')     # another product
    window = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    try:
        keys = [window.field_combo.itemData(i)
                for i in range(window.field_combo.count())]
        assert 'base' in keys and 'file:t_2m' in keys and 'file:rh_2m' in keys
        # The ensemble file shares the run id and is NOT offered: different product,
        # different axis, and `check_pairable` would refuse it anyway.
        assert not any(key.endswith('T_2M') for key in keys)
        labels = [window.field_combo.itemText(i)
                  for i in range(window.field_combo.count())]
        assert any(label.startswith('temp - Temperature - on pressure levels')
                   for label in labels)
    finally:
        window.close()


def test_switching_to_a_surface_map_of_the_same_run_reconfigures_the_control(qapp, tmp_path):
    """20 levels to none, in one window: the picker must not keep the old file's shape."""
    synth.icon_run(tmp_path, fields=('temp', 't_2m'))
    window = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    try:
        keys = [window.field_combo.itemData(i)
                for i in range(window.field_combo.count())]
        window.field_combo.setCurrentIndex(keys.index('file:t_2m'))
        settle(qapp)
        for worker in (window.decompressor, window.builder):
            if worker is not None and worker.isRunning():
                worker.wait(20000)
        settle(qapp)
        assert window.ds.field == 't_2m'
        assert labels_in(window.level_combo) == ['surface']
        assert not window.level_combo.isEnabled()
        assert window.level == 0
        assert not ERRORS, ERRORS[0]
    finally:
        window.close()


# ---- the ensemble is untouched ---------------------------------------------------------------
def test_an_ensemble_file_still_gets_its_six_aggregations_and_member_names(ensemble):
    keys = [ensemble.agg_combo.itemData(i) for i in range(ensemble.agg_combo.count())]
    assert keys == ['mean', 'max', 'min', 'median', 'spread', 'member']
    assert ensemble.agg_combo.isEnabled()
    assert labels_in(ensemble.level_combo)[0].startswith('member')
    assert ensemble.level_title.text().strip() == 'Member:'
    assert ensemble.plot.mean_curve.isVisible()
    assert ensemble.readout.mode == 'member'
    assert 'mean' in ensemble.readout.values


def test_up_on_an_ensemble_shows_a_single_member_rather_than_doing_nothing(ensemble, qapp):
    assert ensemble.agg_combo.currentData() == 'mean'
    ensemble.step_level(1)
    settle(qapp)
    assert ensemble.agg_combo.currentData() == 'member'
    assert ensemble.level == 1
    assert ensemble.map.plot.titleLabel.text.count('member') >= 1


def test_switching_from_a_column_to_an_ensemble_puts_everything_back(qapp, tmp_path):
    """One window, two products: the controls must not carry the previous file's shape."""
    synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc')
    synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    window = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    try:
        assert window.readout.mode == 'level'
        window.open_path(str(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
        settle(qapp)
        assert window.ds.axis.kind == 'member'
        assert window.agg_combo.count() == len(
            [k for k in ('mean', 'max', 'min', 'median', 'spread', 'member')])
        assert window.agg_combo.isEnabled()
        assert window.readout.mode == 'member'
        assert window.plot.mean_curve.isVisible()
        assert labels_in(window.level_combo)[0].startswith('member')
        assert not ERRORS, ERRORS[0]
    finally:
        window.close()


# ---- the levels the file could not name ------------------------------------------------------
def test_a_file_whose_levels_were_assumed_says_so_in_the_status_bar(qapp, tmp_path):
    from imsicon import ncwrite
    data = np.random.default_rng(2).normal(
        250, 10, (3, len(products.PRESSURE_LEVELS), 4, 5)).astype(np.float32)
    ncwrite.write_nc3(tmp_path / 'IE_2026083100_temp.nc', 'temp', 'K', data,
                      variable='temp')
    window = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    try:
        assert labels_in(window.level_combo)[0] == '150 hPa'
        assert window.status_right.text() == '⚠ levels assumed'
        assert 'manual' in window.status_right.toolTip()
    finally:
        window.close()


def test_the_default_level_is_the_low_level_chart(qapp, tmp_path):
    synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc',
                         levels=products.PRESSURE_LEVELS,
                         values=np.zeros((4, len(products.PRESSURE_LEVELS), 4, 5)))
    window = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    try:
        assert window.level_combo.currentText() == '850 hPa'
        assert window.ds.axis.values[window.level] == levels.DEFAULT_LEVEL_HPA
    finally:
        window.close()


# ---- the downloader offers the second product ------------------------------------------
def listing_for(fields, run='2026083100'):
    return ''.join(f'<a href="IE_{run}_{f}.nc.bz2">IE_{run}_{f}.nc.bz2</a>  120000\n'
                   for f in fields)


def test_the_download_dialog_offers_the_deterministic_product(qapp, monkeypatch):
    monkeypatch.setattr(ingest, 'search_roots', lambda near=None: [])
    dialog = downloaddialog.DownloadDialog(None)
    try:
        keys = [dialog.family_combo.itemData(i)
                for i in range(dialog.family_combo.count())]
        assert keys == ['ens', 'icon']
        dialog.family_combo.setCurrentIndex(keys.index('icon'))
        assert dialog.family is products.ICON
        files = downloaddialog.download.parse_listing(
            listing_for(('temp', 't_2m', 'u_10m', 'v_10m')), products.ICON)
        dialog._on_listed((object(), files, products.ICON))
        rows = {dialog.field_list.topLevelItem(i).text(1):
                dialog.field_list.topLevelItem(i).text(2)
                for i in range(dialog.field_list.topLevelItemCount())}
        # Which maps give the viewer a level to step through, said before the download.
        assert rows['temp'] == '22 pressure levels'
        assert rows['t_2m'] == 'surface'
        assert dialog.field_list.topLevelItem(0).text(0).startswith('Temperature')
        dialog._select_wind()
        assert {entry.field for entry in dialog.selected()} == {'u_10m', 'v_10m'}
    finally:
        dialog.close()


def test_a_deterministic_download_lands_in_the_right_place(qapp, monkeypatch):
    """G27 again: the local name is rebuilt from the run and field, never echoed."""
    monkeypatch.setattr(ingest, 'search_roots', lambda near=None: [])
    files = downloaddialog.download.parse_listing(listing_for(('temp',)), products.ICON)
    assert files[0].local_name() == 'IE_2026083100_temp.nc.bz2'
    assert files[0].url.startswith(downloaddialog.download.base_url(products.ICON))
