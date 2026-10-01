"""The settings file at the window level (R10): what a map OPENS with, and the dialog.

Every assertion here is about the default giving way to the reader: the file decides how
a map opens, the toolbar still changes all of it, and a choice made there is not undone
by the file on the next field switch.
"""
import numpy as np
import pytest
from PySide6 import QtGui, QtWidgets

import synth
from imsicon import config, ingest
from imsicon.ncwrite import write_nc3
from imsicon.ui import downloaddialog, settingsdialog
from imsicon.ui.main import SCALE_DATASET, SCALE_FIXED, SCALE_FRAME, MainWindow


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """G30/G38: no modal dialog may open in the offscreen run, and the download dialog
    must never read the developer's keychain."""
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(downloaddialog.download, 'keyring_module', lambda: None)
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical',
                        staticmethod(lambda *a, **k: errors.append(a[2] if len(a) > 2
                                                                   else a)))
    yield
    assert errors == []


@pytest.fixture(autouse=True)
def only_this_test_s_files(monkeypatch):
    from pathlib import Path

    def beside_the_file(near=None):
        if near is None:
            return []
        near = Path(near)
        return [near if near.is_dir() else near.parent]
    monkeypatch.setattr(ingest, 'search_roots', beside_the_file)


def settle(app, ms=30, rounds=5):
    from PySide6.QtTest import QTest
    for _ in range(rounds):
        app.processEvents()
        QTest.qWait(ms)
        app.processEvents()


def finish(window, app):
    settle(app)
    for worker in (window.decompressor, window.builder, window._height_builder):
        if worker is not None and worker.isRunning():
            worker.wait(30000)
    settle(app)
    if window.scan is not None and window.scan.isRunning():
        window.scan.wait(5000)
        window._on_scan_done(window.ds.value_range)
    settle(app)


def choose(window, app, key):
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    assert key in keys, keys
    window.field_combo.setCurrentIndex(keys.index(key))
    finish(window, app)
    return window.ds


def configure(path, text):
    path.write_text(text, encoding='utf-8')
    return config.load(path)


@pytest.fixture
def run_dir(tmp_path):
    """T_2M across ~15..34 degC and RELHUM_2M across 45..99 %, one ensemble run."""
    nt, nm, ny, nx = 4, 3, 24, 18
    step = np.arange(nt)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    warm = 288.0 + 0.55 * y + 0.35 * x + 1.5 * step
    moist = np.clip(99.0 - 2.6 * x - 0.2 * y - 1.0 * step, 45.0, 100.0)

    def members(base, spread):
        return np.stack([np.broadcast_to(base, (nt, ny, nx)) + spread * (m - 1)
                         for m in range(nm)], axis=1)

    write_nc3(tmp_path / 'ICON_ENS_2026082300_T_2M.nc', 'T_2M', 'K', members(warm, 0.4),
              history=synth.HISTORY_TEMPLATE.format(field='T_2M'),
              long_name='2m temperature', standard_name='air_temperature')
    write_nc3(tmp_path / 'ICON_ENS_2026082300_RELHUM_2M.nc', 'RELHUM_2M', '%',
              np.clip(members(moist, 1.5), 1.0, 100.0),
              history=synth.HISTORY_TEMPLATE.format(field='RELHUM_2M'),
              long_name='relative humidity in 2m', standard_name='relative_humidity')
    return tmp_path


def open_window(app, path):
    w = MainWindow(str(path))
    w.resize(1200, 800)
    w.show()
    finish(w, app)
    return w


T2M = 'ICON_ENS_2026082300_T_2M.nc'
SETTINGS = '''
[map]
points = ["31.1234, 34.1234, red, Somewhere", "30.0, 35.0, "]

[fields.T_2M]
units = "F"
colours = "viridis"
scale = [10, 40]
scale_units = "C"
isolines = false
isoline_step = 2

[fields.RELHUM_2M]
isoline_step = 10
colours = "CET-L17"
'''


@pytest.fixture
def window(qapp, run_dir, isolated_config):
    configure(isolated_config, SETTINGS)
    w = open_window(qapp, run_dir / T2M)
    yield w
    w.close()


# ---- a map opens as the file says -------------------------------------------------------
def test_a_map_opens_in_the_configured_units_colours_and_scale(window):
    assert window.ds.units == '°F'
    assert window.units_combo.currentText() == '°F'
    assert window.cmap_combo.currentText() == 'viridis'
    assert window.scale_combo.currentIndex() == SCALE_FIXED
    # [10, 40] degC is [50, 104] degF: the same colours whatever the Units combo says.
    assert window.map.cbar.levels() == pytest.approx((50.0, 104.0))
    assert '50 to 104 °F' in window.scale_combo.itemText(SCALE_FIXED)


def test_the_fixed_range_is_also_the_graph_s_scale(window):
    lo, hi = window.plot.getPlotItem().vb.viewRange()[1]
    assert lo == pytest.approx(50.0, abs=3.0) and hi == pytest.approx(104.0, abs=3.0)


def test_the_fixed_range_follows_a_units_change(window, qapp):
    window.units_combo.setCurrentText('K')
    settle(qapp)
    assert window.scale_combo.currentIndex() == SCALE_FIXED
    assert window.map.cbar.levels() == pytest.approx((283.15, 313.15))


def test_the_scale_can_still_be_changed_in_the_app(window, qapp):
    window.scale_combo.setCurrentIndex(SCALE_DATASET)
    settle(qapp)
    lo, hi = window.map.cbar.levels()
    assert (lo, hi) != pytest.approx((50.0, 104.0))
    lo_ds, hi_ds = window.ds.value_range
    assert (lo, hi) == pytest.approx((lo_ds, hi_ds))


def test_isolines_open_off_and_come_on_at_the_configured_spacing(window, qapp):
    assert window.isolines_check.isEnabled() and not window.isolines_check.isChecked()
    assert window.map.isoline_levels.size == 0
    window.isolines_check.setChecked(True)
    settle(qapp)
    # 2 degC, said in the units on screen (G15): 3.6 degF.
    assert window.ds.isoline_step == pytest.approx(2.0)
    assert window.map.isoline_step == pytest.approx(3.6)
    assert window.isoline_step_label.text() == '3.6 °F'


def test_a_choice_made_in_the_app_survives_a_field_switch(window, qapp):
    window.isolines_check.setChecked(True)
    window.units_combo.setCurrentText('°C')
    settle(qapp)
    choose(window, qapp, 'file:RELHUM_2M')
    choose(window, qapp, 'file:T_2M')
    assert window.isolines_check.isChecked()           # not reset to the file's "off"
    assert window.ds.units == '°C'


def test_a_spacing_makes_a_map_the_app_does_not_contour_contourable(window, qapp):
    ds = choose(window, qapp, 'file:RELHUM_2M')
    assert window.isolines_check.isEnabled()
    interval = ds.isolines
    assert interval is not None and interval.step == pytest.approx(10.0)
    assert window.map.isoline_levels.size > 0
    assert np.allclose(window.map.isoline_levels % 10.0, 0.0)
    assert window.cmap_combo.currentText() == 'CET-L17'
    # a field with no fixed range leaves the Fixed entry out of reach
    assert window.scale_combo.currentIndex() != SCALE_FIXED
    assert not window.scale_combo.model().item(SCALE_FIXED).isEnabled()


# ---- points -------------------------------------------------------------------------------
def test_configured_points_are_drawn_as_dots_and_can_be_hidden(window, qapp):
    xs, ys = window.map.points.getData()
    assert list(zip(ys, xs)) == [(31.1234, 34.1234), (30.0, 35.0)]
    brushes = [spot.brush().color().name() for spot in window.map.points.points()]
    assert brushes == [QtGui.QColor('red').name(), QtGui.QColor('blue').name()]
    assert [label.toPlainText() for label in window.map.point_labels] == ['Somewhere']
    assert window.points_check.isEnabled() and window.points_check.isChecked()
    window.points_check.setChecked(False)
    settle(qapp)
    assert not window.map.points.isVisible()
    assert not window.map.point_labels[0].isVisible()


def test_no_points_leaves_the_toggle_out_of_reach(qapp, run_dir, isolated_config):
    configure(isolated_config, '[map]\npoints = []\n')
    w = open_window(qapp, run_dir / T2M)
    try:
        assert not w.points_check.isEnabled()
        assert len(w.map.points.data) == 0
    finally:
        w.close()


def test_a_broken_settings_file_still_opens_the_map(qapp, run_dir, isolated_config):
    isolated_config.write_text('[fields.T_2M\nunits = "F"', encoding='utf-8')
    w = open_window(qapp, run_dir / T2M)
    try:
        assert w.ds is not None and w.ds.units == '°C'        # the app's own default
        assert w.config.unreadable and w.config.warnings
    finally:
        w.close()


def test_an_unknown_ramp_is_reported_and_the_default_used(qapp, run_dir, isolated_config):
    configure(isolated_config, '[fields.T_2M]\ncolours = "rainbow"\n')
    w = open_window(qapp, run_dir / T2M)
    try:
        assert w.cmap_combo.currentText() == 'turbo'
        assert any('rainbow' in problem for problem in w.config.warnings)
    finally:
        w.close()


# ---- profile, on the deterministic run ---------------------------------------------------
def test_the_profile_default_comes_from_the_file(qapp, tmp_path, isolated_config):
    synth.icon_run(tmp_path, fields=('temp', 'rh'))
    configure(isolated_config, '[fields.temp]\nprofile = true\n[fields.rh]\nprofile = false\n')
    w = open_window(qapp, tmp_path / 'IE_2026083100_temp.nc')
    try:
        assert w.profile_check.isChecked() and w.profile_shown
        choose(w, qapp, 'file:rh')
        assert not w.profile_check.isChecked() and not w.profile_shown
    finally:
        w.close()


# ---- the dialog ---------------------------------------------------------------------------
@pytest.fixture
def dialog(qapp, isolated_config):
    configure(isolated_config, SETTINGS + '\n[credentials]\nuser = "me"\npassword = "pw"\n')
    d = settingsdialog.SettingsDialog(config.load(isolated_config))
    yield d
    d.close()


def test_the_dialog_shows_what_the_file_holds(dialog):
    assert dialog.user_edit.text() == 'me' and dialog.password_edit.text() == 'pw'
    assert dialog.password_edit.echoMode() == QtWidgets.QLineEdit.EchoMode.Password
    assert dialog.points_table.rowCount() == 2
    assert dialog.points_table.item(1, 2).text() == 'blue'
    assert dialog.fields_table.rowCount() == 2
    cell = dialog.fields_table.cellWidget
    assert cell(0, settingsdialog.COL['Map']).currentText() == 'T_2M'
    assert cell(0, settingsdialog.COL['Units']).currentText() == '°F'
    assert cell(0, settingsdialog.COL['Scale']).currentText() == 'Fixed range'
    assert cell(0, settingsdialog.COL['Min']).text() == '10'


def test_saving_the_dialog_round_trips_the_file(dialog, isolated_config):
    before = config.load(isolated_config)
    saved = dialog.save()
    assert saved is not None and saved.warnings == []
    assert saved.points == before.points
    assert saved.fields == before.fields
    assert saved.credentials() == ('me', 'pw')


def test_a_new_point_and_a_new_map_default_are_saved(dialog, isolated_config):
    row = dialog.add_point()
    for col, text in enumerate(('32.5', '35.1', '', 'Afula')):
        dialog.points_table.item(row, col).setText(text)
    row = dialog.add_field_row()
    cell = dialog.fields_table.cellWidget
    cell(row, settingsdialog.COL['Map']).setCurrentText('CAPE_ML')
    cell(row, settingsdialog.COL['Colours']).setCurrentText('inferno')
    cell(row, settingsdialog.COL['Scale']).setCurrentText('Fixed range')
    cell(row, settingsdialog.COL['Min']).setText('0')
    cell(row, settingsdialog.COL['Max']).setText('3000')
    assert dialog.save() is not None
    back = config.load(isolated_config)
    assert back.points[-1] == config.Point(32.5, 35.1, 'blue', 'Afula')
    cape = back.for_names('CAPE_ML')
    assert cape.colours == 'inferno' and cape.fixed_range == (0.0, 3000.0)


def test_a_range_from_zero_keeps_its_zero(qapp, isolated_config):
    configure(isolated_config, '[fields.CAPE_ML]\nscale = [0, 2000]\n')
    d = settingsdialog.SettingsDialog(config.load(isolated_config))
    try:
        assert d.fields_table.cellWidget(0, settingsdialog.COL['Min']).text() == '0'
        assert d.save() is not None                  # not "needs both Min and Max"
        assert config.load(isolated_config).for_names('CAPE_ML').fixed_range == (0.0, 2000.0)
    finally:
        d.close()


def test_bad_input_is_shown_in_the_dialog_and_nothing_is_written(dialog, isolated_config):
    before = isolated_config.read_text(encoding='utf-8')
    row = dialog.add_point()
    dialog.points_table.item(row, 0).setText('91')
    dialog.points_table.item(row, 1).setText('35')
    other = dialog.add_point()
    dialog.points_table.item(other, 0).setText('31')
    dialog.points_table.item(other, 1).setText('35')
    dialog.points_table.item(other, 2).setText('notacolour')
    assert dialog.save() is None
    assert 'latitude' in dialog.error.text() and 'notacolour' in dialog.error.text()
    assert isolated_config.read_text(encoding='utf-8') == before


def test_a_saved_configuration_applies_to_the_open_window(window, qapp, isolated_config):
    d = settingsdialog.SettingsDialog(window.config, current=window.ds)
    d.fields_table.cellWidget(0, settingsdialog.COL['Colours']).setCurrentText('magma')
    d.fields_table.cellWidget(0, settingsdialog.COL['Scale']).setCurrentText('This frame')
    d.points_table.removeRow(0)
    saved = d.save()
    d.close()
    window.apply_config(saved)
    settle(qapp)
    assert window.cmap_combo.currentText() == 'magma'
    assert window.scale_combo.currentIndex() == SCALE_FRAME
    assert len(window.map.points.data) == 1
