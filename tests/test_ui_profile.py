"""R9 on the deterministic run, on the real widgets: the profile, the height, the readout.

Three requests, one column: the date and the value made more visible, the geopotential
height of the level shown put next to them, and -- for relative humidity -- the right-hand
panel drawn as a vertical profile (value across, height up) instead of a time graph.
"""
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

import synth
from imsicon import derived, ingest, isolines, levels
from imsicon.ui import downloaddialog, readout
from imsicon.ui.main import MainWindow

ERRORS = []
G0 = isolines.G0
LEVELS = (1000, 925, 850, 700, 500)
RUN = '2026083100'


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
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


def open_window(app, path):
    window = MainWindow(str(path))
    window.resize(1300, 800)
    window.show()
    settle(app)
    if window.scan is not None and window.scan.isRunning():
        window.scan.wait(5000)
        window._on_scan_done(window.ds.value_range)
    wait_heights(window, app)
    assert window.ds is not None and not ERRORS, ERRORS
    return window


def wait_heights(window, app):
    if window._height_builder is not None and window._height_builder.isRunning():
        window._height_builder.wait(30000)
    settle(app)


# ---- fixtures: one deterministic run -------------------------------------------------------
def column_values(kind, nt=4, ny=4, nx=5, lvls=LEVELS):
    t = np.arange(nt)[:, None, None, None]
    p = np.asarray(lvls, float)[None, :, None, None]
    y = np.arange(ny)[None, None, :, None]
    x = np.arange(nx)[None, None, None, :]
    if kind == 'rh':
        values = np.clip(95.0 - 0.07 * (1000.0 - p) + 3.0 * y - 2.0 * x - 4.0 * t, 5, 100)
    elif kind == 'geopot':
        # hypsometric-ish heights in gpm, in m2 s-2, varying a little with place and time
        values = (8000.0 * np.log(1000.0 / p) + 10.0 * y + 5.0 * x + 7.0 * t) * G0
    else:
        values = 300.0 - 0.06 * (1000.0 - p) + 0.3 * y + 0.2 * x + 1.0 * t
    return np.broadcast_to(values, (nt, len(lvls), ny, nx))


def write_run(tmp_path, fields=('rh', 'geopot', 'temp'), run=RUN, geopot_steps=None,
              nt=4):
    units = {'rh': '%', 'geopot': 'm2 s-2', 'temp': 'K'}
    for field in fields:
        steps = geopot_steps if (field == 'geopot' and geopot_steps) else nt
        synth.pressure_field(tmp_path / f'IE_{run}_{field}.nc', field, units[field],
                             n_times=steps, levels=LEVELS,
                             values=column_values(field, nt=steps), run=run)
    return tmp_path


@pytest.fixture
def run_dir(tmp_path):
    return write_run(tmp_path)


@pytest.fixture
def humidity(qapp, run_dir):
    w = open_window(qapp, run_dir / f'IE_{RUN}_rh.nc')
    yield w
    w.close()


@pytest.fixture
def temperature(qapp, run_dir):
    w = open_window(qapp, run_dir / f'IE_{RUN}_temp.nc')
    yield w
    w.close()


KFT = isolines.KFT


def expected_heights(window, t=None):
    """What the readout and the profile must show: the run's geopot, in kft, at the point."""
    iy, ix = window.point
    raw = window.height_companion.raw.series(iy, ix)
    kft = raw / KFT
    return kft if t is None else kft[t]


def shown(height):
    return readout.format_height(height, 'kft')


# ---- the profile ----------------------------------------------------------------------------
def test_relative_humidity_opens_as_a_profile_with_height_up_and_value_across(humidity):
    w = humidity
    assert w.profile_check.isEnabled() and w.profile_check.isChecked()
    assert w.profile_shown
    assert w.profile.by_height
    assert w.profile.getAxis('left').labelText == 'geopotential height'
    assert w.profile.getAxis('left').labelUnits == 'kft'
    assert w.profile.getAxis('bottom').labelText == 'rh'
    assert w.profile.getAxis('bottom').labelUnits == '%'
    assert not w.profile.getPlotItem().vb.yInverted()


def test_the_profile_is_the_column_at_the_point_and_time(humidity, qapp):
    w = humidity
    iy, ix = w.point
    w.set_time(2)
    settle(qapp)
    np.testing.assert_allclose(w.profile.values, w.ds.series(iy, ix)[2], rtol=1e-6)
    np.testing.assert_allclose(w.profile.ys, expected_heights(w, 2), rtol=1e-6)
    # move the point: a different column
    w.select_point(1, 3)
    settle(qapp)
    np.testing.assert_allclose(w.profile.values, w.ds.series(1, 3)[2], rtol=1e-6)
    np.testing.assert_allclose(w.profile.ys, expected_heights(w, 2), rtol=1e-6)


def test_the_right_hand_axis_names_the_levels_in_hpa(humidity):
    labels = [label for _y, label in humidity.profile.level_ticks]
    assert set(labels) <= set(humidity.ds.member_labels)
    assert '850 hPa' in labels and '500 hPa' in labels


def test_the_map_s_level_is_marked_and_the_mark_follows_up_and_down(humidity, qapp):
    w = humidity
    assert w.level_combo.currentText() == '850 hPa'
    index = w.level
    x, y = w.profile.level_marker.getData()
    assert x[0] == pytest.approx(float(w.profile.values[index]))
    assert y[0] == pytest.approx(float(w.profile.ys[index]))
    w.step_level(1)
    settle(qapp)
    assert w.level_combo.currentText() == '700 hPa'
    x, y = w.profile.level_marker.getData()
    assert y[0] == pytest.approx(float(w.profile.ys[w.level]))
    assert w.profile.level_line.value() == pytest.approx(float(w.profile.ys[w.level]))


def test_clicking_a_level_on_the_profile_moves_the_map_to_it(humidity, qapp):
    w = humidity
    target = int(np.argmin(np.abs(np.asarray(w.ds.axis.values) - 500.0)))
    w.profile.levelPicked.emit(target)
    settle(qapp)
    assert w.level == target
    assert w.level_combo.currentText() == '500 hPa'
    assert '500 hPa' in w.map.plot.titleLabel.text


def test_hovering_a_level_reports_that_level_and_leaving_restores_the_map_s(humidity, qapp):
    w = humidity
    iy, ix = w.point
    other = int(np.argmin(np.abs(np.asarray(w.ds.axis.values) - 925.0)))
    w.profile.levelHovered.emit(other)
    settle(qapp)
    assert w.readout.values['level'].text() == '925 hPa'
    assert w.readout.values['value'].text().startswith(
        readout.format_value(float(w.ds.series(iy, ix)[w.t, other]), w.readout.span, '%'))
    assert w.readout.values['height'].text() == shown(expected_heights(w, w.t)[other])
    w.profile.levelHovered.emit(-1)
    settle(qapp)
    assert w.readout.values['level'].text() == '850 hPa'


def test_the_time_slider_moves_the_profile(humidity, qapp):
    w = humidity
    before = w.profile.values.copy()
    w.slider.setValue(3)
    settle(qapp)
    assert not np.allclose(before, w.profile.values)
    np.testing.assert_allclose(w.profile.values, w.ds.series(*w.point)[3], rtol=1e-6)


def test_the_profile_s_value_axis_is_pinned_to_the_dataset_range(humidity):
    lo, hi = humidity.ds.value_range
    (x0, x1), _ = humidity.profile.getPlotItem().vb.viewRange()
    assert x0 <= min(0.0, lo) and x1 >= hi


def test_temperature_opens_on_the_time_graph_and_can_be_switched(temperature, qapp):
    w = temperature
    assert w.profile_check.isEnabled() and not w.profile_check.isChecked()
    assert not w.profile_shown
    w.profile_check.setChecked(True)
    settle(qapp)
    assert w.profile_shown and w.profile.by_height
    assert w.profile.getAxis('bottom').labelText == 'temp'
    np.testing.assert_allclose(w.profile.values, w.ds.series(*w.point)[w.t], rtol=1e-6)


def choose(w, app, field):
    """Pick a field under Map shows; the open one is keyed `base`, the rest `file:<f>`."""
    keys = [w.field_combo.itemData(i) for i in range(w.field_combo.count())]
    key = 'base' if w.base_ds.field == field else f'file:{field}'
    w.field_combo.setCurrentIndex(keys.index(key))
    settle(app)
    wait_heights(w, app)
    assert w.ds.field == field


def test_the_choice_is_kept_per_field_across_map_shows(temperature, qapp):
    w = temperature
    w.profile_check.setChecked(True)
    settle(qapp)
    choose(w, qapp, 'rh')
    assert w.profile_shown
    w.profile_check.setChecked(False)
    settle(qapp)
    choose(w, qapp, 'temp')
    assert w.profile_shown                                     # temp's own choice, kept
    choose(w, qapp, 'rh')
    assert not w.profile_shown                                 # and so was rh's


def test_an_ensemble_has_no_column_and_no_profile(qapp, tmp_path):
    synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    w = open_window(qapp, tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    try:
        assert not w.profile_check.isEnabled() and not w.profile_check.isChecked()
        assert not w.profile_shown
        assert 'not on pressure levels' in w.profile_check.toolTip()
        assert not w.readout.row_visible('height')
    finally:
        w.close()


def test_without_geopot_the_profile_is_drawn_against_pressure_and_says_so(qapp, tmp_path):
    write_run(tmp_path, fields=('rh',))
    w = open_window(qapp, tmp_path / f'IE_{RUN}_rh.nc')
    try:
        assert w.height_companion is None
        assert w.profile_shown and not w.profile.by_height
        assert w.profile.getAxis('left').labelText == 'pressure level'
        assert w.profile.getAxis('left').labelUnits == 'hPa'
        assert w.profile.getPlotItem().vb.yInverted()
        np.testing.assert_allclose(w.profile.ys, np.asarray(w.ds.axis.values))
        assert w.readout.values['height'].text() == '--'
        assert 'geopot' in w.readout.values['height'].toolTip()
    finally:
        w.close()


# ---- the height beside the value -------------------------------------------------------------
def test_the_height_row_is_the_run_s_geopotential_at_the_level_point_and_time(humidity, qapp):
    w = humidity
    assert w.readout.row_visible('height')
    assert w.height_companion is not None and w.height_companion.units == 'kft'
    for t in (0, 3):
        w.set_time(t)
        settle(qapp)
        assert w.readout.values['height'].text() == shown(expected_heights(w, t)[w.level])
        assert w.readout.values['height'].text().endswith(' kft')
    w.step_level(-1)                                       # 925 hPa: lower, so a smaller height
    settle(qapp)
    assert w.readout.values['height'].text() == shown(expected_heights(w, 3)[w.level])
    assert expected_heights(w, 3)[w.level] < expected_heights(w, 3)[w.ds.axis.nearest(850)]


def test_the_date_and_the_value_are_larger_than_the_rows_around_them(humidity):
    panel = humidity.readout
    assert panel.font_points('time') == panel.font_points('value')
    assert panel.font_points('time') >= panel.font_points('level') + readout.PROMINENT_POINTS
    assert panel.font_points('height') == panel.font_points('level')
    assert panel.values['time'].font().bold() and panel.values['value'].font().bold()


def test_the_geopotential_map_itself_does_not_repeat_its_value_as_a_height(qapp, run_dir):
    w = open_window(qapp, run_dir / f'IE_{RUN}_geopot.nc')
    try:
        assert w.height_companion is None
        assert not w.readout.row_visible('height')
        assert w.readout.row_visible('value')
        assert w.ds.units == 'kft'
    finally:
        w.close()


def test_the_height_unit_is_configurable_and_the_row_follows_the_chart(humidity, qapp):
    """The height row reads in whatever the geopot chart is set to: kft by default,
    gpm, dam or the raw m2 s-2 -- one choice, remembered, shared with the chart."""
    w = humidity
    before = w.readout.values['height'].text()
    assert before.endswith(' kft')
    w.settings.setValue('units/geopot', 'gpm')
    w.height_companion, w._height_key = None, None
    w._sync_height_companion()
    wait_heights(w, qapp)
    assert w.height_companion.units == 'gpm'
    text = w.readout.values['height'].text()
    assert text.endswith(' gpm')
    assert float(text.split()[0].replace(',', '')) == pytest.approx(
        float(before.split()[0]) * 304.8, rel=1e-3)
    assert w.profile.getAxis('left').labelUnits == 'gpm'
    # the chart itself shares the view, so a change there moves the row too
    companion = w.height_companion
    choose(w, qapp, 'geopot')
    assert w.ds is companion and w.units_combo.currentText() == 'gpm'
    w.units_combo.setCurrentText('dam')
    settle(qapp)
    choose(w, qapp, 'rh')
    assert w.readout.values['height'].text().endswith(' dam')


def test_a_surface_field_has_no_level_to_be_the_height_of(qapp, tmp_path):
    write_run(tmp_path, fields=('geopot',))
    synth.surface_field(tmp_path / f'IE_{RUN}_t_2m.nc', 't_2m', 'K', run=RUN)
    w = open_window(qapp, tmp_path / f'IE_{RUN}_t_2m.nc')
    try:
        assert w.height_companion is None
        assert not w.readout.row_visible('height')
    finally:
        w.close()


def test_a_geopotential_that_does_not_fit_the_map_is_refused_not_misread(qapp, tmp_path):
    """G45's lesson: a geopot with fewer steps than the map would index past its end."""
    write_run(tmp_path, fields=('rh', 'geopot'), geopot_steps=2)
    w = open_window(qapp, tmp_path / f'IE_{RUN}_rh.nc')
    try:
        assert w.height_companion is None
        assert w.readout.values['height'].text() == '--'
        assert 'forecast steps' in w.readout.values['height'].toolTip()
        assert not w.profile.by_height                     # drawn against pressure instead
        w.set_time(3)                                      # past the geopot's end: safe
        settle(qapp)
        assert not ERRORS
    finally:
        w.close()


def test_the_companion_is_kept_across_fields_of_the_run_and_reused_as_a_map(humidity, qapp):
    w = humidity
    companion = w.height_companion
    keys = [w.field_combo.itemData(i) for i in range(w.field_combo.count())]
    w.field_combo.setCurrentIndex(keys.index('file:temp'))
    settle(qapp)
    assert w.ds.field == 'temp'
    assert w.height_companion is companion                  # same run, same shape: kept
    assert w.readout.row_visible('height')
    w.field_combo.setCurrentIndex(keys.index('file:geopot'))
    settle(qapp)
    assert w.ds is companion                                # opened once, shown as itself
    assert w.field_combo.currentData() == 'base'


def test_a_four_digit_colorbar_fits_inside_the_map_widget(qapp, run_dir):
    """Found while building R9: pyqtgraph pins the colorbar axis to 45 px, and G36's
    title margin was 110 px, so on a scale over 999 the layout overflowed the widget by
    20 px and every label lost its last digit -- a 500 hPa height read 552 for 5,520 gpm.
    """
    w = open_window(qapp, run_dir / f'IE_{RUN}_geopot.nc')
    try:
        w.resize(1500, 880)
        w.units_combo.setCurrentText('gpm')                        # a four-digit scale
        settle(qapp)
        m = w.map
        assert m.cbar.axis.fixedWidth is None                   # sizes itself to its text
        assert m.ci.geometry().width() <= m.width() + 0.5
        assert m.cbar.geometry().right() <= m.width()
        lo, hi = m.cbar.levels()
        assert hi > 999                                           # a four-digit scale
    finally:
        w.close()
