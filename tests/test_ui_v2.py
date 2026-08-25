"""v2 at the window level: one switch has to move every panel together.

The point of the FieldView layer is that the map cannot read degC while the readout reads
K, so these tests check the surfaces SIMULTANEOUSLY rather than one at a time.
"""
import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

import synth
from imsicon.ui.main import MainWindow, _finite_max, _finite_min


def settle(app, ms=40, rounds=5):
    for _ in range(rounds):
        app.processEvents()
        QTest.qWait(ms)
        app.processEvents()


def open_window(qapp, path):
    w = MainWindow(str(path))
    w.show()
    settle(qapp)
    if w.scan is not None and w.scan.isRunning():
        w.scan.wait(5000)
        w._on_scan_done(w.ds.value_range)
    settle(qapp)
    assert w.ds is not None
    w.select_point(2, 2)
    settle(qapp)
    return w


@pytest.fixture
def temp_window(qapp, tmp_path):
    w = open_window(qapp, synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
    yield w
    w.close()


@pytest.fixture
def precip_window(qapp, tmp_path):
    path, hourly = synth.accumulated_precip(tmp_path / 'ICON_ENS_2026082300_TOT_PREC.nc')
    w = open_window(qapp, path)
    w.hourly = hourly
    yield w
    w.close()


def surfaces(w):
    """Everything that has to agree about units, read at one instant."""
    axis = w.plot.getAxis('left')
    return {'title': w.map.plot.titleLabel.text,
            'colorbar': tuple(round(float(v), 3) for v in w.map.cbar.levels()),
            'y_axis': axis.labelUnits,
            'readout': w.readout.values['mean'].text(),
            'summary': w.status_left.text(),
            'units': w.ds.units}


# ---- A: units --------------------------------------------------------------------
def test_a_temperature_file_opens_in_celsius(temp_window):
    state = surfaces(temp_window)
    assert state['units'] == '°C'
    assert '[°C]' in state['title'] and '[°C]' in state['summary']
    assert state['y_axis'] == '°C' and state['readout'].endswith('°C')
    assert -60.0 < state['colorbar'][0] < 60.0          # plainly not Kelvin


def test_switching_units_moves_every_surface_together(temp_window, qapp):
    before = surfaces(temp_window)
    temp_window.units_combo.setCurrentText('K')
    settle(qapp)
    after = surfaces(temp_window)
    assert after['units'] == 'K'
    assert '[K]' in after['title'] and '[K]' in after['summary']
    assert after['y_axis'] == 'K' and after['readout'].endswith('K')
    assert after['colorbar'] == pytest.approx(tuple(v + 273.15 for v in before['colorbar']),
                                              abs=1e-2)
    assert float(after['readout'].split()[0]) == pytest.approx(
        float(before['readout'].split()[0]) + 273.15, abs=1e-2)


def test_the_hover_status_bar_uses_the_same_units(temp_window, qapp):
    temp_window._on_map_cursor(28.05, 33.05, 21.5)
    assert temp_window.status_right.text().endswith('°C')
    temp_window.units_combo.setCurrentText('K')
    settle(qapp)
    temp_window._on_map_cursor(28.05, 33.05, 294.65)
    assert temp_window.status_right.text().endswith('K')


def test_the_graph_data_itself_is_converted_not_just_the_label(temp_window, qapp):
    celsius = temp_window.plot.series.copy()
    temp_window.units_combo.setCurrentText('K')
    settle(qapp)
    assert np.allclose(temp_window.plot.series, celsius + 273.15, atol=1e-3)


def test_a_field_with_no_conversion_disables_the_combo(qapp, tmp_path):
    path = synth.cloud(tmp_path / 'ICON_ENS_2026082300_CLCT.nc', encoding='zero')
    w = open_window(qapp, path)
    try:
        assert not w.units_combo.isEnabled()             # G22: undecidable, so refuse
        assert w.units_warning.isVisible()
        assert 'cannot be decided' in w.units_warning.toolTip()
    finally:
        w.close()


def test_a_unit_choice_is_remembered_per_field(temp_window, qapp, tmp_path):
    temp_window.units_combo.setCurrentText('°F')
    settle(qapp)
    assert temp_window.settings.value('units/T_2M') == '°F'


# ---- C: rate ---------------------------------------------------------------------
def test_the_rate_control_is_disabled_for_an_instantaneous_field(temp_window):
    assert not temp_window.rate_combo.isEnabled()
    assert 'not an accumulated field' in temp_window.rate_combo.toolTip()


def test_the_rate_control_is_enabled_for_precipitation(precip_window):
    assert precip_window.rate_combo.isEnabled()
    assert [precip_window.rate_combo.itemText(i)
            for i in range(precip_window.rate_combo.count())] == ['as stored', '1 h', '3 h']


def test_switching_to_a_rate_jumps_past_the_window_edge(precip_window, qapp):
    """V2.4.4: landing on a blank map would read as a broken app."""
    precip_window.set_time(0)
    precip_window.rate_combo.setCurrentText('3 h')
    settle(qapp)
    assert precip_window.t == precip_window.ds.window_steps == 3
    frame = precip_window.map._frame
    assert np.isfinite(frame).all()
    assert precip_window.slider.value() == 3


def test_a_time_already_past_the_edge_is_left_alone(precip_window, qapp):
    precip_window.set_time(5)
    precip_window.rate_combo.setCurrentText('1 h')
    settle(qapp)
    assert precip_window.t == 5


def test_the_rate_moves_the_graph_the_axis_and_the_readout(precip_window, qapp):
    precip_window.rate_combo.setCurrentText('1 h')
    settle(qapp)
    assert precip_window.ds.units == 'mm h-1'
    assert precip_window.plot.getAxis('left').labelUnits == 'mm h-1'
    assert precip_window.readout.values['mean'].text().endswith('mm h-1')
    assert '[mm h-1]' in precip_window.map.plot.titleLabel.text
    series = precip_window.plot.series[:, 0]
    assert np.isnan(series[0])                                   # the gap
    assert np.allclose(series[1:], precip_window.hourly[1:], atol=1e-4)


def test_the_readout_says_which_window_the_number_covers(precip_window, qapp):
    precip_window.rate_combo.setCurrentText('3 h')
    settle(qapp)
    precip_window.plot.hovered.emit(4)
    settle(qapp)
    assert '3 h to' in precip_window.readout.values['time'].text()


# ---- G20 --------------------------------------------------------------------------
def test_finite_helpers_survive_an_all_nan_frame():
    """`float(np.nanmax(f)) or 1.0` is nan, and `nan <= lo` is False, so nan used to
    reach cbar.setLevels(high=nan)."""
    blank = np.full((4, 5), np.nan)
    assert bool(float('nan')) is True          # the reason `nan or 1.0` returns nan
    assert not (float('nan') <= 0.0)           # the reason the downstream guard missed it
    assert _finite_max(blank, 1.0) == 1.0
    assert _finite_min(blank, 0.0) == 0.0
    assert _finite_max(np.array([np.nan, 3.0]), 1.0) == 3.0
    assert _finite_min(np.array([np.nan, 3.0]), 0.0) == 3.0


def test_an_all_nan_frame_never_reaches_the_colorbar(precip_window, qapp):
    """The window edge really does produce an all-NaN frame, so this is not theoretical."""
    precip_window.ds.set_rate(3)
    precip_window.scale_combo.setCurrentText('This frame')
    precip_window.t = 0
    precip_window.refresh_map()
    settle(qapp)
    assert np.isnan(precip_window.map._frame).all()
    lo, hi = precip_window.map.cbar.levels()
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
