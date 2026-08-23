"""Feature-level tests: F1 (open at startup), F3 (click the map), F4 (hover the graph)."""
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from imsicon.dataset import member_stats
from imsicon.ui.main import MainWindow


def settle(app, ms=60, rounds=6):
    for _ in range(rounds):
        app.processEvents()
        QTest.qWait(ms)
        app.processEvents()


@pytest.fixture
def window(qapp, cape_path):
    w = MainWindow(str(cape_path))
    w.show()
    settle(qapp)
    assert w.ds is not None, 'file did not load'
    yield w
    w.close()


# ---- F1 --------------------------------------------------------------------------
def test_startup_without_a_file_asks_for_one(qapp, monkeypatch, cape_path):
    """F1.1/F1.2: the picker fires on startup and its result is loaded."""
    calls = []

    def fake_dialog(parent, caption, directory, filt):
        calls.append(directory)
        return str(cape_path), filt

    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', staticmethod(fake_dialog))
    w = MainWindow(None)
    w.show()
    settle(qapp)
    assert calls, 'no file dialog was shown at startup'
    assert w.ds is not None and w.stack.currentIndex() == 1
    w.close()


def test_cancelling_the_picker_leaves_a_usable_window(qapp, monkeypatch):
    """F1.3: Cancel is not fatal - the welcome page stays, with Open still available."""
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    w = MainWindow(None)
    w.show()
    settle(qapp)
    assert w.ds is None and w.stack.currentIndex() == 0
    w.close()


def test_unreadable_file_reports_instead_of_crashing(qapp, monkeypatch, tmp_path):
    junk = tmp_path / 'junk.nc'
    junk.write_bytes(b'not a netcdf file at all')
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical',
                        staticmethod(lambda *a, **k: shown.append(a[-1])))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    w = MainWindow(None)
    w.open_path(junk)
    settle(qapp, ms=20, rounds=2)
    assert shown and w.ds is None
    w.close()


# ---- F3 --------------------------------------------------------------------------
def test_map_click_moves_the_graph_to_that_point(window, qapp):
    """F3.3: clicking anywhere on the map redirects the right-hand panel."""
    ds = window.ds
    target_lat, target_lon = 34.075, 35.800
    expected = ds.nearest_index(target_lat, target_lon)
    assert window.point != expected, 'test would pass trivially'

    vb = window.map.plot.vb
    scene_pt = vb.mapViewToScene(QtCore.QPointF(target_lon, target_lat))
    QTest.mouseClick(window.map.viewport(), QtCore.Qt.MouseButton.LeftButton,
                     QtCore.Qt.KeyboardModifier.NoModifier,
                     window.map.mapFromScene(scene_pt))
    settle(qapp, ms=30, rounds=3)

    assert window.point == expected
    assert np.array_equal(window.plot.series, ds.series(*expected))


def test_click_outside_the_grid_snaps_to_the_edge(window, qapp):
    """F3.3: clamped, not ignored."""
    ds = window.ds
    window.map.pointPicked.emit(*ds.nearest_index(99.0, 99.0))
    settle(qapp, ms=20, rounds=2)
    assert window.point == (ds.ny - 1, ds.nx - 1)


def test_wheel_zoom_narrows_the_view_and_stops_the_auto_refit(window, qapp):
    """F3.2: the wheel zooms, and zooming hands control to the user."""
    vb = window.map.plot.vb
    before = vb.viewRange()[0]
    vb.scaleBy((0.5, 0.5))                      # what a wheel notch does
    vb.sigRangeChangedManually.emit(vb.state['mouseEnabled'])
    settle(qapp, ms=20, rounds=2)
    after = vb.viewRange()[0]
    assert (after[1] - after[0]) < (before[1] - before[0])
    assert window.map._user_zoomed is True


def test_reset_view_shows_the_whole_domain(window, qapp):
    window.map.plot.vb.scaleBy((0.2, 0.2))
    window.map.reset_view()
    settle(qapp, ms=30, rounds=3)
    (x0, x1), (y0, y1) = window.map.plot.vb.viewRange()
    ex0, ex1, ey0, ey1 = window.ds.extent()
    assert x0 <= ex0 and x1 >= ex1 and y0 <= ey0 and y1 >= ey1


# ---- F4 --------------------------------------------------------------------------
def test_hover_shows_the_six_statistics_for_that_time(window, qapp):
    """F4.3: hovering the graph fills time/mean/max/min/P90/P10 for the hovered step."""
    ds = window.ds
    window.select_point(*ds.nearest_index(34.075, 35.800))
    settle(qapp, ms=20, rounds=2)

    window.plot.hovered.emit(110)
    settle(qapp, ms=20, rounds=2)

    expected = member_stats(ds.series(*window.point)[110])
    shown = {k: v.text() for k, v in window.readout.values.items()}
    assert shown['time'] == '2026-08-27 14:00Z  (+110 h)'
    assert shown['mean'] == f'{expected["mean"]:,.1f} J kg-1'
    assert shown['max'] == f'{expected["max"]:,.1f} J kg-1'
    assert shown['min'] == f'{expected["min"]:,.1f} J kg-1'
    assert shown['p90'] == f'{expected["p90"]:,.1f} J kg-1'
    assert shown['p10'] == f'{expected["p10"]:,.1f} J kg-1'


def test_mouse_move_over_the_plot_emits_the_right_time_index(window, qapp):
    """The real event path: a mouse move on the plot resolves to a forecast hour."""
    seen = []
    window.plot.hovered.connect(seen.append)
    vb = window.plot.getPlotItem().vb
    y = float(np.nanmean(window.plot.series))
    scene_pt = vb.mapViewToScene(QtCore.QPointF(72.0, y))
    QTest.mouseMove(window.plot.viewport(), window.plot.mapFromScene(scene_pt))
    settle(qapp, ms=30, rounds=3)
    assert seen and seen[-1] == 72


def test_leaving_the_plot_falls_back_to_the_map_time(window, qapp):
    """F4.1: -1 must not blank the panel."""
    window.set_time(30)
    window.plot.hovered.emit(90)
    settle(qapp, ms=20, rounds=2)
    assert window.readout.values['time'].text() == window.ds.label_for(90)
    window.plot.hovered.emit(-1)
    settle(qapp, ms=20, rounds=2)
    assert window.readout.values['time'].text() == window.ds.label_for(30)


def test_time_slider_and_map_stay_in_step(window, qapp):
    window.set_time(64)
    settle(qapp, ms=20, rounds=2)
    assert window.slider.value() == 64
    assert window.plot.time_line.value() == pytest.approx(64.0)
    assert '+64 h' in window.time_label.text()
