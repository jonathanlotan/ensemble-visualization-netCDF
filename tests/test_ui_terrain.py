"""R9, on the real widgets: the Topography tick draws shaded relief under any map.

What the request asked for is an OPTION, so most of this is about the option behaving:
off until asked, on across fields and time once asked, remembered, honest in the title,
and absent -- disabled, with the reason -- when the bundle is not there.
"""
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

import synth
from imsicon import ingest, terrain
from imsicon.ui import downloaddialog, mapview
from imsicon.ui.main import MainWindow

ERRORS = []


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
    window.resize(1200, 800)
    window.show()
    settle(app)
    if window.scan is not None and window.scan.isRunning():
        window.scan.wait(5000)
        window._on_scan_done(window.ds.value_range)
    settle(app)
    assert window.ds is not None and not ERRORS, ERRORS
    return window


def real_grid(path, field='T_2M'):
    """An ensemble temperature on the REAL 261x161 grid, so the land mask has a coast."""
    nt, nm, ny, nx = 2, 2, 261, 161
    base = 290.0 + 0.02 * np.arange(ny)[:, None] + 0.01 * np.arange(nx)[None, :]
    data = np.stack([np.stack([base + m for m in range(nm)]) for _ in range(nt)])
    synth.write_nc3(path, field, 'K', data, history=synth.HISTORY_TEMPLATE.format(field=field),
                    lat=28.0 + 0.025 * np.arange(ny), lon=33.0 + 0.025 * np.arange(nx),
                    long_name='2m temperature', standard_name='air_temperature')
    return path


@pytest.fixture
def window(qapp, tmp_path):
    w = open_window(qapp, real_grid(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
    yield w
    w.close()


def alpha_at(window, lat, lon):
    """The relief image's alpha at a place, through the item's own transform."""
    item = window.map.terrain
    image = item.image
    inverse, ok = item.transform().inverted()
    assert ok
    point = inverse.map(QtCore.QPointF(lon, lat))
    ix, iy = int(point.x()), int(point.y())
    return int(image[iy, ix, 3])


# ---- the option --------------------------------------------------------------------------
def test_topography_is_offered_and_off_until_asked(window):
    assert window.topo_check.isEnabled()
    assert not window.topo_check.isChecked()
    assert not window.map.terrain_drawn
    assert 'terrain shading' not in window.map.plot.titleLabel.text


def test_ticking_it_draws_relief_between_the_field_and_the_isolines(window, qapp):
    window.topo_check.setChecked(True)
    settle(qapp)
    assert window.map.terrain_drawn
    assert 'terrain shading' in window.map.plot.titleLabel.text
    assert mapview.Z_FIELD < mapview.Z_TERRAIN < mapview.Z_ISOLINE < mapview.Z_COAST
    item = window.map.terrain
    assert item.zValue() == mapview.Z_TERRAIN
    assert item.paintMode == QtGui.QPainter.CompositionMode.CompositionMode_Multiply
    assert item.image.shape[-1] == 4 and item.image.dtype == np.uint8


def test_the_sea_stays_white_and_the_land_is_shaded(window, qapp):
    window.topo_check.setChecked(True)
    settle(qapp)
    assert alpha_at(window, 31.78, 35.22) == 255      # Jerusalem: land
    assert alpha_at(window, 31.50, 35.45) == 255      # the Dead Sea shore: land below sea level
    assert alpha_at(window, 33.00, 34.00) == 0        # the Mediterranean off Haifa
    assert alpha_at(window, 32.50, 33.50) == 0        # the Mediterranean off Gaza
    assert alpha_at(window, 33.50, 34.50) == 0        # the Mediterranean off Beirut
    image = window.map.terrain.image
    land = image[..., 3] == 255
    assert (image[..., 0][land] < 255).any()          # some slope is shaded
    assert (image[..., 0][land] == 255).any()         # and flat ground is left alone


def test_the_relief_is_built_once_and_survives_time_and_a_new_field(window, qapp, tmp_path):
    window.topo_check.setChecked(True)
    settle(qapp)
    built = window.map.terrain.image
    for t in range(window.ds.n_times):
        window.set_time(t)
    assert window.map.terrain.image is built
    real_grid(tmp_path / 'ICON_ENS_2026082300_T_S.nc', 'T_S')
    window._open_field_file(tmp_path / 'ICON_ENS_2026082300_T_S.nc')
    settle(qapp)
    assert window.ds.field == 'T_S'
    assert window.topo_check.isChecked() and window.map.terrain_drawn
    assert window.map.terrain.image is built
    assert 'terrain shading' in window.map.plot.titleLabel.text


def test_unticking_takes_the_relief_down_and_the_title_note_with_it(window, qapp):
    window.topo_check.setChecked(True)
    settle(qapp)
    window.topo_check.setChecked(False)
    settle(qapp)
    assert not window.map.terrain_drawn
    assert 'terrain shading' not in window.map.plot.titleLabel.text


def test_the_choice_is_remembered_for_the_next_window(window, qapp, tmp_path):
    window.topo_check.setChecked(True)
    settle(qapp)
    assert str(window.settings.value('display/topography')).lower() in ('true', '1')
    again = open_window(qapp, tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    try:
        assert again.topo_check.isChecked() and again.map.terrain_drawn
    finally:
        again.close()


def test_without_the_bundle_the_tick_is_disabled_and_says_why(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(terrain, 'TERRAIN', tmp_path / 'missing.npz')
    cache = {}
    monkeypatch.setattr(terrain.load_terrain, '__defaults__', (None, cache))
    w = open_window(qapp, real_grid(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
    try:
        assert not w.topo_check.isEnabled()
        assert 'missing' in w.topo_check.toolTip()
        assert w.map.set_terrain(True) is False
        assert not w.map.terrain_drawn
    finally:
        w.close()


def test_the_scrub_cost_does_not_grow_with_the_relief(window, qapp):
    """The relief is a static image; a scrub must not rebuild it (R9 measured +0.35 ms
    a frame for the paint, and 0.005 ms for the cached check)."""
    import time
    window.topo_check.setChecked(True)
    settle(qapp)
    key = window.map._terrain_key
    t0 = time.perf_counter()
    for _ in range(20):
        window.map.set_terrain(True)
    assert (time.perf_counter() - t0) < 0.05
    assert window.map._terrain_key == key
