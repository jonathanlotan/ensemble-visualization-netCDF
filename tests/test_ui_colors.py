"""Transparent-at-zero and the grey land, at the window level (R6).

`test_colors.py` checks the two colour transforms and the predicate. This checks the wire
between them and the map: which scale a real window decides a real field deserves, that
the decision follows the aggregation, the units and the Scale control, and that the land
is underneath it when it does.
"""
import numpy as np
import pytest
from PySide6 import QtWidgets

import synth
from imsicon import ingest
from imsicon.ncwrite import write_nc3
from imsicon.ui import derivedialog, downloaddialog
from imsicon.ui.main import DIVERGING_MAPS, MainWindow


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """G30: an unpatched modal file dialog hangs the offscreen run forever, and the
    download dialog must never read the developer's keychain.

    `QMessageBox.critical` is caught for the same reason and one more: `MainWindow._error`
    is reached from a worker's failure signal, so a broken fixture does not fail the test
    -- it wedges the whole run inside `processEvents` with no failing test to point at.
    Collecting the text turns that into an assertion (`no_errors`) that names the cause.
    """
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


ERRORS = []


def no_errors():
    assert not ERRORS, ERRORS[0]


@pytest.fixture(autouse=True)
def only_this_test_s_files(monkeypatch):
    """G25's reasoning: the fixtures use run 2026082300, which is also the run the 407 MB
    reference file carries, so discovery must not see the developer's own ./data."""
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


def finish_scan(window, app):
    if window.scan is not None and window.scan.isRunning():
        window.scan.wait(5000)
        window._on_scan_done(window.ds.value_range)
    settle(app)


NT, NM, NY, NX = 4, 3, 12, 9


def _members(base, spread):
    return np.stack([np.asarray(base, dtype=float) + spread * (m - 1)
                     for m in range(NM)], axis=1)


@pytest.fixture
def run_dir(tmp_path):
    """One field with a true zero floor (CAPE) and one without (temperature).

    The CAPE field is mostly zero with one active corner, which is what a real convective
    field looks like and what the whole change is about.
    """
    step = np.arange(NT)[:, None, None]
    y = np.arange(NY)[None, :, None]
    x = np.arange(NX)[None, None, :]
    cape = np.clip(120.0 * (y - 7) * (x - 5) - 40.0 * step, 0.0, None)
    write_nc3(tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc', 'CAPE_ML', 'J kg-1',
              np.clip(_members(cape, 25.0), 0.0, None),
              history=synth.HISTORY_TEMPLATE.format(field='CAPE_ML'),
              long_name='cape of mean surface layer parcel')
    warm = 291.0 + 0.55 * y + 0.35 * x + 1.5 * step
    write_nc3(tmp_path / 'ICON_ENS_2026082300_T_2M.nc', 'T_2M', 'K',
              _members(warm, 0.4),
              history=synth.HISTORY_TEMPLATE.format(field='T_2M'),
              long_name='2m temperature', standard_name='air_temperature')
    damp = np.broadcast_to(80.0 - 2.0 * x - 1.0 * step + 0.0 * y, (NT, NY, NX))
    # A zero floor that also has something on the Units combo, so the fade can be shown
    # to survive an affine: 0 m of snow is still 0 cm of snow.
    snow = np.clip(0.004 * (y - 9) * (x - 6) - 0.002 * step, 0.0, None)
    write_nc3(tmp_path / 'ICON_ENS_2026082300_H_SNOW.nc', 'H_SNOW', 'm',
              np.clip(_members(snow, 0.001), 0.0, None),
              history=synth.HISTORY_TEMPLATE.format(field='H_SNOW'),
              long_name='weighted snow depth')
    write_nc3(tmp_path / 'ICON_ENS_2026082300_RELHUM_2M.nc', 'RELHUM_2M', '%',
              np.clip(_members(damp, 1.5), 1.0, 100.0),
              history=synth.HISTORY_TEMPLATE.format(field='RELHUM_2M'),
              long_name='relative humidity in 2m', standard_name='relative_humidity')
    return tmp_path


def open_window(qapp, path):
    window = MainWindow(str(path))
    window.resize(1000, 700)
    window.show()
    settle(qapp)
    finish_scan(window, qapp)
    return window


@pytest.fixture
def cape(qapp, run_dir):
    window = open_window(qapp, run_dir / 'ICON_ENS_2026082300_CAPE_ML.nc')
    yield window
    window.close()


@pytest.fixture
def temperature(qapp, run_dir):
    window = open_window(qapp, run_dir / 'ICON_ENS_2026082300_T_2M.nc')
    yield window
    window.close()


def alpha(window, n=256):
    return np.asarray(window.map.cmap.getLookupTable(0.0, 1.0, n, alpha=True))[:, 3]


# ---- which map gets the fade ------------------------------------------------------------
def test_a_field_whose_floor_is_zero_is_not_painted_at_zero(cape):
    assert cape.map.cbar.levels()[0] == 0.0
    assert alpha(cape)[0] == 0
    assert alpha(cape)[-1] == 255


def test_a_temperature_map_stays_opaque_all_the_way_down(temperature):
    """Its floor is the coldest air in the domain, which is a reading and not an absence.
    Fading it would hide the coldest place on the map."""
    assert temperature.map.cbar.levels()[0] > 0.0
    assert (alpha(temperature) == 255).all()


def test_the_decision_follows_the_aggregation(temperature, qapp):
    """`spread` is pinned to zero whatever the field is: no spread means the members
    agree, which is an absence of disagreement and reads as one."""
    assert (alpha(temperature) == 255).all()
    keys = [temperature.agg_combo.itemData(i)
            for i in range(temperature.agg_combo.count())]
    temperature.agg_combo.setCurrentIndex(keys.index('spread'))
    settle(qapp)
    assert temperature.map.cbar.levels()[0] == 0.0
    assert alpha(temperature)[0] == 0
    temperature.agg_combo.setCurrentIndex(keys.index('mean'))
    settle(qapp)
    assert (alpha(temperature) == 255).all()


def test_the_decision_follows_the_scale_control(temperature, qapp):
    """"This frame" moves the floor, so it moves the answer with it."""
    temperature.scale_combo.setCurrentIndex(1)          # this frame
    settle(qapp)
    assert temperature.map.cbar.levels()[0] > 0.0
    assert (alpha(temperature) == 255).all()


def test_a_units_change_does_not_lose_the_fade(qapp, run_dir):
    """The floor is a float that has been through an affine; 0 m is still 0 cm."""
    window = open_window(qapp, run_dir / 'ICON_ENS_2026082300_H_SNOW.nc')
    try:
        assert window.ds.units == 'cm' and 'mm' in window.ds.unit_labels
        assert alpha(window)[0] == 0
        window.units_combo.setCurrentText('mm')
        settle(qapp)
        no_errors()
        assert window.ds.units == 'mm'
        assert alpha(window)[0] == 0
    finally:
        window.close()


def test_a_difference_map_keeps_its_diverging_ramp_untouched(temperature, qapp):
    """Its centre is a reading -- "no difference" -- not an absence, and the ramp is
    already pale there. Boosting or fading it would be answering a question nobody asked.
    """
    import pyqtgraph as pg

    keys = [temperature.field_combo.itemData(i)
            for i in range(temperature.field_combo.count())]
    assert derivedialog.DEPRESSION in keys
    temperature.field_combo.setCurrentIndex(keys.index(derivedialog.DEPRESSION))
    settle(qapp)
    if temperature.builder is not None and temperature.builder.isRunning():
        temperature.builder.wait(30000)
    settle(qapp)
    finish_scan(temperature, qapp)
    no_errors()
    assert temperature.cmap_combo.currentText() in DIVERGING_MAPS
    stock = np.asarray(pg.colormap.get(temperature.cmap_combo.currentText())
                       .getLookupTable(0.0, 1.0, 64, alpha=True))
    assert np.array_equal(np.asarray(temperature.map.cmap.getLookupTable(
        0.0, 1.0, 64, alpha=True)), stock)


# ---- the land underneath ----------------------------------------------------------------
def test_the_map_paints_land_under_the_field(cape):
    assert cape.map.land_rings > 5
    assert not cape.map.land.path().isEmpty()
    assert cape.map.land.zValue() < cape.map.img.zValue()


def test_the_land_survives_switching_fields(cape, qapp):
    """`set_dataset` rebuilds every layer; the fill must be rebuilt with them."""
    keys = [cape.field_combo.itemData(i) for i in range(cape.field_combo.count())]
    cape.field_combo.setCurrentIndex(keys.index('file:T_2M'))
    settle(qapp)
    for worker in (cape.decompressor, cape.builder):
        if worker is not None and worker.isRunning():
            worker.wait(30000)
    settle(qapp)
    no_errors()
    assert cape.ds.field == 'T_2M'
    assert cape.map.land_rings > 5 and not cape.map.land.path().isEmpty()


# ---- the guard that keeps a scrub cheap -------------------------------------------------
def test_scrubbing_does_not_rebuild_the_lookup_table(cape, qapp, monkeypatch):
    rebuilds = []
    original = cape.map.set_colormap
    monkeypatch.setattr(cape.map, 'set_colormap',
                        lambda cmap: (rebuilds.append(cmap), original(cmap))[1])
    for t in range(cape.ds.n_times):
        cape.set_time(t)
    assert rebuilds == []
