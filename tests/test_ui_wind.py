"""The wind map at the window level (R4): the barbs on the real widgets, and the zoom.

The headline requirement is "the resolution of the wind barbs changes based on the zoom
on the map", so that is checked here on a real `MapView` with a real ViewBox rather than
on the geometry helper -- the arithmetic being right in `barbs.py` is not the same claim
as the map actually redrawing when the view range moves.
"""
import numpy as np
import pytest
from PySide6 import QtWidgets

import synth
from imsicon import barbs, derived, ingest
from imsicon.ui import derivedialog, downloaddialog
from imsicon.ui.main import MainWindow


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """The same two guards test_ui_derived.py explains: an unpatched modal file dialog
    hangs the offscreen run forever (G30), and the download dialog must never read the
    developer's keychain."""
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(downloaddialog.download, 'stored_credentials', lambda: None)
    monkeypatch.setattr(downloaddialog.download, 'keyring_module', lambda: None)


@pytest.fixture(autouse=True)
def only_this_test_s_files(monkeypatch):
    """Field discovery must not see the developer's own ./data or cache (G25's reasoning)."""
    def beside_the_file(near=None):
        if near is None:
            return []
        near = __import__('pathlib').Path(near)
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


@pytest.fixture
def run_dir(tmp_path):
    """A run big enough that thinning is a real decision: the model's own 0.025 deg grid."""
    ny, nx, nt, nm = 60, 40, 3, 4
    step = np.arange(nt)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    synth.wind_component(tmp_path / 'ICON_ENS_2026082300_U_10M.nc', 'U_10M',
                         np.broadcast_to(4.0 + 0.2 * x + 1.5 * step, (nt, ny, nx)), nm, 0.3)
    synth.wind_component(tmp_path / 'ICON_ENS_2026082300_V_10M.nc', 'V_10M',
                         np.broadcast_to(-6.0 + 0.1 * y - 0.8 * step, (nt, ny, nx)), nm, -0.2)
    return tmp_path


@pytest.fixture
def window(qapp, run_dir):
    w = MainWindow(str(run_dir / 'ICON_ENS_2026082300_U_10M.nc'))
    w.resize(1200, 800)
    w.show()
    settle(qapp)
    finish_scan(w, qapp)
    yield w
    w.close()


def choose(window, app, key):
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    assert key in keys, keys
    window.field_combo.setCurrentIndex(keys.index(key))
    settle(app)
    for worker in (window.decompressor, window.builder):
        if worker is not None and worker.isRunning():
            worker.wait(30000)
    settle(app)
    finish_scan(window, app)
    return window.ds


def look_at(window, app, span):
    """Zoom the map to a `span`-degree box about the middle of the domain."""
    window.map._user_zoomed = True          # or the layout refits the domain under us
    cx = float(np.mean(window.ds.lon))
    cy = float(np.mean(window.ds.lat))
    window.map.plot.vb.setRange(xRange=(cx - span / 2, cx + span / 2),
                                yRange=(cy - span / 2, cy + span / 2), padding=0)
    settle(app, rounds=3)
    return window.map.barb_stride, window.map.barb_count


# ---- getting to the wind map -----------------------------------------------------------
def test_the_wind_map_is_offered_when_both_components_are_on_disk(window):
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    labels = [window.field_combo.itemText(i) for i in range(window.field_combo.count())]
    assert keys == ['base', 'file:V_10M', derivedialog.WIND]
    assert labels[-1].startswith(derived.WIND_FIELD) and 'barbs' in labels[-1]


def test_one_component_alone_offers_no_wind_map(qapp, tmp_path):
    """Half a wind is not a wind, and a menu entry that always fails is worse than none."""
    synth.wind_component(tmp_path / 'ICON_ENS_2026082300_U_10M.nc', 'U_10M',
                         np.zeros((3, 4, 5)), 3)
    w = MainWindow(str(tmp_path / 'ICON_ENS_2026082300_U_10M.nc'))
    w.show()
    settle(qapp)
    finish_scan(w, qapp)
    try:
        keys = [w.field_combo.itemData(i) for i in range(w.field_combo.count())]
        assert derivedialog.WIND not in keys
    finally:
        w.close()


def test_choosing_it_puts_the_speed_on_every_panel_and_barbs_on_the_map(window, qapp):
    ds = choose(window, qapp, derivedialog.WIND)
    assert ds.field == 'WSPD_10M'
    assert window.map.plot.titleLabel.text.startswith(derived.WIND_NAME)
    assert window.plot.getAxis('left').labelText == derived.WIND_NAME
    assert window.readout.subtitle.text().startswith(derived.WIND_NAME)
    assert window.status_left.text().startswith(derived.WIND_NAME)
    assert window.map.barb_count > 0
    assert len(window.map.barb.over.getData()[0]) > 0


def test_the_map_says_which_wind_the_barbs_are(window, qapp):
    choose(window, qapp, derivedialog.WIND)
    assert 'barbs (kt)' in window.map.plot.titleLabel.text
    index = [window.agg_combo.itemData(i) for i in range(window.agg_combo.count())]
    window.agg_combo.setCurrentIndex(index.index('spread'))
    settle(qapp)
    assert 'mean vector' in window.map.plot.titleLabel.text


def test_the_barbs_control_offers_the_runs_wind_over_a_field_that_has_none(window, qapp):
    """v8: over a plain field the control is live -- it draws the RUN's wind on top.

    Enabled but unticked: the barbs are an addition the user asks for, because building
    them opens two more files. On the wind map itself it is ticked, as it always was.
    """
    assert window.barbs_check.isEnabled()
    assert not window.barbs_check.isChecked()
    assert 'U_10M' in window.barbs_check.toolTip()
    choose(window, qapp, derivedialog.WIND)
    assert window.barbs_check.isEnabled() and window.barbs_check.isChecked()
    assert 'half feather 5 kt' in window.barbs_check.toolTip()


def test_unticking_the_control_removes_the_barbs_and_leaves_the_speed_map(window, qapp):
    choose(window, qapp, derivedialog.WIND)
    drawn = window.map.barb_count
    window.barbs_check.setChecked(False)
    settle(qapp)
    assert drawn > 0 and window.map.barb_count == 0
    assert len(window.map.barb.over.getData()[0] or []) == 0
    assert window.map.img.image is not None            # the colours are still there
    assert 'barbs' not in window.map.plot.titleLabel.text
    window.barbs_check.setChecked(True)
    settle(qapp)
    assert window.map.barb_count > 0


# ---- the requirement: resolution follows the zoom --------------------------------------
def test_zooming_in_draws_more_barbs_of_the_same_grid(window, qapp):
    """Wide out, the grid is thinned; zoomed in, the stride walks down to every point --
    while the number on screen stays in the same ballpark, which is the actual goal."""
    choose(window, qapp, derivedialog.WIND)
    wide = look_at(window, qapp, 1.5)
    middle = look_at(window, qapp, 0.5)
    close = look_at(window, qapp, 0.15)
    assert wide[0][0] > middle[0][0] > close[0][0] == 1      # (rows, cols) strides
    assert wide[0][1] > middle[0][1] > close[0][1] == 1
    assert all(count > 0 for _stride, count in (wide, middle, close))


def test_the_barbs_are_only_built_for_the_part_of_the_grid_on_screen(window, qapp):
    """At full resolution a 60x40 grid is 2,400 glyphs; a window showing a tenth of it
    must not pay for the other nine."""
    choose(window, qapp, derivedialog.WIND)
    _stride, whole = look_at(window, qapp, 1.5)
    _stride, corner = look_at(window, qapp, 0.15)
    assert corner < window.ds.ny * window.ds.nx
    assert corner < 4 * whole


def test_panning_does_not_reshuffle_the_lattice(window, qapp):
    choose(window, qapp, derivedialog.WIND)
    stride, _count = look_at(window, qapp, 0.6)
    box = window.map.plot.vb.viewRange()
    window.map.plot.vb.setRange(xRange=(box[0][0] + 0.05, box[0][1] + 0.05),
                                yRange=tuple(box[1]), padding=0)
    settle(qapp, rounds=3)
    assert window.map.barb_stride == stride


def test_the_glyphs_keep_their_size_on_screen_as_the_zoom_changes(window, qapp):
    """A barb is defined in pixels, so its footprint in DEGREES has to shrink in step
    with the view -- the opposite of what drawing it in degrees would do. Measured
    against the range actually on screen, since the aspect lock and the pan limits both
    have a say in what a request for a box turns into."""
    choose(window, qapp, derivedialog.WIND)

    def staff():
        xs, ys = window.map.barb.over.getData()
        px, py = window.map.plot.vb.viewPixelSize()
        degrees = float(np.hypot(xs[1] - xs[0], ys[1] - ys[0]))
        pixels = float(np.hypot((xs[1] - xs[0]) / px, (ys[1] - ys[0]) / py))
        return degrees, pixels

    look_at(window, qapp, 1.5)
    wide_degrees, wide_pixels = staff()
    look_at(window, qapp, 0.15)
    close_degrees, close_pixels = staff()
    assert wide_pixels == pytest.approx(barbs.SHAFT_PX, rel=1e-6)
    assert close_pixels == pytest.approx(barbs.SHAFT_PX, rel=1e-6)
    assert close_degrees < wide_degrees / 5.0


# ---- reading the wind ------------------------------------------------------------------
def test_the_status_bar_adds_the_direction_the_wind_comes_from(window, qapp):
    """A speed with no direction is half a reading -- and it is offered whether or not
    the barbs are switched on, because the two answer different questions."""
    choose(window, qapp, derivedialog.WIND)
    lat, lon = float(window.ds.lat[2]), float(window.ds.lon[2])
    expected = window.ds.direction_at(window.t, 2, 2, window.agg_combo.currentData())
    for barbs_on in (True, False):
        window.barbs_check.setChecked(barbs_on)
        settle(qapp)
        window._on_map_cursor(lat, lon, 7.5)
        text = window.status_right.text()
        assert 'from' in text and f'{expected:.0f}' in text


def test_a_field_with_no_direction_says_nothing_about_one(window, qapp):
    window._on_map_cursor(float(window.ds.lat[1]), float(window.ds.lon[1]), 4.0)
    assert 'from' not in window.status_right.text()


def staff_angle(window):
    """The bearing of the first barb's staff, in screen terms."""
    xs, ys = window.map.barb.over.getData()
    return float(np.degrees(np.arctan2(ys[1] - ys[0], xs[1] - xs[0])))


def test_the_time_step_moves_the_barbs_with_the_map(window, qapp):
    """The fixture's wind veers and freshens with the forecast hour, so scrubbing time
    has to turn the staffs -- barbs are not a static overlay."""
    choose(window, qapp, derivedialog.WIND)
    before = staff_angle(window)
    window.set_time(2)
    settle(qapp)
    assert abs(staff_angle(window) - before) > 1.0
    assert window.map.barb_count > 0


def test_switching_back_to_a_plain_field_takes_the_barbs_down_with_it(window, qapp):
    """The speed map's own barbs come down with it. The control stays live, because the
    run's wind can be drawn over the U_10M map too (v8) -- but only when asked."""
    choose(window, qapp, derivedialog.WIND)
    assert window.map.barb_count > 0
    choose(window, qapp, 'base')
    assert window.ds.field == 'U_10M'
    assert window.map.barb_count == 0
    assert not window.barbs_check.isChecked()
    assert window.barbs_check.isEnabled()


def test_a_wind_view_carries_its_kind_so_the_combo_names_what_is_on_screen(window, qapp):
    """The bug R3.10 found for --derive, checked for the wind map before it can recur."""
    choose(window, qapp, derivedialog.WIND)
    assert window.ds.derived_kind == derivedialog.WIND
    assert window.field_combo.currentData() == derivedialog.WIND
    assert window.field_combo.currentText().startswith(derived.WIND_FIELD)


def test_the_dialog_offers_the_wind_map_and_says_which_files_it_uses(qapp, run_dir):
    dialog = derivedialog.DerivedDialog(None, near=run_dir, run='2026082300')
    dialog.wind_radio.setChecked(True)
    assert dialog.kind() == derivedialog.WIND
    request = dialog.request()
    assert [p.name for p in request.paths] == ['ICON_ENS_2026082300_U_10M.nc',
                                               'ICON_ENS_2026082300_V_10M.nc']
    assert 'U_10M' in dialog.wind_inputs_label.text()
    assert dialog.status.text() == ''


def test_the_dialog_says_what_is_missing_instead_of_offering_a_broken_wind_map(qapp, tmp_path):
    synth.wind_component(tmp_path / 'ICON_ENS_2026082300_U_10M.nc', 'U_10M',
                         np.zeros((3, 4, 5)), 3)
    dialog = derivedialog.DerivedDialog(None, near=tmp_path, run='2026082300')
    dialog.wind_radio.setChecked(True)
    assert 'V_10M' in dialog.status.text()
    assert dialog.request() is None
    ok = dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()


def test_the_download_shortcut_ticks_exactly_the_two_components(qapp):
    """One component on its own is the half-a-feature R3.11 found for the dew point."""
    dialog = downloaddialog.DownloadDialog(None)
    dialog._on_listed((object(), downloaddialog.download.parse_listing(
        ''.join(f'<a href="ICON_ENS_2026082300_{f}.nc.bz2">'
                f'ICON_ENS_2026082300_{f}.nc.bz2</a>'
                for f in ('CAPE_ML', 'U_10M', 'V_10M', 'VMAX_10M')))))
    dialog._select_wind()
    assert {entry.field for entry in dialog.selected()} == {'U_10M', 'V_10M'}
    dialog.close()
