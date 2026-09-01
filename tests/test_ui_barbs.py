"""v8: wind barbs over ANY map, not only over the wind map.

R4 drew barbs from the view on screen, so only a `WindView` could have them; R4.8 left
"barbs over another field" undone because it needs a SECOND view alongside the one being
drawn. That is what these tests are about: which wind gets picked, that it matches the map
it is drawn over, that the title says the colours and the feathers are different
quantities, and that it is opt-in rather than a surprise.
"""
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

import synth
from imsicon import ingest
from imsicon.ui import derivedialog, downloaddialog
from imsicon.ui.main import MainWindow

ERRORS = []


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """G30/G38: modal dialogs wedge an offscreen run, and the keychain is not ours."""
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
    """G25: discovery must not see the developer's ./data or caches."""
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
    assert window.ds is not None and not ERRORS, ERRORS[:1]
    return window


def barb_geometry(window):
    """The drawn barb polylines, as a comparable snapshot."""
    xs, ys = window.map.barb.over.getData()
    return (np.asarray(xs, dtype=float).round(6).tobytes(),
            np.asarray(ys, dtype=float).round(6).tobytes())


def tick_barbs(window, app, on=True):
    """Tick "Wind barbs" the way a user does, and let the overlay open."""
    window.barbs_check.setChecked(on)
    settle(app)
    if window._overlay_builder is not None and window._overlay_builder.isRunning():
        window._overlay_builder.wait(20000)
    settle(app)
    assert not ERRORS, ERRORS[0]
    return window.map.barb_count


def ensemble_run(directory, run='2026082300'):
    """A CAPE map, and the wind that can be drawn over it."""
    shape = synth.SHAPE
    cape = np.linspace(0.0, 3000.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    synth.write_nc3(directory / f'ICON_ENS_{run}_CAPE_ML.nc', 'CAPE_ML', 'J kg-1', cape,
                    history=synth.HISTORY_TEMPLATE.format(field='CAPE_ML'),
                    long_name='cape of mean surface layer parcel')
    synth.wind_pair(directory, n_times=shape[0], n_members=shape[1], ny=shape[2],
                    nx=shape[3], run=run)
    return directory / f'ICON_ENS_{run}_CAPE_ML.nc'


# ---- over a plain ensemble field -----------------------------------------------------
def test_the_runs_wind_can_be_drawn_over_a_field_that_has_no_direction(qapp, tmp_path):
    window = open_window(qapp, ensemble_run(tmp_path))
    try:
        assert window.ds.field == 'CAPE_ML'
        assert window.barbs_check.isEnabled()
        # Opt-in: opening a CAPE map does not silently open two more files.
        assert not window.barbs_check.isChecked()
        assert window.map.barb_count == 0
        assert tick_barbs(window, qapp) > 0
        assert window.ds.field == 'CAPE_ML'          # the MAP is still CAPE
    finally:
        window.close()


def test_the_title_says_the_feathers_are_a_different_field(qapp, tmp_path):
    """The colours are CAPE and the barbs are wind: a map showing both has to say so."""
    window = open_window(qapp, ensemble_run(tmp_path))
    try:
        tick_barbs(window, qapp)
        title = window.map.plot.titleLabel.text
        assert 'CAPE_ML' in title
        assert 'barbs (kt) from U_10M/V_10M' in title
        assert 'ensemble mean vector' in title
    finally:
        window.close()


def test_the_overlay_follows_the_aggregation_and_the_time_step(qapp, tmp_path):
    window = open_window(qapp, ensemble_run(tmp_path))
    try:
        tick_barbs(window, qapp)
        keys = [window.agg_combo.itemData(i) for i in range(window.agg_combo.count())]
        window.agg_combo.setCurrentIndex(keys.index('member'))
        settle(qapp)
        assert 'one member' in window.map.plot.titleLabel.text
        assert window.map.barb_count > 0
        # The fixture's wind freshens with the forecast hour, so the glyphs must change.
        before = barb_geometry(window)
        window.set_time(4)
        settle(qapp)
        assert barb_geometry(window) != before
    finally:
        window.close()


def test_unticking_takes_the_overlay_off_and_leaves_the_map(qapp, tmp_path):
    window = open_window(qapp, ensemble_run(tmp_path))
    try:
        tick_barbs(window, qapp)
        image = np.array(window.map.img.image, copy=True)
        assert tick_barbs(window, qapp, on=False) == 0
        assert np.allclose(window.map.img.image, image)     # the colours are untouched
    finally:
        window.close()


def test_the_overlay_survives_switching_to_another_field_of_the_run(qapp, tmp_path):
    """Built once: flipping between two maps of one run must not reopen the wind."""
    window = open_window(qapp, ensemble_run(tmp_path))
    try:
        tick_barbs(window, qapp)
        overlay = window.wind_overlay
        keys = [window.field_combo.itemData(i)
                for i in range(window.field_combo.count())]
        window.field_combo.setCurrentIndex(keys.index('file:U_10M'))
        settle(qapp)
        for worker in (window.decompressor, window.builder, window._overlay_builder):
            if worker is not None and worker.isRunning():
                worker.wait(20000)
        settle(qapp)
        assert window.ds.field == 'U_10M'
        assert window.wind_overlay is overlay        # the same object, not rebuilt
        assert window.barbs_check.isChecked() and window.map.barb_count > 0
    finally:
        window.close()


def test_a_run_with_no_wind_on_disk_has_the_control_disabled(qapp, tmp_path):
    """Disabled, not hidden -- and the tooltip says what to download."""
    shape = synth.SHAPE
    cape = np.zeros(shape, dtype=np.float32)
    synth.write_nc3(tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc', 'CAPE_ML', 'J kg-1',
                    cape, history=synth.HISTORY_TEMPLATE.format(field='CAPE_ML'),
                    long_name='cape of mean surface layer parcel')
    window = open_window(qapp, tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc')
    try:
        assert not window.barbs_check.isEnabled()
        assert not window.barbs_check.isChecked()
        assert 'not on disk' in window.barbs_check.toolTip()
        assert window.map.barb_count == 0
    finally:
        window.close()


# ---- over a pressure level -------------------------------------------------------------
def icon_run_with_wind(directory, run='2026083100'):
    """A deterministic run holding BOTH wind pairs: 3-D u/v and 10 m u_10m/v_10m."""
    rng = np.random.default_rng(11)
    shape = (4, len(synth.ICON_LEVELS), 4, 5)
    synth.pressure_field(directory / f'IE_{run}_temp.nc', run=run)
    synth.pressure_field(directory / f'IE_{run}_u.nc', field='u', units='m s-1',
                         values=rng.normal(0, 10, shape), run=run)
    synth.pressure_field(directory / f'IE_{run}_v.nc', field='v', units='m s-1',
                         values=rng.normal(0, 10, shape), run=run)
    synth.surface_field(directory / f'IE_{run}_t_2m.nc', run=run)
    synth.surface_field(directory / f'IE_{run}_u_10m.nc', field='u_10m', units='m s-1',
                        values=np.full((4, 4, 5), 5.0), run=run)
    synth.surface_field(directory / f'IE_{run}_v_10m.nc', field='v_10m', units='m s-1',
                        values=np.full((4, 4, 5), -5.0), run=run)
    return directory


def test_barbs_over_a_pressure_map_are_the_wind_at_that_level(qapp, tmp_path):
    """Not the 10 m wind that happens to be in the same run: the wind AT 850 hPa."""
    directory = icon_run_with_wind(tmp_path)
    window = open_window(qapp, directory / 'IE_2026083100_temp.nc')
    try:
        assert window.ds.axis.is_pressure
        assert tick_barbs(window, qapp) > 0
        assert window.wind_overlay.source_label == 'u/v'
        assert 'the wind at this level' in window.map.plot.titleLabel.text
        # Moving up the column moves the barbs, because they are that level's wind.
        before = barb_geometry(window)
        window.step_level(1)
        settle(qapp)
        assert barb_geometry(window) != before
    finally:
        window.close()


def test_barbs_over_a_surface_map_are_the_10_m_wind(qapp, tmp_path):
    """The same run, the other pair: a 2 m temperature gets the 10 m wind."""
    directory = icon_run_with_wind(tmp_path)
    window = open_window(qapp, directory / 'IE_2026083100_t_2m.nc')
    try:
        assert window.ds.axis.kind == 'single'
        assert tick_barbs(window, qapp) > 0
        assert window.wind_overlay.source_label == 'u_10m/v_10m'
        assert 'barbs (kt) from u_10m/v_10m' in window.map.plot.titleLabel.text
    finally:
        window.close()


def test_an_overlay_built_for_one_shape_is_not_reused_on_another(qapp, tmp_path):
    """A column's wind cannot be drawn over a surface map: dropped, not silently reused."""
    directory = icon_run_with_wind(tmp_path)
    window = open_window(qapp, directory / 'IE_2026083100_temp.nc')
    try:
        tick_barbs(window, qapp)
        assert window.wind_overlay is not None
        keys = [window.field_combo.itemData(i)
                for i in range(window.field_combo.count())]
        window.field_combo.setCurrentIndex(keys.index('file:t_2m'))
        settle(qapp)
        for worker in (window.decompressor, window.builder, window._overlay_builder):
            if worker is not None and worker.isRunning():
                worker.wait(20000)
        settle(qapp)
        assert window.ds.field == 't_2m'
        assert window.wind_overlay is None          # the column's wind does not fit
        assert window.map.barb_count == 0
        assert window.barbs_check.isEnabled()       # the 10 m pair is still offered
        assert tick_barbs(window, qapp) > 0
        assert window.wind_overlay.source_label == 'u_10m/v_10m'
    finally:
        window.close()


def test_the_wind_maps_own_barbs_still_win(qapp, tmp_path):
    """A WindView has vectors of its own; the overlay must not shadow them."""
    directory = ensemble_run(tmp_path).parent
    window = open_window(qapp, directory / 'ICON_ENS_2026082300_CAPE_ML.nc')
    try:
        tick_barbs(window, qapp)
        keys = [window.field_combo.itemData(i)
                for i in range(window.field_combo.count())]
        window.field_combo.setCurrentIndex(keys.index(derivedialog.WIND))
        settle(qapp)
        if window.builder is not None and window.builder.isRunning():
            window.builder.wait(20000)
        settle(qapp)
        source, is_overlay = window.wind_source()
        assert source is window.ds and not is_overlay
        assert 'from' not in window.map.plot.titleLabel.text.split('barbs')[-1]
    finally:
        window.close()


# ---- the wind has to fit the map, not just itself (G45) --------------------------------
def test_a_wind_that_stops_earlier_than_the_map_is_refused(qapp, tmp_path):
    """Found on real data: an interrupted download stops each file at its own step.

    The two components pair with each other perfectly and still cannot go over a map with
    more steps -- the barbs would run off the end of the wind the moment the slider passed
    the shorter file's last step, which wedged the app before this check existed.
    """
    shape = synth.SHAPE                                   # 6 steps
    cape = np.zeros(shape, dtype=np.float32)
    synth.write_nc3(tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc', 'CAPE_ML', 'J kg-1',
                    cape, history=synth.HISTORY_TEMPLATE.format(field='CAPE_ML'),
                    long_name='cape of mean surface layer parcel')
    # ...and a wind that holds only 3 of those steps.
    synth.wind_pair(tmp_path, n_times=3, n_members=shape[1], ny=shape[2], nx=shape[3],
                    run='2026082300')
    window = open_window(qapp, tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc')
    try:
        assert window.barbs_check.isEnabled()             # the files are there...
        assert tick_barbs(window, qapp) == 0              # ...but they do not fit
        assert window.wind_overlay is None
        assert not window.barbs_check.isChecked()
        assert 'no barbs over this map' in window.status_right.text()
        assert 'forecast steps' in window.status_right.toolTip()
        # The map itself is untouched, and scrubbing to a step the wind never had is safe.
        window.set_time(window.ds.n_times - 1)
        settle(qapp)
        assert window.map.barb_count == 0
        assert not ERRORS, ERRORS[0]
    finally:
        window.close()


def test_the_shape_key_notices_a_different_number_of_steps(qapp, tmp_path):
    """The reuse path has to see it too, not only the build path."""
    window = open_window(qapp, ensemble_run(tmp_path))
    try:
        tick_barbs(window, qapp)
        assert window.wind_source()[0] is window.wind_overlay
        window.ds.raw.n_times -= 1                        # pretend a shorter map
        assert window.wind_source() == (None, False)
        assert window.map.barb_count >= 0                 # and refreshing is still safe
        window.refresh_map()
        settle(qapp)
        assert window.map.barb_count == 0
    finally:
        window.close()


def test_a_range_that_arrives_after_the_view_moved_on_is_ignored(qapp, tmp_path):
    """G46: `_apply_range(None)` used to unpack None and take the window down.

    A range scan is per transform signature, so a Rate change while one is running leaves
    the view with no range for the new signature -- and the finished scan then hands the
    window a None it used to unpack. Seen for real with `--rate 3h --barbs on`, where
    opening the wind held the event loop long enough for the two to cross.
    """
    path, _hourly = synth.accumulated_precip(tmp_path / 'ICON_ENS_2026082300_TOT_PREC.nc')
    window = open_window(qapp, path)
    try:
        window._apply_range(None)            # must be a no-op, not a crash
        window._on_scan_done(None)
        window.set_rate = getattr(window, 'set_rate', None)
        # ...and the ordinary path still works.
        window._apply_range((0.0, 5.0))
        assert window.readout.span == 5.0
        assert not ERRORS, ERRORS[0]
    finally:
        window.close()
