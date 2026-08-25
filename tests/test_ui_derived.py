"""v3 at the window level: a derived field must be an ordinary dataset to the whole UI.

The claim `derived.py` makes is that `MainWindow`, `MapView`, `PlotView` and
`ReadoutPanel` need no special case. These tests check that claim on the real widgets --
map title, colorbar, y axis, readout and status bar read at one instant, the same way
`test_ui_v2.py` checks the units switch.
"""
import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QTest

import synth
from imsicon import derived, ncwrite
from imsicon.ui import derivedialog, downloaddialog
from imsicon.ui.main import MainWindow, _symmetric


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """Two things a UI test must never touch.

    The file picker: `MainWindow(None)` schedules one, and an unpatched modal dialog under
    the offscreen platform simply never returns -- the run hangs rather than failing.

    The keychain: `DownloadDialog` prefills from `stored_credentials()`, so on a developer
    machine with `keyring` installed an unguarded test would read (and on some platforms
    prompt for) their real IMS password. Same reasoning as G25.
    """
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(downloaddialog.download, 'stored_credentials', lambda: None)
    monkeypatch.setattr(downloaddialog.download, 'keyring_module', lambda: None)


def settle(app, ms=30, rounds=5):
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
    synth.pair(tmp_path)
    return tmp_path


@pytest.fixture
def window(qapp, run_dir):
    w = MainWindow(str(run_dir / 'ICON_ENS_2026082300_T_2M.nc'))
    w.show()
    settle(qapp)
    finish_scan(w, qapp)
    assert w.ds is not None
    yield w
    w.close()


def install(window, app, kind, title):
    """Build a derived view the way the window does, and put it on screen."""
    paths = [window.ds.path.parent / 'ICON_ENS_2026082300_T_2M.nc',
             window.ds.path.parent / 'ICON_ENS_2026082300_RELHUM_2M.nc']
    view = derivedialog.build(derivedialog.DerivedRequest(kind, paths, title),
                              dict(window.opened))
    window._install(view, title)
    settle(app)
    finish_scan(window, app)
    return view


def surfaces(w):
    """Everything that has to agree about the field, read at one instant."""
    return {'title': w.map.plot.titleLabel.text,
            'colorbar': tuple(round(float(v), 3) for v in w.map.cbar.levels()),
            'y_axis': w.plot.getAxis('left').labelUnits,
            'readout': w.readout.values['mean'].text(),
            'summary': w.status_left.text(),
            'units': w.ds.units}


# ---- the dew point on screen ---------------------------------------------------------
def test_a_dew_point_drives_every_panel(window, qapp):
    install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    window.select_point(2, 2)
    settle(qapp)
    state = surfaces(window)
    assert state['units'] == '°C'
    assert 'TD_2M' in state['title'] and '[°C]' in state['title']
    assert 'TD_2M' in state['summary'] and state['y_axis'] == '°C'
    assert state['readout'].endswith('°C')
    assert -40.0 < state['colorbar'][0] <= state['colorbar'][1] < 60.0   # not Kelvin


def test_the_graph_shows_one_curve_per_member(window, qapp):
    view = install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    window.select_point(1, 1)
    settle(qapp)
    assert len(window.plot._curves) == view.n_members
    assert window.plot.series.shape == (view.n_times, view.n_members)
    assert np.allclose(window.plot.series, view.series(1, 1), equal_nan=True)


def test_switching_units_on_a_derived_field_moves_everything_together(window, qapp):
    install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    window.select_point(2, 2)
    settle(qapp)
    before = surfaces(window)
    window.units_combo.setCurrentText('K')
    settle(qapp)
    after = surfaces(window)
    assert after['units'] == 'K'
    assert '[K]' in after['title'] and '[K]' in after['summary']
    assert after['y_axis'] == 'K' and after['readout'].endswith('K')
    assert after['colorbar'] == pytest.approx(
        tuple(v + 273.15 for v in before['colorbar']), abs=0.01)


def test_the_rate_control_is_disabled_for_a_derived_field(window, qapp):
    install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    assert not window.rate_combo.isEnabled()
    assert window.units_combo.isEnabled()          # but units still apply


def test_the_time_slider_still_spans_the_forecast(window, qapp):
    view = install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    assert window.slider.maximum() == view.n_times - 1
    window.set_time(4)
    settle(qapp)
    assert '+4 h' in window.time_label.text()
    assert np.allclose(window.map._frame, view.agg_frame(4, 'mean'), equal_nan=True)


# ---- the difference map ---------------------------------------------------------------
def test_a_difference_map_gets_a_diverging_scale_centred_on_zero(window, qapp):
    install(window, qapp, derivedialog.DEPRESSION, 'Depression')
    settle(qapp)
    assert window.ds.diverging
    assert window.cmap_combo.currentText().startswith('CET-D')
    lo, hi = window.map.cbar.levels()
    assert float(lo) == pytest.approx(-float(hi))
    assert float(hi) > 0


def test_leaving_a_difference_restores_a_sequential_map(window, qapp):
    install(window, qapp, derivedialog.DEPRESSION, 'Depression')
    assert window.cmap_combo.currentText().startswith('CET-D')
    install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    assert window.cmap_combo.currentText() == 'turbo'


def test_the_spread_of_a_difference_keeps_a_zero_based_scale(window, qapp):
    """`spread` is max-min, which is non-negative: a symmetric scale would waste half."""
    install(window, qapp, derivedialog.DEPRESSION, 'Depression')
    window.agg_combo.setCurrentIndex(
        [window.agg_combo.itemData(i) for i in range(window.agg_combo.count())].index('spread'))
    settle(qapp)
    lo, _hi = window.map.cbar.levels()
    assert float(lo) == pytest.approx(0.0)


@pytest.mark.parametrize('lo, hi, expected', [
    (-3.0, 8.0, (-8.0, 8.0)), (2.0, 5.0, (-5.0, 5.0)), (0.0, 0.0, (-1.0, 1.0)),
])
def test_symmetric_range(lo, hi, expected):
    assert _symmetric(lo, hi) == expected


# ---- the dialog ------------------------------------------------------------------------
def test_the_dialog_finds_the_pair_beside_the_open_file(qapp, run_dir):
    dialog = derivedialog.DerivedDialog(None, near=run_dir / 'ICON_ENS_2026082300_T_2M.nc')
    assert set(dialog._fields_for('2026082300')) >= {'T_2M', 'RELHUM_2M'}
    assert dialog.request().kind == derivedialog.DEW_POINT
    dialog.close()


def test_the_dialog_says_what_is_missing_rather_than_offering_a_broken_choice(qapp, tmp_path):
    synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')     # no humidity
    dialog = derivedialog.DerivedDialog(None, near=tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    ok = dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()
    assert 'RELHUM_2M' in dialog.status.text()
    dialog.close()


def test_the_difference_pair_defaults_to_something_subtractable(qapp, run_dir):
    dewpoint = derived.dew_point(
        derived.open_field(run_dir / 'ICON_ENS_2026082300_T_2M.nc'),
        derived.open_field(run_dir / 'ICON_ENS_2026082300_RELHUM_2M.nc'))
    ncwrite.write_canonical(run_dir / 'ICON_ENS_2026082300_TD_2M.nc', dewpoint)
    dialog = derivedialog.DerivedDialog(None, near=run_dir)
    dialog.difference_radio.setChecked(True)
    assert (dialog.a_combo.currentData(), dialog.b_combo.currentData()) == ('T_2M', 'TD_2M')
    assert dialog.status.text() == ''
    dialog.close()


def test_an_unsubtractable_pair_is_refused_before_any_file_is_opened(qapp, run_dir):
    dialog = derivedialog.DerivedDialog(None, near=run_dir)
    dialog.difference_radio.setChecked(True)
    fields = [dialog.b_combo.itemData(i) for i in range(dialog.b_combo.count())]
    dialog.a_combo.setCurrentIndex(fields.index('T_2M'))
    dialog.b_combo.setCurrentIndex(fields.index('RELHUM_2M'))
    ok = dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled() and dialog.request() is None
    assert 'same units' in dialog.status.text()
    dialog.close()


def test_an_already_open_file_is_reused_rather_than_remapped(window, qapp):
    """Building on the open field must not map another 407 MB of the same bytes."""
    view = install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    assert view.temperature is window.opened[window.ds.temperature.path]


def test_a_pairing_failure_reaches_the_user_as_a_message(qapp, tmp_path, monkeypatch):
    synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    synth.humidity(tmp_path / 'ICON_ENS_2026082300_RELHUM_2M.nc', ny=3)   # wrong grid
    request = derivedialog.DerivedRequest(
        derivedialog.DEW_POINT,
        [tmp_path / 'ICON_ENS_2026082300_T_2M.nc',
         tmp_path / 'ICON_ENS_2026082300_RELHUM_2M.nc'], 'Dew point')
    worker = derivedialog.BuildWorker(request)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()                                    # synchronous: no thread needed here
    assert failures and 'Different grids' in failures[0]


# ---- saving ---------------------------------------------------------------------------
def test_save_is_disabled_until_something_is_open(qapp, tmp_path):
    w = MainWindow(None)
    try:
        assert not w.save_action.isEnabled()
    finally:
        w.close()


def test_the_save_worker_writes_a_reopenable_file(window, qapp, tmp_path):
    from imsicon.ui.main import WriteWorker
    view = install(window, qapp, derivedialog.DEW_POINT, 'Dew point')
    out = tmp_path / 'written' / 'TD.nc'
    out.parent.mkdir()
    worker = WriteWorker(view, str(out))
    worker.run()                                    # synchronous: no thread needed here
    from imsicon.dataset import EnsembleFile
    assert np.allclose(EnsembleFile(out).ens_frame(0), view.canonical_ens_frame(0),
                       equal_nan=True)


# ---- the download dialog ----------------------------------------------------------------
def test_the_download_dialog_lists_the_catalogue_for_a_run(qapp, monkeypatch):
    dialog = downloaddialog.DownloadDialog(None)
    files = downloaddialog.download.parse_listing(
        '<a href="ICON_ENS_2026082300_CAPE_ML.nc.bz2">ICON_ENS_2026082300_CAPE_ML.nc.bz2</a>'
        '<a href="ICON_ENS_2026082300_T_2M.nc.bz2">ICON_ENS_2026082300_T_2M.nc.bz2</a>'
        '<a href="ICON_ENS_2026082300_RELHUM_2M.nc.bz2">ICON_ENS_2026082300_RELHUM_2M.nc.bz2</a>')
    dialog._on_listed((object(), files))
    labels = [dialog.field_list.topLevelItem(i).text(1)
              for i in range(dialog.field_list.topLevelItemCount())]
    # Catalogue order, not alphabetical: CAPE and precipitation are what a forecaster
    # reaches for first.
    assert labels == ['CAPE_ML', 'T_2M', 'RELHUM_2M']
    assert dialog.field_list.topLevelItem(0).text(0).startswith('CAPE')
    dialog.close()


def test_the_dew_point_shortcut_ticks_exactly_the_two_fields_it_needs(qapp):
    dialog = downloaddialog.DownloadDialog(None)
    dialog._on_listed((object(), downloaddialog.download.parse_listing(
        ''.join(f'<a href="ICON_ENS_2026082300_{f}.nc.bz2">'
                f'ICON_ENS_2026082300_{f}.nc.bz2</a>'
                for f in ('CAPE_ML', 'T_2M', 'RELHUM_2M', 'TOT_PREC')))))
    dialog._select_dew_point()
    assert {entry.field for entry in dialog.selected()} == {'T_2M', 'RELHUM_2M'}
    dialog.close()


def test_nothing_can_be_downloaded_before_connecting(qapp):
    dialog = downloaddialog.DownloadDialog(None)
    assert not dialog.download_button.isEnabled()
    dialog.close()


def test_the_run_combo_shows_the_run_it_is_actually_using(qapp, tmp_path):
    """The combo lists newest-first, so an older open file must not silently disagree."""
    synth.pair(tmp_path, run='2026082300')
    synth.pair(tmp_path, run='2026082400')
    dialog = derivedialog.DerivedDialog(None, near=tmp_path, run='2026082300')
    assert dialog.run_combo.currentData() == '2026082300'
    assert dialog.run == '2026082300'
    assert '2026-08-23' in dialog.run_combo.currentText()
    for path in dialog.request().paths:
        assert '2026082300' in path.name
    dialog.close()


def test_an_unknown_run_falls_back_to_the_newest_on_disk(qapp, tmp_path):
    synth.pair(tmp_path, run='2026082400')
    dialog = derivedialog.DerivedDialog(None, near=tmp_path, run='2020010100')
    assert dialog.run == '2026082400'
    assert dialog.request() is not None
    dialog.close()


# ---- "Map shows": the field selector ----------------------------------------------------
def test_the_map_shows_combo_offers_the_derived_maps(window, qapp):
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    labels = [window.field_combo.itemText(i) for i in range(window.field_combo.count())]
    assert keys == ['base', derivedialog.DEW_POINT, derivedialog.DEPRESSION]
    assert labels[0].startswith('T_2M')
    assert 'TD_2M' in labels[1]
    # The depression is named the way a forecaster reads it, not by its machine name.
    assert labels[2].startswith('T-Td') and 'T_2M-TD_2M' not in labels[2]
    assert window.field_combo.isEnabled()


def test_only_the_open_field_is_offered_when_the_humidity_is_missing(qapp, tmp_path):
    synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc')
    w = MainWindow(str(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
    w.show()
    settle(qapp)
    finish_scan(w, qapp)
    try:
        assert [w.field_combo.itemData(i) for i in range(w.field_combo.count())] == ['base']
        assert not w.field_combo.isEnabled()
    finally:
        w.close()


def test_picking_a_derived_map_from_the_combo_installs_it(window, qapp):
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    window.field_combo.setCurrentIndex(keys.index(derivedialog.DEPRESSION))
    settle(qapp)
    if window.builder is not None:
        window.builder.wait(30000)
    settle(qapp)
    finish_scan(window, qapp)
    assert window.ds.display_name == derived.DEPRESSION_NAME
    assert window.ds.field == 'T_2M-TD_2M'          # the machine name is unchanged
    assert window.map.plot.titleLabel.text.startswith('T-Td')
    assert window.plot.getAxis('left').labelText == 'T-Td'


def test_switching_back_to_the_file_reuses_the_view_already_open(window, qapp):
    base = window.ds
    install(window, qapp, derivedialog.DEPRESSION, 'Depression')
    assert window.ds is not base
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    window.field_combo.setCurrentIndex(keys.index('base'))
    settle(qapp)
    assert window.ds is base                        # not reopened, the same object
    assert window.map.plot.titleLabel.text.startswith('T_2M')


def test_the_combo_names_whatever_is_on_screen_even_for_an_ad_hoc_difference(window, qapp):
    """A `--difference A B` view is not one of the standard entries, and the combo must
    still not claim the map is showing the plain field."""
    directory = window.ds.path.parent
    synth.write_nc3(directory / 'ICON_ENS_2026082300_T_S.nc', 'T_S', 'K',
                    np.full((6, 3, 4, 5), 295.0, dtype=np.float32),
                    history=synth.HISTORY_TEMPLATE.format(field='T_S'))
    request = derivedialog.DerivedRequest(
        derivedialog.DIFFERENCE,
        [directory / 'ICON_ENS_2026082300_T_2M.nc',
         directory / 'ICON_ENS_2026082300_T_S.nc'], 'T_2M - T_S')
    view = derivedialog.build(request, dict(window.opened))
    window._install(view, request.title)
    settle(qapp)
    finish_scan(window, qapp)
    assert view.derived_kind == derivedialog.DIFFERENCE
    assert window.field_combo.currentData() == derivedialog.DIFFERENCE
    assert window.field_combo.currentText().startswith('T_2M-T_S')
    # ...and the standard entries are still there to switch back to.
    keys = [window.field_combo.itemData(i) for i in range(window.field_combo.count())]
    assert 'base' in keys and derivedialog.DEPRESSION in keys


def test_a_derived_view_carries_its_kind_however_it_was_built(run_dir):
    """`build` stamps it, so the dialog, the combo and --derive all agree."""
    paths = [run_dir / 'ICON_ENS_2026082300_T_2M.nc',
             run_dir / 'ICON_ENS_2026082300_RELHUM_2M.nc']
    for kind in (derivedialog.DEW_POINT, derivedialog.DEPRESSION):
        view = derivedialog.build(derivedialog.DerivedRequest(kind, paths, kind))
        assert view.derived_kind == kind
        assert view.derived_request.kind == kind
