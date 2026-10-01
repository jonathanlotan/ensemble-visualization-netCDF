"""Isolines and the T-Td sort scale at the window level (R5).

The requirement is about what a forecaster sees, so these run on a real `MapView` with a
real ViewBox and colorbar: that `isolines.py` computes the right lines is a different
claim from the map actually drawing them, keeping them when time moves, and taking them
down when the field changes.
"""
import numpy as np
import pytest
from PySide6 import QtWidgets

import synth
from imsicon import derived, ingest, isolines
from imsicon.ui import derivedialog, downloaddialog
from imsicon.ui.main import MainWindow


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch):
    """The two guards test_ui_derived.py explains: an unpatched modal file dialog hangs
    the offscreen run forever (G30), and the download dialog must never read the
    developer's keychain."""
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: ('', '')))
    monkeypatch.setattr(downloaddialog.download, 'stored_credentials', lambda: None)
    monkeypatch.setattr(downloaddialog.download, 'keyring_module', lambda: None)


@pytest.fixture(autouse=True)
def only_this_test_s_files(monkeypatch):
    """Field discovery must not see the developer's own ./data or cache (G25's reasoning:
    the fixtures use run 2026082300, the same number the 407 MB reference file carries)."""
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


@pytest.fixture
def run_dir(tmp_path):
    """A T_2M / RELHUM_2M pair with enough range to hold a dozen isolines.

    The temperature sweeps ~19 degC across the domain and the humidity from saturated to
    middling, so the depression crosses the 2 degC sort threshold inside one frame -- the
    band has to have something on both sides of it to be worth testing -- while staying
    inside the line cap, so the 0.5 degC interval is the one actually drawn.
    """
    nt, nm, ny, nx = 4, 3, 24, 18
    step = np.arange(nt)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    warm = 288.0 + 0.55 * y + 0.35 * x + 1.5 * step
    moist = np.clip(99.0 - 2.6 * x - 0.2 * y - 1.0 * step, 45.0, 100.0)
    from imsicon.ncwrite import write_nc3

    def members(base, spread):
        return np.stack([np.broadcast_to(base, (nt, ny, nx)) + spread * (m - 1)
                         for m in range(nm)], axis=1)

    write_nc3(tmp_path / 'ICON_ENS_2026082300_T_2M.nc', 'T_2M', 'K',
              members(warm, 0.4), history=synth.HISTORY_TEMPLATE.format(field='T_2M'),
              long_name='2m temperature', standard_name='air_temperature')
    write_nc3(tmp_path / 'ICON_ENS_2026082300_RELHUM_2M.nc', 'RELHUM_2M', '%',
              np.clip(members(moist, 1.5), 1.0, 100.0),
              history=synth.HISTORY_TEMPLATE.format(field='RELHUM_2M'),
              long_name='relative humidity in 2m', standard_name='relative_humidity')
    return tmp_path


@pytest.fixture
def window(qapp, run_dir):
    w = MainWindow(str(run_dir / 'ICON_ENS_2026082300_T_2M.nc'))
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


def drawn(window):
    """(ordinary points, heavy points) currently on the map."""
    def count(layer):
        xs = layer.over.getData()[0]
        return 0 if xs is None else int(np.size(xs))
    return count(window.map.isoline), count(window.map.isoline_heavy)


# ---- the temperature map ----------------------------------------------------------------
def test_a_temperature_map_is_contoured_every_degree(window):
    assert window.isolines_check.isEnabled() and window.isolines_check.isChecked()
    ordinary, heavy = drawn(window)
    assert ordinary > 0
    levels = window.map.isoline_levels
    assert levels.size > 3
    assert window.map.isoline_step == pytest.approx(1.0)
    assert np.allclose(levels % 1.0, 0.0)           # whole degrees Celsius
    assert np.allclose(np.diff(levels), 1.0)
    assert 'isolines 1 °C' in window.map.plot.titleLabel.text


def test_every_fifth_line_is_drawn_heavier(window):
    heavy_levels = [level for level in window.map.isoline_levels if level % 5 == 0]
    assert heavy_levels                            # the fixture spans more than 5 degC
    assert drawn(window)[1] > 0
    # and the two layers are different items, so one can be styled without the other
    assert window.map.isoline.over is not window.map.isoline_heavy.over


def test_unticking_removes_the_lines_and_leaves_the_colours(window, qapp):
    window.isolines_check.setChecked(False)
    settle(qapp)
    assert drawn(window) == (0, 0)
    assert window.map.isoline_levels.size == 0
    assert window.map.img.image is not None
    assert 'isolines' not in window.map.plot.titleLabel.text
    window.isolines_check.setChecked(True)
    settle(qapp)
    assert drawn(window)[0] > 0


def test_the_lines_follow_the_time_step(window, qapp):
    first = window.map.isoline.over.getData()[0].copy()
    window.set_time(3)
    settle(qapp)
    later = window.map.isoline.over.getData()[0]
    assert later.size and not (first.size == later.size and np.allclose(
        first, later, equal_nan=True))
    assert np.allclose(window.map.isoline_levels % 1.0, 0.0)


def test_the_lines_follow_the_aggregation(window, qapp):
    mean = window.map.isoline_levels.copy()
    index = [window.agg_combo.itemData(i) for i in range(window.agg_combo.count())]
    window.agg_combo.setCurrentIndex(index.index('spread'))
    settle(qapp)
    # the spread of this ensemble is well under a degree, so a 1 degC contour of it has
    # at most one line -- the point is that it was recomputed, not that it vanished
    assert window.map.isoline_levels.size < mean.size


def test_a_field_with_no_interval_disables_the_control(window, qapp):
    ds = choose(window, qapp, 'file:RELHUM_2M')
    assert ds.field == 'RELHUM_2M'
    assert not window.isolines_check.isEnabled()
    assert 'not contoured' in window.isolines_check.toolTip()
    assert drawn(window) == (0, 0)
    assert 'isolines' not in window.map.plot.titleLabel.text


def test_switching_units_keeps_the_same_isotherms(window, qapp):
    """G15 in the place it is easiest to get wrong: "every 1 degree Celsius" in degF is
    the same lines relabelled, not a new set on whole Fahrenheit."""
    celsius = window.map.isoline_levels.copy()
    window.units_combo.setCurrentText('°F')
    settle(qapp)
    fahrenheit = window.map.isoline_levels
    assert window.map.isoline_step == pytest.approx(1.8)
    assert np.allclose(celsius * 1.8 + 32.0, fahrenheit)
    assert 'isolines 1.8 °F' in window.map.plot.titleLabel.text

    window.units_combo.setCurrentText('K')
    settle(qapp)
    assert np.allclose(celsius + 273.15, window.map.isoline_levels)


# ---- the dew point depression -----------------------------------------------------------
def test_the_depression_is_contoured_every_half_degree(window, qapp):
    ds = choose(window, qapp, derivedialog.DEPRESSION)
    assert ds.display_name == derived.DEPRESSION_NAME
    assert window.map.isoline_step == pytest.approx(0.5)
    levels = window.map.isoline_levels
    assert levels.size > 3 and np.allclose(np.diff(levels), 0.5)
    assert np.allclose(levels % 0.5, 0.0)
    assert 'isolines 0.5 °C' in window.map.plot.titleLabel.text
    assert drawn(window)[0] > 0 and drawn(window)[1] > 0     # heavy every whole degree


def test_the_dew_point_is_contoured_like_a_temperature(window, qapp):
    choose(window, qapp, derivedialog.DEW_POINT)
    assert window.map.isoline_step == pytest.approx(1.0)
    assert np.allclose(window.map.isoline_levels % 1.0, 0.0)


# ---- the sort band ----------------------------------------------------------------------
def test_sort_is_offered_on_the_depression_and_nowhere_else(window, qapp):
    assert not window.sort_check.isEnabled()
    assert 'applies to the dew point depression' in window.sort_check.toolTip()
    choose(window, qapp, derivedialog.DEW_POINT)
    assert not window.sort_check.isEnabled()
    choose(window, qapp, derivedialog.DEPRESSION)
    assert window.sort_check.isEnabled() and not window.sort_check.isChecked()
    assert '2 °C' in window.sort_check.toolTip()
    assert 'red' in window.sort_check.toolTip()


def test_sorting_colours_only_the_band_below_two_degrees(window, qapp):
    choose(window, qapp, derivedialog.DEPRESSION)
    diverging = window.map.cbar.levels()
    assert diverging[0] < 0                      # R3's symmetric difference scale
    window.sort_check.setChecked(True)
    settle(qapp)
    assert window.map.cbar.levels() == (0.0, 2.0)
    lut = window.map.cmap.getLookupTable(0.0, 1.0, 3, alpha=True)
    assert tuple(lut[0]) == (215, 25, 28, 255)   # 0 degC: red
    # 2 degC and drier: white, and not painted at all -- the colours run out for real, so
    # the grey land under the map shows through instead of being covered over.
    assert tuple(lut[-1]) == (255, 255, 255, 0)
    assert tuple(lut[1])[0] > 240 and tuple(lut[1])[2] < 130   # 1 degC: yellow-orange
    assert lut[1][3] > 100                       # inside the band, still painted
    assert 'sorted: colour only below 2 °C' in window.map.plot.titleLabel.text


def test_unsorting_puts_the_difference_scale_back(window, qapp):
    choose(window, qapp, derivedialog.DEPRESSION)
    before = window.map.cbar.levels()
    window.sort_check.setChecked(True)
    settle(qapp)
    window.sort_check.setChecked(False)
    settle(qapp)
    assert window.map.cbar.levels() == before
    assert 'sorted' not in window.map.plot.titleLabel.text
    assert window.cmap_combo.isEnabled() and window.scale_combo.isEnabled()
    assert 'Untick Sort' not in window.cmap_combo.toolTip()     # nor a stale explanation


def test_while_sorting_the_colour_and_scale_controls_are_disabled(window, qapp):
    """They are not silently ignored: the band IS the colours and IS the range."""
    choose(window, qapp, derivedialog.DEPRESSION)
    assert window.cmap_combo.isEnabled() and window.scale_combo.isEnabled()
    window.sort_check.setChecked(True)
    settle(qapp)
    assert not window.cmap_combo.isEnabled()
    assert not window.scale_combo.isEnabled()
    assert 'Untick Sort' in window.cmap_combo.toolTip()


def test_the_band_scales_with_the_units(window, qapp):
    choose(window, qapp, derivedialog.DEPRESSION)
    window.sort_check.setChecked(True)
    settle(qapp)
    window.units_combo.setCurrentText('°F')
    settle(qapp)
    low, high = window.map.cbar.levels()
    assert (low, round(high, 6)) == (0.0, 3.6)   # 2 degC of difference is 3.6 degF
    assert 'below 3.6 °F' in window.map.plot.titleLabel.text


def test_a_spread_map_is_never_sorted(window, qapp):
    """A max-minus-min across the members is a width, not a depression; colouring it
    against the fog thresholds would read as a forecast nobody made."""
    choose(window, qapp, derivedialog.DEPRESSION)
    window.sort_check.setChecked(True)
    settle(qapp)
    index = [window.agg_combo.itemData(i) for i in range(window.agg_combo.count())]
    window.agg_combo.setCurrentIndex(index.index('spread'))
    settle(qapp)
    assert window.map.cbar.levels() != (0.0, 2.0)
    assert 'sorted' not in window.map.plot.titleLabel.text
    assert window.cmap_combo.isEnabled()


def test_sort_is_cleared_when_the_map_moves_to_a_field_without_one(window, qapp):
    choose(window, qapp, derivedialog.DEPRESSION)
    window.sort_check.setChecked(True)
    settle(qapp)
    choose(window, qapp, 'base')
    assert not window.sort_check.isChecked() and not window.sort_check.isEnabled()
    assert window.cmap_combo.isEnabled()
    assert 'sorted' not in window.map.plot.titleLabel.text


def test_isolines_and_sort_are_independent(window, qapp):
    """The lines still run through the uncoloured air, which is most of the point: they
    are what says how far past the threshold a dry area is."""
    choose(window, qapp, derivedialog.DEPRESSION)
    window.sort_check.setChecked(True)
    settle(qapp)
    assert drawn(window)[0] > 0
    assert (window.map.isoline_levels > 2.0).any()


def test_a_frame_with_too_many_lines_says_the_interval_it_settled_for(
        window, qapp, monkeypatch):
    """G35: the cap coarsens the step so the map is not a hatch pattern, and the title
    reports the interval the lines were DRAWN at, not the one that was asked for."""
    monkeypatch.setattr(isolines, 'MAX_LINES', 3)
    window.refresh_map()
    settle(qapp)
    assert window.map.isoline_step > 1.0
    assert window.map.isoline_levels.size <= 3
    text = window.map.plot.titleLabel.text
    assert f'isolines {window.map.isoline_step:g} °C' in text
    assert 'too many lines at 1 °C' in text


# ---- the layout trap the notes uncovered ------------------------------------------------
def test_a_long_title_does_not_crop_the_map(window, qapp):
    """G36. A LabelItem's minimum width is its text width and a GraphicsLayout honours it,
    so before the title was wrapped, turning Sort on widened the plot column, swallowed
    the colorbar and -- the ViewBox being aspect-locked -- cropped two thirds of the
    latitude out of the map (G10 reached from the other end)."""
    before = window.map.plot.vb.viewRange()
    window.map.set_title('T-Td [°C] - Ensemble mean - 2026-08-23 02:00Z  (+2 h)  |  '
                         'isolines 0.5 °C  |  sorted: colour only below 2 °C  |  '
                         'and a great deal more text than any real title would carry')
    settle(qapp)
    after = window.map.plot.vb.viewRange()
    assert np.allclose(before, after)
    assert window.map.cbar.isVisible()


# ---- how close the lines are (R5.9) -----------------------------------------------------
def set_spacing(window, app, step):
    """Move the slider the way a hand does -- by notch, not by calling the view."""
    from imsicon import isolines as iso
    window.isoline_step_slider.setValue(iso.STEP_CHOICES.index(step))
    settle(app)


def test_the_spacing_slider_starts_on_the_field_s_own_interval(window):
    from imsicon import isolines as iso
    assert window.isoline_step_slider.isEnabled()
    assert window.isoline_step_slider.value() == iso.STEP_CHOICES.index(1.0)
    assert window.isoline_step_label.text() == '1 °C'
    assert window.map.isoline_step == pytest.approx(1.0)


def test_moving_the_slider_spaces_the_lines_further_apart(window, qapp):
    close = window.map.isoline_levels.size
    set_spacing(window, qapp, 2.0)
    levels = window.map.isoline_levels
    assert window.map.isoline_step == pytest.approx(2.0)
    assert levels.size < close
    assert np.allclose(np.diff(levels), 2.0)
    assert np.allclose(levels % 2.0, 0.0)            # still anchored on 0 °C
    assert window.isoline_step_label.text() == '2 °C'
    assert 'isolines 2 °C' in window.map.plot.titleLabel.text
    assert drawn(window)[0] > 0

    set_spacing(window, qapp, 4.0)
    assert window.map.isoline_step == pytest.approx(4.0)
    assert window.map.isoline_levels.size < levels.size


def test_the_closest_spacing_draws_more_lines_and_heavies_every_whole_degree(window, qapp):
    """At 0.5 °C "every 5th" would put the heavy line on 2.5 °C, which is no landmark:
    the emphasis follows the spacing so it lands on a whole degree instead."""
    ordinary, _heavy = drawn(window)
    set_spacing(window, qapp, 0.5)
    assert window.map.isoline_step == pytest.approx(0.5)
    assert drawn(window)[0] > ordinary
    heavy = [level for level in window.map.isoline_levels if level % 1.0 == 0.0]
    assert len(heavy) > 2 and drawn(window)[1] > 0
    assert 'isolines 0.5 °C' in window.map.plot.titleLabel.text


def test_unticking_isolines_puts_the_slider_out_of_reach_and_gives_it_back(window, qapp):
    set_spacing(window, qapp, 3.0)
    window.isolines_check.setChecked(False)
    settle(qapp)
    assert not window.isoline_step_slider.isEnabled()
    assert window.isoline_step_label.text() == ''
    assert drawn(window) == (0, 0)

    window.isolines_check.setChecked(True)
    settle(qapp)
    assert window.isoline_step_slider.isEnabled()
    assert window.map.isoline_step == pytest.approx(3.0)   # the choice was kept, not reset
    assert np.allclose(window.map.isoline_levels % 3.0, 0.0)


def test_the_label_follows_the_units_while_the_lines_stay_where_they_were(window, qapp):
    """The notches are canonical degrees and the label is in the units on screen, which
    is the point of G15: 2 °C reads as 3.6 °F over exactly the same isotherms."""
    from imsicon import isolines as iso
    set_spacing(window, qapp, 2.0)
    celsius = window.map.isoline_levels.copy()
    window.units_combo.setCurrentText('°F')
    settle(qapp)
    assert window.isoline_step_label.text() == '3.6 °F'
    assert window.isoline_step_slider.value() == iso.STEP_CHOICES.index(2.0)
    assert window.map.isoline_step == pytest.approx(3.6)
    assert np.allclose(celsius * 1.8 + 32.0, window.map.isoline_levels)


def test_each_field_keeps_the_spacing_it_was_last_read_at(window, qapp):
    """Per field, because 2 °C on a temperature map and 2 °C on a depression are
    different readings -- so flipping between them must neither carry one choice onto the
    other nor throw the first one away."""
    from imsicon import isolines as iso
    set_spacing(window, qapp, 3.0)

    choose(window, qapp, derivedialog.DEPRESSION)
    assert window.isoline_step_slider.value() == iso.STEP_CHOICES.index(0.5)
    assert window.map.isoline_step == pytest.approx(0.5)
    set_spacing(window, qapp, 2.0)

    choose(window, qapp, 'base')
    assert window.ds.field == 'T_2M'
    assert window.isoline_step_slider.value() == iso.STEP_CHOICES.index(3.0)
    assert window.map.isoline_step == pytest.approx(3.0)

    choose(window, qapp, derivedialog.DEPRESSION)
    assert window.map.isoline_step == pytest.approx(2.0)


def test_a_field_with_no_interval_leaves_the_slider_out_of_reach(window, qapp):
    choose(window, qapp, 'file:RELHUM_2M')
    assert not window.isoline_step_slider.isEnabled()
    assert window.isoline_step_label.text() == ''
    assert 'not contoured' in window.isoline_step_slider.toolTip()


# ---- R9: the slider offers the field's own ladder -------------------------------------------
iso = isolines


@pytest.fixture
def height_run(tmp_path):
    """A geopotential column beside a temperature one, on five levels."""
    nt, ny, nx = 3, 24, 18
    lvls = (1000, 925, 850, 700, 500)
    p = np.asarray(lvls, float)[None, :, None, None]
    y = np.arange(ny)[None, None, :, None]
    x = np.arange(nx)[None, None, None, :]
    t = np.arange(nt)[:, None, None, None]
    z = (8000.0 * np.log(1000.0 / p) + 6.0 * y + 4.0 * x + 3.0 * t) * iso.G0
    synth.pressure_field(tmp_path / 'IE_2026083100_geopot.nc', 'geopot', 'm2 s-2',
                         n_times=nt, ny=ny, nx=nx, levels=lvls,
                         values=np.broadcast_to(z, (nt, len(lvls), ny, nx)))
    synth.pressure_field(tmp_path / 'IE_2026083100_temp.nc', 'temp', 'K',
                         n_times=nt, ny=ny, nx=nx, levels=lvls)
    return tmp_path


@pytest.fixture
def height(qapp, height_run):
    w = MainWindow(str(height_run / 'IE_2026083100_geopot.nc'))
    w.resize(1200, 800)
    w.show()
    settle(qapp)
    finish_scan(w, qapp)
    yield w
    w.close()


def test_a_height_chart_offers_gpm_spacings_and_opens_at_40(height):
    slider = height.isoline_step_slider
    assert height.isolines_check.isEnabled() and slider.isEnabled()
    assert slider.maximum() == len(iso.HEIGHT.steps) - 1
    assert slider.value() == iso.HEIGHT.steps.index(40.0)
    assert height.isoline_step_label.text() == '40 gpm'
    assert height.map.isoline_step == pytest.approx(40.0)
    assert 'gpm' in slider.toolTip() and '120' in slider.toolTip()
    assert 'isolines 40 gpm' in height.map.plot.titleLabel.text


def test_moving_the_slider_on_a_height_chart_spaces_the_lines_in_gpm(height, qapp):
    height.isoline_step_slider.setValue(iso.HEIGHT.steps.index(120.0))
    settle(qapp)
    assert height.isoline_step_label.text() == '120 gpm'
    assert height.map.isoline_step == pytest.approx(120.0)
    assert np.all(np.isclose(height.map.isoline_levels % 120.0, 0.0)
                  | np.isclose(height.map.isoline_levels % 120.0, 120.0))
    assert 'isolines 120 gpm' in height.map.plot.titleLabel.text


def test_decametres_relabel_the_same_lines(height, qapp):
    height.isoline_step_slider.setValue(iso.HEIGHT.steps.index(60.0))
    settle(qapp)
    in_gpm = np.array(height.map.isoline_levels)
    height.units_combo.setCurrentText('dam')
    settle(qapp)
    assert height.isoline_step_label.text() == '6 dam'
    assert height.map.isoline_step == pytest.approx(6.0)
    np.testing.assert_allclose(np.array(height.map.isoline_levels) * 10.0, in_gpm)
    assert height.isoline_step_slider.value() == iso.HEIGHT.steps.index(60.0)


def test_switching_to_a_temperature_puts_the_degree_ladder_back(height, qapp):
    keys = [height.field_combo.itemData(i) for i in range(height.field_combo.count())]
    height.field_combo.setCurrentIndex(keys.index('file:temp'))
    settle(qapp)
    for worker in (height.decompressor, height.builder, height._height_builder):
        if worker is not None and worker.isRunning():
            worker.wait(30000)
    settle(qapp)
    assert height.ds.field == 'temp'
    assert height.isoline_step_slider.maximum() == len(iso.STEP_CHOICES) - 1
    assert height.isoline_step_slider.value() == iso.STEP_CHOICES.index(1.0)
    assert height.isoline_step_label.text() == '1 °C'


def test_the_command_line_spacing_is_snapped_to_the_field_s_ladder(height, capsys):
    """`--isoline-step 50` on a height chart means 50 gpm, and the nearest notch is 40."""
    import argparse
    from imsicon.__main__ import _apply_display
    args = argparse.Namespace(derive=None, difference=None, units=None, rate=None,
                              isolines=None, isoline_step=50.0, level=None, barbs=None,
                              topo=None, profile=None, sort=False, write=None,
                              screenshot=None)
    _apply_display(height, args, None)
    assert height.isoline_step_label.text() == '40 gpm'
    assert 'using 40' in capsys.readouterr().err
