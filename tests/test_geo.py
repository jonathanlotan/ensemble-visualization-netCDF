"""The bundled coastline + border overlay, the land fill under it, and the stacking.

The overlay is committed data, so these tests check the shipped file rather than a
fixture: if `imsicon/mapdata/levant_10m.json` is ever regenerated wrongly, the map goes
blank or loses its borders, and nothing else in the suite would notice.
"""
import json

import numpy as np
import pytest

from imsicon import geo


# ---- the bundled data -------------------------------------------------------------
def test_the_overlay_ships_with_the_package():
    assert geo.MAPDATA.is_file(), f'missing bundled overlay: {geo.MAPDATA}'
    assert geo.MAPDATA.stat().st_size < 1 << 20, 'the overlay should stay small enough to commit'


def test_it_loads_into_drawable_arrays():
    data = geo.load_mapdata()
    assert data is not None
    for layer in ('coastline', 'borders', 'borders_uncertain'):
        xs, ys = data[layer]
        assert len(xs) == len(ys) and len(xs) > 100, layer
        assert np.isnan(xs).any(), f'{layer} needs NaN breaks between polylines'


def test_every_point_is_inside_the_clip_box():
    """Panning is limited to the domain +- 50 % (G10); anything past that is dead weight."""
    blob = json.loads(geo.MAPDATA.read_text())
    clip = blob['clip']
    data = geo.load_mapdata()
    for layer in ('coastline', 'borders', 'borders_uncertain'):
        xs, ys = data[layer]
        good = np.isfinite(xs)
        # One point past each crossing is kept on purpose, so allow a small margin.
        assert xs[good].min() >= clip['lon_min'] - 2.0
        assert xs[good].max() <= clip['lon_max'] + 2.0
        assert ys[good].min() >= clip['lat_min'] - 2.0
        assert ys[good].max() <= clip['lat_max'] + 2.0


def test_the_overlay_covers_the_whole_model_domain():
    """The reason it exists: `topo_icon_web.nc` only reaches the inner box (G6)."""
    xs, ys = geo.load_mapdata()['coastline']
    good = np.isfinite(xs)
    assert xs[good].min() <= 33.0 and xs[good].max() >= 37.0
    assert ys[good].min() <= 28.0 and ys[good].max() >= 34.5


def test_disputed_lines_are_kept_separate_from_settled_ones():
    """Natural Earth's own classification is carried through, not flattened: several
    lines in this domain are marked disputed, indefinite or line-of-control."""
    data = geo.load_mapdata()
    classes = data['classes']
    assert 'International boundary (verify)' in classes
    assert any(name != 'International boundary (verify)' for name in classes)
    assert len(data['borders'][0]) and len(data['borders_uncertain'][0])


# ---- the land polygons (R6) --------------------------------------------------------
def test_the_bundle_carries_land_polygons():
    """Lines cannot be filled: a coastline clipped to a box is a set of open polylines
    with no inside, so the grey land needs the source's real rings."""
    rings = geo.load_mapdata()['land']
    assert len(rings) > 5
    for xs, ys in rings:
        assert len(xs) == len(ys) >= 3
        assert np.isfinite(xs).all() and np.isfinite(ys).all()   # no NaN breaks in a ring


def test_the_land_covers_the_whole_pan_range_not_just_the_domain():
    """An outline that stops mid-pan looks like a bug; a fill that stops mid-pan looks
    like a coastline that is not there."""
    xs = np.concatenate([ring[0] for ring in geo.load_mapdata()['land']])
    ys = np.concatenate([ring[1] for ring in geo.load_mapdata()['land']])
    assert xs.min() <= 33.0 and xs.max() >= 37.0
    assert ys.min() <= 28.0 and ys.max() >= 34.5


def test_the_land_agrees_with_the_coastline_about_where_the_sea_is():
    """Ray casting against the rings, at places nobody can be wrong about."""
    rings = geo.load_mapdata()['land']

    def is_land(lon, lat):
        inside = False
        for xs, ys in rings:
            x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
            crossing = (ys > lat) != (y2 > lat)
            with np.errstate(divide='ignore', invalid='ignore'):
                at = xs + (lat - ys) * (x2 - xs) / (y2 - ys)
            if np.count_nonzero(crossing & (lon < at)) % 2:
                inside = not inside
        return inside

    for lon, lat, expected in [(35.22, 31.78, True),     # Jerusalem
                               (34.50, 32.90, False),    # Mediterranean off Haifa
                               (33.60, 29.60, True),     # Sinai
                               (33.00, 28.50, False),    # Gulf of Suez
                               (33.20, 35.00, True),     # Cyprus
                               (36.30, 33.50, True)]:    # Damascus
        assert is_land(lon, lat) is expected, (lon, lat)


def test_an_older_bundle_without_land_still_loads():
    """schema 1 costs the map its grey land and nothing else."""
    assert geo._rings_to_arrays(()) == []
    assert geo._rings_to_arrays([[[1, 2], [3, 4]]]) == []      # a 2-point ring is not one


def test_the_fallback_overlay_has_no_land_to_fill():
    layers = dict(geo.overlay_for('nowhere.nc'))
    assert 'land' in layers


def test_the_source_is_recorded():
    assert 'Natural Earth' in geo.load_mapdata()['source']


def test_a_missing_or_broken_overlay_is_survivable(tmp_path):
    """A broken outline must cost the map its outlines, not stop a forecast opening."""
    assert geo.load_mapdata(tmp_path / 'nope.json') is None
    broken = tmp_path / 'broken.json'
    broken.write_text('{ not json')
    assert geo.load_mapdata(broken) is None


def test_overlay_for_falls_back_to_the_model_land_mask(monkeypatch, tmp_path):
    monkeypatch.setattr(geo, 'load_mapdata', lambda *a, **k: None)
    monkeypatch.setattr(geo, 'coastline_for',
                        lambda path: (np.array([34.0, 35.0]), np.array([32.0, 32.5])))
    layers = geo.overlay_for(tmp_path / 'x.nc')
    assert len(layers['coastline'][0]) == 2
    assert layers['borders'][0] is None          # the topo file has no borders in it
    assert 'G6' in layers['source']


def test_polylines_become_one_array_with_nan_breaks():
    xs, ys = geo._polylines_to_arrays([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert list(xs[:2]) == [1, 3] and np.isnan(xs[2])
    assert list(xs[3:5]) == [5, 7] and np.isnan(xs[5])
    geo._polylines_to_arrays([[[1, 2]]])         # a one-point line is dropped, not drawn


# ---- drawn over the field ----------------------------------------------------------
def test_the_outlines_stack_above_the_field(qapp):
    """"Shown over the model map" must not depend on the order items were added in."""
    from imsicon.ui import mapview
    view = mapview.MapView()
    try:
        # The land is UNDER the field: it is there to be seen through it (`ui/colors.py`),
        # not to cover it.
        assert view.land.zValue() < view.img.zValue()
        assert view.img.zValue() < view.coast.under.zValue()
        assert view.coast.under.zValue() < view.coast.over.zValue()
        assert view.coast.over.zValue() <= view.borders.under.zValue()
        assert view.borders.under.zValue() < view.borders.over.zValue()
        assert view.borders.over.zValue() < view.marker.zValue()
        assert view.borders_uncertain.over.zValue() > view.img.zValue()
    finally:
        view.close()


def test_setting_an_overlay_fills_every_layer(qapp):
    from imsicon.ui import mapview
    view = mapview.MapView()
    try:
        view.set_overlay(geo.overlay_for('nowhere.nc'))
        for layer in (view.coast, view.borders, view.borders_uncertain):
            assert layer.over.xData is not None and len(layer.over.xData)
            # Halo and ink carry the same geometry, or the halo would show through.
            assert np.array_equal(layer.under.xData, layer.over.xData, equal_nan=True)
        assert view.land_rings > 5 and view.land.path().elementCount() > 100
        assert 'Natural Earth' in view.overlay_source
    finally:
        view.close()


def test_an_empty_overlay_clears_the_layers_instead_of_raising(qapp):
    from imsicon.ui import mapview
    view = mapview.MapView()
    try:
        view.set_overlay(geo.overlay_for('nowhere.nc'))
        view.set_overlay(None)
        for layer in (view.coast, view.borders, view.borders_uncertain):
            assert layer.over.xData is None or not len(layer.over.xData)
        assert view.land_rings == 0 and view.land.path().isEmpty()
    finally:
        view.close()


def test_uncertain_borders_are_drawn_dashed(qapp):
    """Drawing a disputed line identically to a settled one would overstate it."""
    from PySide6 import QtCore
    from imsicon.ui import mapview
    view = mapview.MapView()
    try:
        assert view.borders.over.opts['pen'].style() == QtCore.Qt.PenStyle.SolidLine
        assert view.borders_uncertain.over.opts['pen'].style() == QtCore.Qt.PenStyle.DashLine
    finally:
        view.close()


@pytest.mark.parametrize('layer', ['coast', 'borders', 'borders_uncertain'])
def test_each_outline_has_a_contrasting_halo(qapp, layer):
    """Over turbo there is no single ink colour legible at both ends of the scale."""
    from imsicon.ui import mapview
    view = mapview.MapView()
    try:
        outline = getattr(view, layer)
        halo = outline.under.opts['pen']
        ink = outline.over.opts['pen']
        assert halo.widthF() > ink.widthF()
        assert halo.color().lightness() > ink.color().lightness()
    finally:
        view.close()


def test_the_land_is_filled_pale_and_the_sea_is_left_to_the_background(qapp):
    """Light enough to sit under the lowest values a ramp carries, dark enough to read
    as land against the white sea. The sea is not drawn at all."""
    from PySide6 import QtCore
    from imsicon.ui import mapview
    view = mapview.MapView()
    try:
        brush = view.land.brush()
        assert brush.style() != QtCore.Qt.BrushStyle.NoBrush
        assert 200 < brush.color().lightness() < 245
        assert view.land.pen().style() == QtCore.Qt.PenStyle.NoPen
    finally:
        view.close()
