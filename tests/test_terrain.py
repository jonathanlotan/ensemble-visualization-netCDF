"""R9: the bundled elevation grid and the shaded relief built from it -- no window.

The relief is an arithmetic claim before it is a drawing: flat ground must come out
exactly 1 (so Multiply leaves the field alone), a slope facing the light must not darken,
one facing away must, and the sea must be transparent whatever the sea floor does.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from imsicon import terrain

ROOT = Path(__file__).resolve().parent.parent


# ---- the shipped bundle ------------------------------------------------------------------
def test_the_shipped_bundle_loads_and_spans_the_whole_pan_range():
    data = terrain.load_terrain()
    assert data is not None, 'imsicon/mapdata/levant_etopo1.npz is missing or unreadable'
    lat, lon, z = data['lat'], data['lon'], data['z']
    assert z.shape == (lat.size, lon.size) and z.dtype == np.int16
    # The union of the two products' pan ranges (R7: the deterministic domain is bigger).
    assert lat[0] <= 24.5 and lat[-1] >= 39.5
    assert lon[0] <= 29.5 and lon[-1] >= 39.5
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)
    assert 'ETOPO1' in data['source']
    assert data['meta'].get('units') == 'm'


def test_the_bundle_is_the_real_relief_at_places_nobody_can_be_wrong_about():
    data = terrain.load_terrain()
    assert terrain.height_at(data, 33.41, 35.86) > 2000       # Mount Hermon
    assert 600 < terrain.height_at(data, 31.78, 35.22) < 1000  # Jerusalem
    assert terrain.height_at(data, 31.50, 35.45) < -300        # the Dead Sea
    assert terrain.height_at(data, 33.00, 34.00) < 0           # the Mediterranean off Haifa
    assert np.isnan(terrain.height_at(data, 10.0, 10.0))       # outside the bundle


def test_the_sea_floor_is_flattened_but_the_land_below_sea_level_is_not():
    """The clamp in tools/build_terrain.py: nothing below -450 m survives, so the Dead
    Sea shore (-430 m) does, and the Mediterranean abyss (-3000 m) does not."""
    z = terrain.load_terrain()['z']
    assert int(z.min()) == -450
    assert (z < -400).any()


def test_a_missing_or_corrupt_bundle_costs_the_relief_and_nothing_else(tmp_path):
    assert terrain.load_terrain(tmp_path / 'nope.npz') is None
    junk = tmp_path / 'junk.npz'
    junk.write_bytes(b'not a numpy archive')
    assert terrain.load_terrain(junk) is None
    # the right shape but descending coordinates: refused rather than drawn upside down
    bad = tmp_path / 'descending.npz'
    np.savez_compressed(bad, z=np.zeros((3, 4), np.int16), lat=np.array([3., 2., 1.]),
                        lon=np.arange(4.0), meta=json.dumps({}))
    assert terrain.load_terrain(bad) is None


def test_the_cache_hands_back_the_same_object():
    assert terrain.load_terrain() is terrain.load_terrain()
    assert terrain.available()


# ---- the shade ---------------------------------------------------------------------------
def _grid(n=41):
    return np.linspace(30.0, 31.0, n), np.linspace(34.0, 35.0, n)


def test_flat_ground_is_exactly_one_so_multiply_leaves_the_field_alone():
    lat, lon = _grid()
    shade = terrain.hillshade(np.full((41, 41), 500.0), lat, lon)
    assert shade.dtype == np.float32
    assert np.all(shade == 1.0)


def test_a_slope_facing_the_light_stays_white_and_one_facing_away_darkens():
    """The sun is in the north-west. A surface that falls towards the north-west faces
    it; the same surface the other way up faces south-east."""
    lat, lon = _grid()
    y = np.arange(41)[:, None] * 50.0
    x = np.arange(41)[None, :] * 50.0
    faces_nw = -y + x          # falls to the north (rows up) and to the west
    faces_se = y - x
    lit = terrain.hillshade(faces_nw, lat, lon)[20, 20]
    dark = terrain.hillshade(faces_se, lat, lon)[20, 20]
    assert lit == pytest.approx(1.0)
    assert 0.0 <= dark < 0.9


def test_a_steeper_slope_away_from_the_light_is_darker_and_never_below_zero():
    lat, lon = _grid()
    y = np.arange(41)[:, None]
    gentle = terrain.hillshade((y - np.arange(41)[None, :]) * 20.0, lat, lon)[20, 20]
    steep = terrain.hillshade((y - np.arange(41)[None, :]) * 200.0, lat, lon)[20, 20]
    assert steep < gentle
    assert steep >= 0.0


def test_the_gradient_is_taken_in_metres_with_longitude_scaled_by_cos_lat():
    """One degree of longitude is shorter than one of latitude, so a ramp of equal
    degrees per cell is STEEPER east-west -- and shades more."""
    lat, lon = _grid()
    y = np.arange(41)[:, None] * 50.0
    x = np.arange(41)[None, :] * 50.0
    north_facing = terrain.hillshade(-y + 0 * x, lat, lon)[20, 20]    # falls northward
    east_facing = terrain.hillshade(0 * y - x, lat, lon)[20, 20]      # falls eastward
    # both face partly away from a NW sun by the same ground-metres-per-degree logic,
    # but the east-west cell is shorter so the eastward ramp is steeper and darker
    assert east_facing < north_facing


def test_the_shade_on_the_real_grid_is_mostly_flat_and_has_real_shadows():
    data = terrain.load_terrain()
    shade = terrain.hillshade(data['z'], data['lat'], data['lon'])
    assert shade.shape == data['z'].shape
    assert shade.max() == 1.0 and shade.min() >= 0.0
    assert 0.3 < float((shade < 0.999).mean()) < 0.6       # hills, not a flat plate


# ---- the image --------------------------------------------------------------------------
def test_relief_rgba_is_grey_on_a_shadow_white_on_flat_and_transparent_off_land():
    shade = np.array([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    mask = np.array([[True, True], [False, True]])
    rgba = terrain.relief_rgba(shade, mask)
    assert rgba.dtype == np.uint8 and rgba.shape == (2, 2, 4)
    assert tuple(rgba[0, 0]) == (255, 255, 255, 255)        # flat: no change
    darkest = int(round(255 * terrain.AMBIENT))
    assert tuple(rgba[0, 1][:3]) == (darkest, darkest, darkest)   # fully away: ambient
    assert rgba[1, 0, 3] == 0 and rgba[1, 1, 3] == 255      # sea clear, land opaque
    assert np.all(rgba[..., 0] == rgba[..., 1]) and np.all(rgba[..., 1] == rgba[..., 2])


def test_without_a_mask_everything_is_opaque_and_the_elevation_fallback_is_above_sea():
    rgba = terrain.relief_rgba(np.ones((2, 3), np.float32))
    assert np.all(rgba[..., 3] == 255)
    z = np.array([[5, 0, -5], [-430, 100, 1]])
    mask = terrain.land_from_elevation(z)
    assert mask.tolist() == [[True, False, False], [False, True, True]]


def test_extent_is_the_cell_edges():
    data = {'lat': np.array([10.0, 10.5, 11.0]), 'lon': np.array([20.0, 21.0])}
    assert terrain.extent(data) == pytest.approx((19.5, 21.5, 9.75, 11.25))
