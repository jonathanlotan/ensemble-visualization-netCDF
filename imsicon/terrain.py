"""Terrain under the map: the bundled elevation grid and the shaded relief built from it.

R9. The map already paints the land pale grey and the sea white (R6), so that a field
drawn transparent where it is zero still shows which side of the coast you are on. This is
the next thing a forecaster wants to see through the field: *where the hills are*. A CAPE
plume over the Judean hills and one over the coastal plain are two different forecasts,
and a flat grey land cannot tell them apart.

    mapdata/levant_etopo1.npz ──► load_terrain() ──► hillshade() ──► relief_rgba() ──► MapView
          (ETOPO1, 1 arc-min)        lat, lon, z        0..1 shade      grey + alpha      (Multiply)

**Shadow only, by design.** The relief is composited over the field with the painter's
*Multiply* mode, which can darken a colour and never brighten it. So the shade is
normalised to the flat-ground value: flat terrain and slopes facing the light come out
white (the field underneath is untouched), and slopes facing away darken it. That is the
classic way to lay relief under a thematic map without washing its colours out -- a
hypsometric tint or an opaque hillshade would have to fight the colour scale for the
same pixels, and the colour scale is the reading.

**No Qt here.** The land mask that keeps the sea white is rasterised in `ui/mapview.py`
from the land polygons the map already fills, and handed in; this module only does the
arithmetic, so it can be tested on a ramp and measured without a window.
"""
import json
from pathlib import Path

import numpy as np

TERRAIN = Path(__file__).resolve().parent / 'mapdata' / 'levant_etopo1.npz'

# Metres per degree of latitude (2 pi R / 360 for R = 6371 km); longitude scales by cos(lat).
METRES_PER_DEGREE = 111_195.0

# The light: from the north-west, 45 degrees up -- the cartographic convention, and the
# one direction a reader's eye already assumes, so hills read as hills and not as valleys.
AZIMUTH_DEG = 315.0
ALTITUDE_DEG = 45.0
# The Levant's relief is modest against a 1.85 km cell: a 100 m rise over one cell is a
# 3 degree slope, which shades by about 4 % at this sun -- invisible. The exaggeration is
# what makes the hills legible, and it is the usual z-factor trick, not a measurement:
# the shading says WHERE the slopes are, never how steep.
EXAGGERATION = 6.0
# How dark the steepest shaded slope may get, as a multiplier on the field under it. 0.55
# keeps a turbo red recognisably red in the deepest shadow; lower and the shadow starts to
# read as a different value of the field.
AMBIENT = 0.55


def load_terrain(path=None, cache={}):
    """The bundled elevation grid, or None when it is missing or unreadable.

    -> {'lat': (ny,), 'lon': (nx,), 'z': (ny, nx) int16 metres, 'source': str, 'meta': dict}

    Never raises: a lost bundle costs the map its relief and nothing else, the same
    failure mode `geo.load_mapdata` has for the outlines.
    """
    path = Path(path or TERRAIN)
    key = str(path)
    if key in cache:
        return cache[key]
    try:
        with np.load(path) as blob:
            z = np.asarray(blob['z'], dtype=np.int16)
            lat = np.asarray(blob['lat'], dtype=np.float64)
            lon = np.asarray(blob['lon'], dtype=np.float64)
            meta = json.loads(str(blob['meta'])) if 'meta' in blob else {}
        if z.ndim != 2 or z.shape != (lat.size, lon.size) or lat.size < 2 or lon.size < 2:
            raise ValueError('terrain grid does not match its coordinates')
        if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
            raise ValueError('terrain coordinates must ascend')
        result = {'lat': lat, 'lon': lon, 'z': z, 'meta': meta,
                  'source': str(meta.get('source', ''))}
    except Exception:
        result = None
    cache[key] = result
    return result


def available():
    return load_terrain() is not None


def hillshade(z, lat, lon, azimuth_deg=AZIMUTH_DEG, altitude_deg=ALTITUDE_DEG,
              exaggeration=EXAGGERATION):
    """-> float32 (ny, nx) in 0..1: how lit each cell is, RELATIVE TO FLAT GROUND.

    1.0 is flat ground and every slope facing the light; below that is a slope facing
    away, down to 0 for one turned fully from it. Normalising to flat ground (rather than
    to the usual `n . L`) is what lets the result multiply the map without darkening the
    whole domain by cos(zenith).

    The gradient is taken in metres, with a degree of longitude scaled by cos(lat) row by
    row -- at 32 N a cell is 1.85 km north-south and 1.57 km east-west, and a shade that
    ignored the difference would light every ridge at the wrong angle.
    """
    z = np.asarray(z, dtype=np.float32) * np.float32(exaggeration)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dy = np.gradient(lat) * METRES_PER_DEGREE                      # per row
    dx = (np.gradient(lon)[None, :] * METRES_PER_DEGREE
          * np.cos(np.deg2rad(lat))[:, None])                       # per cell
    dzdy = np.gradient(z, axis=0) / dy[:, None].astype(np.float32)
    dzdx = np.gradient(z, axis=1) / dx.astype(np.float32)
    # Unit normal (-dz/dx, -dz/dy, 1) and the light vector in the same east/north/up frame,
    # with azimuth measured clockwise from north.
    az, alt = np.deg2rad(azimuth_deg), np.deg2rad(altitude_deg)
    light = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
    dot = -dzdx * np.float32(light[0]) - dzdy * np.float32(light[1]) + np.float32(light[2])
    norm = np.sqrt(dzdx * dzdx + dzdy * dzdy + np.float32(1.0))
    shade = dot / (norm * np.float32(np.sin(alt)))        # flat ground -> exactly 1
    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def land_from_elevation(z):
    """A fallback mask when no land polygons are available: above sea level is land.

    Loses the land below sea level (the Dead Sea shore, the Jordan valley), which is why
    the map prefers the polygon mask it already has.
    """
    return np.asarray(z) > 0


def relief_rgba(shade, mask=None, ambient=AMBIENT):
    """-> uint8 (ny, nx, 4), grey where the relief darkens and transparent elsewhere.

    The grey is `ambient + (1 - ambient) * shade`, so the deepest shadow multiplies the
    field by `ambient` and flat ground by 1. `mask` (True on land) sets the alpha: the sea
    is left fully transparent so it stays the widget's own white, whatever the sea floor
    under it does.
    """
    shade = np.asarray(shade, dtype=np.float32)
    grey = np.rint(255.0 * (ambient + (1.0 - ambient) * np.clip(shade, 0.0, 1.0)))
    out = np.empty(shade.shape + (4,), dtype=np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = grey.astype(np.uint8)
    if mask is None:
        out[..., 3] = 255
    else:
        out[..., 3] = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    return out


def extent(data):
    """(lon_min, lon_max, lat_min, lat_max) of the grid's cell EDGES, for an image transform."""
    lat, lon = data['lat'], data['lon']
    dlat, dlon = float(np.diff(lat).mean()), float(np.diff(lon).mean())
    return (float(lon[0]) - dlon / 2, float(lon[-1]) + dlon / 2,
            float(lat[0]) - dlat / 2, float(lat[-1]) + dlat / 2)


def height_at(data, lat, lon):
    """Elevation in metres at the nearest grid cell, or NaN outside the bundle."""
    if data is None:
        return float('nan')
    lats, lons = data['lat'], data['lon']
    if not (lats[0] <= lat <= lats[-1] and lons[0] <= lon <= lons[-1]):
        return float('nan')
    iy = int(np.abs(lats - lat).argmin())
    ix = int(np.abs(lons - lon).argmin())
    return float(data['z'][iy, ix])
