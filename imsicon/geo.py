"""Coastline and terrain overlay from the IMS topography file.

`topo_icon_web.nc` (MANUALS/IMS_ICON) is a *different, smaller* grid than the ensemble
files -- lat 29.0-34.0, lon 34.0-36.0 -- but the same 0.025 deg spacing, aligned on exact
integer offsets (CLAUDE.md G6). Its `fr_land` field gives an offline coastline for the
inner box; outside it the map simply has no coastline drawn.
"""
import numpy as np
from pathlib import Path

from . import nc3

TOPO_NAMES = ('topo_icon_web.nc', 'icon_topo_web.nc')


def find_topo(near):
    """Look for the topography file beside the data file, then in ./data."""
    near = Path(near)
    roots = [near.parent, near.parent / 'data', Path.cwd(), Path.cwd() / 'data',
             Path(__file__).resolve().parent.parent / 'data']
    for root in roots:
        for name in TOPO_NAMES:
            candidate = root / name
            if candidate.exists():
                return candidate
    return None


def load_topo(path):
    """-> dict(lat, lon, fr_land, topography) or None if the file is unusable."""
    try:
        hdr = nc3.parse(path)
        out = {'lat': np.asarray(nc3.view(path, hdr, 'lat'), dtype=float),
               'lon': np.asarray(nc3.view(path, hdr, 'lon'), dtype=float)}
        for key, var in (('fr_land', 'fr_land'), ('topography', 'topography_c')):
            if var in hdr['vars']:
                out[key] = np.asarray(nc3.view(path, hdr, var), dtype=np.float32)
        return out if 'fr_land' in out else None
    except Exception:
        return None


def contour_segments(x, y, z, level=0.5):
    """Marching squares -> (xs, ys) polyline arrays separated by NaN.

    Dependency-free (no matplotlib/scipy) so the frozen exe stays small. Saddle cells are
    resolved by pairing crossings in order, which is fine for drawing a coastline.
    """
    z = np.asarray(z, dtype=float)
    ny, nx = z.shape
    xs, ys = [], []

    def cross(v0, v1, c0, c1):
        """Interpolated crossing of `level` on the edge between two corners."""
        if (v0 - level) * (v1 - level) >= 0 or v1 == v0:
            return None
        f = (level - v0) / (v1 - v0)
        return (c0[0] + f * (c1[0] - c0[0]), c0[1] + f * (c1[1] - c0[1]))

    for i in range(ny - 1):
        for j in range(nx - 1):
            a, b = z[i, j], z[i, j + 1]
            d, c = z[i + 1, j], z[i + 1, j + 1]
            if not np.isfinite(a + b + c + d):
                continue
            lo = min(a, b, c, d)
            hi = max(a, b, c, d)
            if lo >= level or hi < level:          # no crossing in this cell
                continue
            p00, p10 = (x[j], y[i]), (x[j + 1], y[i])
            p01, p11 = (x[j], y[i + 1]), (x[j + 1], y[i + 1])
            pts = [p for p in (cross(a, b, p00, p10), cross(b, c, p10, p11),
                               cross(d, c, p01, p11), cross(a, d, p00, p01))
                   if p is not None]
            for k in range(0, len(pts) - 1, 2):
                (x0, y0), (x1, y1) = pts[k], pts[k + 1]
                xs.extend((x0, x1, np.nan))
                ys.extend((y0, y1, np.nan))
    return np.array(xs), np.array(ys)


def coastline_for(data_path, cache={}):
    """Cached coastline polylines for the file's domain, or (None, None)."""
    topo_path = find_topo(data_path)
    if topo_path is None:
        return None, None
    key = str(topo_path)
    if key not in cache:
        topo = load_topo(topo_path)
        cache[key] = ((None, None) if topo is None else
                      contour_segments(topo['lon'], topo['lat'], topo['fr_land'], 0.5))
    return cache[key]
