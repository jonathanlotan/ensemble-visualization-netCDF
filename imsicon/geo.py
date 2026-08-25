"""Map overlay: coastlines and national borders, drawn over the model field.

Two sources, and the bundled one is why this file changed shape:

* `imsicon/mapdata/levant_10m.json` -- Natural Earth 1:10 m coastline and admin-0 boundary
  lines, clipped to the domain by `tools/build_mapdata.py` and committed (85 kB). This is
  what CLAUDE.md Phase 4.3 asked for, and it covers the WHOLE domain.
* `topo_icon_web.nc` (MANUALS/IMS_ICON), whose `fr_land` gives a coastline derived from the
  model's own land mask. It is a *different, smaller* grid -- lat 29.0-34.0, lon 34.0-36.0,
  same 0.025 deg spacing, aligned on exact integer offsets (CLAUDE.md **G6**) -- so it
  covers only the inner box and is kept as the fallback for when the bundle is missing.

**On the borders.** Natural Earth's own `FEATURECLA` is carried through unmodified, and
several lines in this domain are marked by the source as `Disputed`, `Indefinite` or
`Line of control`. Those are drawn dashed and the settled ones solid, so the overlay shows
the source's uncertainty instead of flattening every line into a settled boundary. Nothing
in this app decides where a border is.
"""
import json
import numpy as np
from pathlib import Path

from . import nc3

TOPO_NAMES = ('topo_icon_web.nc', 'icon_topo_web.nc')
MAPDATA = Path(__file__).resolve().parent / 'mapdata' / 'levant_10m.json'


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
    """Cached coastline from the model's own land mask, or (None, None) -- the fallback."""
    topo_path = find_topo(data_path)
    if topo_path is None:
        return None, None
    key = str(topo_path)
    if key not in cache:
        topo = load_topo(topo_path)
        cache[key] = ((None, None) if topo is None else
                      contour_segments(topo['lon'], topo['lat'], topo['fr_land'], 0.5))
    return cache[key]


# ---- the bundled Natural Earth overlay --------------------------------------------------
def _polylines_to_arrays(polylines):
    """[[[lon, lat], ...], ...] -> (xs, ys) with NaN between lines.

    One `PlotDataItem` with `connect='finite'` draws the whole layer, which is what keeps
    3,000 coastline points from becoming 200 scene items.
    """
    xs, ys = [], []
    for line in polylines:
        if len(line) < 2:
            continue
        for lon, lat in line:
            xs.append(float(lon))
            ys.append(float(lat))
        xs.append(np.nan)
        ys.append(np.nan)
    return np.array(xs), np.array(ys)


def load_mapdata(path=None, cache={}):
    """The bundled overlay as ready-to-draw arrays, or None if it is not there.

    -> {'coastline': (xs, ys), 'borders': (xs, ys), 'borders_uncertain': (xs, ys),
        'source': str, 'classes': {name: n}}

    Never raises: a missing or corrupt overlay costs the map its outlines, and that must
    not stop a forecast being opened.
    """
    path = Path(path or MAPDATA)
    key = str(path)
    if key in cache:
        return cache[key]
    try:
        blob = json.loads(path.read_text())
        solid = set(blob.get('solid_classes') or ())
        borders, uncertain, classes = [], [], {}
        for line in blob.get('borders') or ():
            name = line.get('class', '')
            classes[name] = classes.get(name, 0) + 1
            (borders if name in solid else uncertain).append(line['points'])
        result = {
            'coastline': _polylines_to_arrays(blob.get('coastline') or ()),
            'borders': _polylines_to_arrays(borders),
            'borders_uncertain': _polylines_to_arrays(uncertain),
            'source': str(blob.get('source', '')),
            'classes': classes,
        }
    except Exception:
        result = None
    cache[key] = result
    return result


def overlay_for(data_path):
    """Every layer the map should draw over the field, in one dict.

    Prefers the bundled Natural Earth coastline because it spans the whole domain; falls
    back to the model's own `fr_land` contour, which stops at the inner box (**G6**), only
    when the bundle is unavailable.
    """
    layers = {'coastline': (None, None), 'borders': (None, None),
              'borders_uncertain': (None, None), 'source': '', 'classes': {}}
    bundled = load_mapdata()
    if bundled is not None:
        layers.update(bundled)
    if layers['coastline'][0] is None or not len(layers['coastline'][0]):
        layers['coastline'] = coastline_for(data_path)
        layers['source'] = 'coastline from the model land mask (inner box only -- G6)'
    return layers
