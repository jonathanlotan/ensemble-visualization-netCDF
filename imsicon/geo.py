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

from . import isolines, nc3

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

    The land mask is contoured once at startup and cached, so this was a plain Python
    loop over the cells until R5 needed the same algorithm on every redraw of a field.
    It now delegates to `isolines.contour_lines`, which is the vectorised version of
    exactly this: one marching squares in the codebase rather than two that can disagree
    about a saddle. Both are dependency-free (no matplotlib, no scipy), which is what
    keeps the frozen exe small.
    """
    return isolines.contour_lines(x, y, z, [level])


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
