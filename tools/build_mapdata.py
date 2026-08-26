"""Build the bundled coastline + border overlay from Natural Earth (CLAUDE.md Phase 4.3).

Dev-only, run once when the overlay needs rebuilding. The output is committed, so the app
never downloads anything and works offline -- which is the whole point: `topo_icon_web.nc`
only covers the inner box (G6), so the rest of the domain had no coastline at all.

    python tools/build_mapdata.py --fetch                 # download, clip, write
    python tools/build_mapdata.py ne_10m_coastline.geojson \
        ne_10m_admin_0_boundary_lines_land.geojson ne_10m_land.geojson

Source: Natural Earth 1:10 m physical coastline, land polygons and cultural admin-0
boundary lines, public domain (naturalearthdata.com terms: "no permission needed").
Fetched from the nvkelso/natural-earth-vector mirror, which is the same data in GeoJSON
form.

**The land polygons** (schema 2) are what lets the viewer paint the land pale grey and
leave the sea white, so that a map whose field is transparent where it is zero still
shows you where you are. Lines cannot do that: a coastline clipped to a box is a set of
open polylines with no inside, so the fill needs the real rings. They are clipped with
Sutherland-Hodgman, which keeps a ring a ring -- the degenerate edges it can leave along
the box enclose no area, so an odd-even fill is unaffected by them.

**The border classification is carried through unmodified.** Natural Earth marks several
lines in this domain as `Disputed`, `Indefinite` or `Line of control`, and this build keeps
that field so the viewer can draw them differently instead of presenting every line as a
settled international boundary. Nothing here decides where a border is; it renders what the
source says, including the source's own uncertainty.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'imsicon' / 'mapdata' / 'levant_10m.json'

BASE = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/')
COASTLINE_URL = BASE + 'ne_10m_coastline.geojson'
BORDERS_URL = BASE + 'ne_10m_admin_0_boundary_lines_land.geojson'
LAND_URL = BASE + 'ne_10m_land.geojson'

# The model domain is 28-34.5 N / 33-37 E, and MapView allows panning to domain +- 50 % of
# the span (G10), so the overlay has to cover everything the user can reach -- otherwise
# the coastline stops mid-pan and looks like a bug.
CLIP = dict(lon_min=30.5, lon_max=39.5, lat_min=24.5, lat_max=38.0)
DECIMALS = 4                    # ~11 m; the model grid is 2.5 km, so this is far finer

# Natural Earth's own classes, mapped to how the viewer draws them. Anything the source
# flags as less than settled is drawn dashed rather than solid.
SOLID_CLASSES = ('International boundary (verify)',)


def fetch(url):
    print(f'  fetching {url.rsplit("/", 1)[-1]} ...', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode('utf8'))


def _inside(point):
    lon, lat = point[0], point[1]
    return (CLIP['lon_min'] <= lon <= CLIP['lon_max']
            and CLIP['lat_min'] <= lat <= CLIP['lat_max'])


def clip_line(points):
    """-> [polyline] inside the clip box, keeping one point past each crossing.

    Keeping the neighbour means a line that leaves the box is still drawn all the way to
    the edge rather than stopping a pixel short of it; the ViewBox does the visual clip.
    """
    runs, current = [], []
    for index, point in enumerate(points):
        neighbour_in = ((index > 0 and _inside(points[index - 1]))
                        or (index + 1 < len(points) and _inside(points[index + 1])))
        if _inside(point) or neighbour_in:
            current.append([round(float(point[0]), DECIMALS),
                            round(float(point[1]), DECIMALS)])
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [run for run in runs if len(run) >= 2]


def clip_ring(ring, box=None):
    """Sutherland-Hodgman: one closed ring clipped to the box -> one closed ring.

    Unlike `clip_line`, which may split a polyline into several runs, this has to come
    back as a single ring or the fill is meaningless. Where the source ring leaves and
    re-enters the box the result runs along the box edge instead, which is what makes the
    whole domain fill correctly rather than stopping at the last vertex inside it.
    """
    box = box or (CLIP['lon_min'], CLIP['lon_max'], CLIP['lat_min'], CLIP['lat_max'])
    lon_min, lon_max, lat_min, lat_max = box

    def inside(point, edge):
        return (point[0] >= lon_min, point[0] <= lon_max,
                point[1] >= lat_min, point[1] <= lat_max)[edge]

    def cut(a, b, edge):
        if edge < 2:
            x = lon_min if edge == 0 else lon_max
            t = (x - a[0]) / (b[0] - a[0])
            return (x, a[1] + t * (b[1] - a[1]))
        y = lat_min if edge == 2 else lat_max
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    out = [(float(p[0]), float(p[1])) for p in ring]
    for edge in range(4):
        if not out:
            return []
        buf, previous = [], out[-1]
        for current in out:
            here, there = inside(current, edge), inside(previous, edge)
            if here:
                if not there:
                    buf.append(cut(previous, current, edge))
                buf.append(current)
            elif there:
                buf.append(cut(previous, current, edge))
            previous = current
        out = buf
    return out


def geometry_rings(geometry):
    """Every ring of a (Multi)Polygon, exterior and holes alike.

    Holes are kept rather than dropped: the viewer fills with the odd-even rule, so a hole
    ring inside an exterior one punches its own hole with no further bookkeeping.
    """
    kind, coords = geometry.get('type'), geometry.get('coordinates')
    polygons = [coords] if kind == 'Polygon' else (coords if kind == 'MultiPolygon' else [])
    return [ring for polygon in polygons for ring in polygon]


def collect_land(features):
    """-> [[[lon, lat], ...], ...] closed rings inside the clip box."""
    rings = []
    for feature in features:
        for ring in geometry_rings(feature.get('geometry') or {}):
            clipped = clip_ring(ring)
            if len(clipped) >= 3:
                rings.append([[round(x, DECIMALS), round(y, DECIMALS)]
                              for x, y in clipped])
    return rings


def geometry_lines(geometry):
    kind, coords = geometry['type'], geometry['coordinates']
    if kind == 'LineString':
        return [coords]
    if kind == 'MultiLineString':
        return list(coords)
    return []


def collect(features, classify=None):
    """-> [{'class': str, 'points': [[lon, lat], ...]}] clipped to the domain."""
    out = []
    for feature in features:
        properties = feature.get('properties') or {}
        kind = classify(properties) if classify else 'coastline'
        for line in geometry_lines(feature.get('geometry') or {}):
            out.extend({'class': kind, 'points': run} for run in clip_line(line))
    return out


def border_class(properties):
    """Natural Earth's FEATURECLA, kept verbatim as the class name."""
    return str(properties.get('FEATURECLA') or 'Boundary').strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('coastline', nargs='?', help='ne_10m_coastline.geojson')
    ap.add_argument('borders', nargs='?', help='ne_10m_admin_0_boundary_lines_land.geojson')
    ap.add_argument('land', nargs='?', help='ne_10m_land.geojson')
    ap.add_argument('--fetch', action='store_true', help='download the three source files')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)

    if args.fetch:
        coastline, borders = fetch(COASTLINE_URL), fetch(BORDERS_URL)
        land = fetch(LAND_URL)
    elif args.coastline and args.borders and args.land:
        coastline = json.loads(Path(args.coastline).read_text())
        borders = json.loads(Path(args.borders).read_text())
        land = json.loads(Path(args.land).read_text())
    else:
        ap.error('give the three GeoJSON files, or --fetch')

    coast_lines = collect(coastline['features'])
    border_lines = collect(borders['features'], classify=border_class)
    land_rings = collect_land(land['features'])
    if not coast_lines:
        print('no coastline survived the clip -- check CLIP', file=sys.stderr)
        return 2
    if not land_rings:
        print('no land survived the clip -- check CLIP', file=sys.stderr)
        return 2

    blob = {
        'schema': 2,
        'source': 'Natural Earth 1:10m (public domain) -- physical coastline, land '
                  'polygons and cultural admin-0 boundary lines, clipped to the ICON-IL '
                  'domain by tools/build_mapdata.py',
        'clip': CLIP,
        'solid_classes': list(SOLID_CLASSES),
        'coastline': [line['points'] for line in coast_lines],
        'borders': border_lines,
        'land': land_rings,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, separators=(',', ':')))

    classes = {}
    for line in border_lines:
        classes[line['class']] = classes.get(line['class'], 0) + 1
    print(f'\nwrote {out} ({out.stat().st_size / 1024:.0f} kB)')
    print(f'  coastline: {len(coast_lines)} polylines, '
          f'{sum(len(l["points"]) for l in coast_lines)} points')
    print(f'  borders  : {len(border_lines)} polylines, '
          f'{sum(len(l["points"]) for l in border_lines)} points')
    print(f'  land     : {len(land_rings)} rings, '
          f'{sum(len(r) for r in land_rings)} points')
    for name, count in sorted(classes.items()):
        style = 'solid' if name in SOLID_CLASSES else 'dashed'
        print(f'      {count:3d} x {name}  -> {style}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
