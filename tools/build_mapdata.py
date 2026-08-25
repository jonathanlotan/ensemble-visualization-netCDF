"""Build the bundled coastline + border overlay from Natural Earth (CLAUDE.md Phase 4.3).

Dev-only, run once when the overlay needs rebuilding. The output is committed, so the app
never downloads anything and works offline -- which is the whole point: `topo_icon_web.nc`
only covers the inner box (G6), so the rest of the domain had no coastline at all.

    python tools/build_mapdata.py --fetch                 # download, clip, write
    python tools/build_mapdata.py ne_10m_coastline.geojson ne_10m_admin_0_boundary_lines_land.geojson

Source: Natural Earth 1:10 m physical coastline and cultural admin-0 boundary lines,
public domain (naturalearthdata.com terms: "no permission needed"). Fetched from the
nvkelso/natural-earth-vector mirror, which is the same data in GeoJSON form.

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
    ap.add_argument('--fetch', action='store_true', help='download the two source files')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)

    if args.fetch:
        coastline, borders = fetch(COASTLINE_URL), fetch(BORDERS_URL)
    elif args.coastline and args.borders:
        coastline = json.loads(Path(args.coastline).read_text())
        borders = json.loads(Path(args.borders).read_text())
    else:
        ap.error('give the two GeoJSON files, or --fetch')

    coast_lines = collect(coastline['features'])
    border_lines = collect(borders['features'], classify=border_class)
    if not coast_lines:
        print('no coastline survived the clip -- check CLIP', file=sys.stderr)
        return 2

    blob = {
        'schema': 1,
        'source': 'Natural Earth 1:10m (public domain) -- physical coastline and '
                  'cultural admin-0 boundary lines, clipped to the ICON-IL domain by '
                  'tools/build_mapdata.py',
        'clip': CLIP,
        'solid_classes': list(SOLID_CLASSES),
        'coastline': [line['points'] for line in coast_lines],
        'borders': border_lines,
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
    for name, count in sorted(classes.items()):
        style = 'solid' if name in SOLID_CLASSES else 'dashed'
        print(f'      {count:3d} x {name}  -> {style}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
