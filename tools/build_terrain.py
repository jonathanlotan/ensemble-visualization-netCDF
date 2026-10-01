"""Build the bundled elevation grid the map shades its terrain from (R9).

Dev-only, run once when the bundle needs rebuilding. The output is committed, so the app
never downloads anything and works offline -- the same arrangement as the coastline
bundle `tools/build_mapdata.py` produces, and for the same reason.

    python tools/build_terrain.py --fetch            # download the subset, write the .npz
    python tools/build_terrain.py etopo_subset.nc    # from a file fetched by hand

Source: **ETOPO1**, NOAA NCEI's 1 arc-minute global relief model (Amante & Eakins 2009),
ice-surface version, grid-registered. Public domain -- NOAA's licence says the data "may
be used and redistributed for free". It is fetched as a NetCDF-3 subset from NOAA's
ERDDAP server, which this app's own `nc3` reader parses, so the tool needs nothing that
the app does not already have.

Why ETOPO1 and not the model's own topography: `topo_icon_web.nc` covers the inner box
only -- lat 29..34, lon 34..36 (CLAUDE.md G6) -- while the map pans across 24.5..39.5 N
and 29.5..39.5 E. A relief that stopped at the inner box would fade out mid-domain, which
is exactly the failure R6.8 declined to ship. One arc-minute (~1.85 km) against the
model's 2.5 km grid is finer than the field it sits under, which is what a backdrop needs.

The clip box is `build_mapdata.CLIP`, deliberately: the relief is masked to the bundled
land polygons when it is drawn, so elevation outside the coastline bundle's box could
never be shown and would only make the file bigger.
"""
import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from imsicon import nc3                                      # noqa: E402
from tools.build_mapdata import CLIP, _ssl_context           # noqa: E402

OUT = ROOT / 'imsicon' / 'mapdata' / 'levant_etopo1.npz'
SEA_FLOOR = -450                # m; below the lowest land there is on Earth
ERDDAP = ('https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?'
          'altitude%5B({lat_min}):1:({lat_max})%5D%5B({lon_min}):1:({lon_max})%5D')
SOURCE = ('ETOPO1 1 arc-minute global relief (NOAA NCEI, Amante & Eakins 2009), ice '
          'surface, grid-registered; public domain. Subset fetched from NOAA ERDDAP '
          '(dataset etopo180) and clipped to the ICON-IL pan range by '
          'tools/build_terrain.py')


def fetch(target):
    url = ERDDAP.format(**CLIP)
    print(f'  fetching {url} ...', flush=True)
    with urllib.request.urlopen(url, timeout=180, context=_ssl_context()) as response, \
            open(target, 'wb') as out:
        out.write(response.read())
    return target


def read_subset(path):
    """-> (lat ascending, lon ascending, int16 elevation in m) from the ERDDAP NetCDF."""
    hdr = nc3.parse(path)
    names = {name.lower(): name for name in hdr['vars']}
    lat_name = names.get('latitude') or names.get('lat')
    lon_name = names.get('longitude') or names.get('lon')
    z_name = names.get('altitude') or names.get('z') or names.get('elevation')
    if not (lat_name and lon_name and z_name):
        raise SystemExit(f'{path}: expected latitude/longitude/altitude, found '
                         f'{list(hdr["vars"])}')
    lat = np.asarray(nc3.view(path, hdr, lat_name), dtype=np.float64)
    lon = np.asarray(nc3.view(path, hdr, lon_name), dtype=np.float64)
    z = np.asarray(nc3.view(path, hdr, z_name))
    fill = hdr['vars'][z_name]['attrs'].get('_FillValue')
    z = z.astype(np.float64)
    if fill is not None:
        z[z == float(np.asarray(fill).ravel()[0])] = np.nan
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    if lon[0] > lon[-1]:
        lon, z = lon[::-1], z[:, ::-1]
    if np.isnan(z).any():
        raise SystemExit(f'{int(np.isnan(z).sum())} missing cells in the subset')
    # The sea floor is never drawn -- the relief is masked to the land polygons -- and
    # bathymetry is the part of the grid that compresses worst, so everything below the
    # deepest land on Earth (the Dead Sea shore, -430 m) is flattened. Measured: 751 kB
    # without the clamp, 591 kB with it, and not one land cell is touched.
    z = np.maximum(z, SEA_FLOOR)
    return lat, lon, np.rint(z).astype(np.int16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('subset', nargs='?', help='an ETOPO1 NetCDF subset already fetched')
    ap.add_argument('--fetch', action='store_true', help='download the subset from ERDDAP')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args(argv)

    if args.fetch:
        subset = fetch(Path(tempfile.mkdtemp()) / 'etopo_subset.nc')
    elif args.subset:
        subset = Path(args.subset)
    else:
        ap.error('give the fetched NetCDF subset, or --fetch')

    lat, lon, z = read_subset(subset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, z=z, lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        meta=json.dumps({'schema': 1, 'source': SOURCE, 'units': 'm',
                                         'clip': CLIP}))
    print(f'\nwrote {out} ({out.stat().st_size / 1024:.0f} kB)')
    print(f'  grid     : {z.shape[0]} x {z.shape[1]} at {np.diff(lat).mean() * 60:.2f} '
          f'arc-min, lat {lat[0]:g}..{lat[-1]:g}, lon {lon[0]:g}..{lon[-1]:g}')
    print(f'  elevation: {int(z.min())}..{int(z.max())} m; {np.count_nonzero(z > 0)} cells '
          f'above sea level of {z.size}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
