"""Dev-only: why does the deterministic geopotential not download/open?

    venv/bin/python tools/diagnose_geopot.py      # asks for user and password

Prints (1) every anchor in the IMS_ICON listing that mentions 'geo' or 'fi', whether or
not it matches the IE_ name grammar, (2) the fields the app's parser does see for the
newest run, and (3) the header of a 4 MiB prefix of the geopot file, if it is listed.
Credentials come from the environment only and are never printed.
"""
import bz2
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imsicon import download, nc3, products  # noqa: E402

import getpass
user = os.environ.get('IMS_USER') or input('IMS user: ').strip()
pw = os.environ.get('IMS_PASS') or getpass.getpass('IMS password (not echoed): ')
s = download.make_session(user, pw)
url = download.base_url(products.ICON)
r = s.get(url, timeout=60)
print('listing', url, '->', r.status_code, len(r.text), 'chars')
if r.status_code != 200:
    sys.exit('login refused (HTTP %d) - check the user name and password' % r.status_code)

names = set(re.findall(r'>([^<>]+\.nc[^<>]*)<', r.text))
names |= {h.rpartition('/')[2] for h in re.findall(r'href="([^"]+)"', r.text)}
print('\n(1) raw names mentioning geo / fi / z_:')
for n in sorted(names):
    low = n.lower()
    if 'geo' in low or '_fi' in low or '_z.' in low or '_z_' in low:
        ok = bool(products.ICON.name_re.match(n))
        print(f'   {n!r:60} parser accepts: {ok}')

files = download.parse_listing(r.text, products.ICON)
newest = download.runs_in(files)[0]
fields = sorted(f.field for f in files if f.run == newest)
print(f'\n(2) parser sees {len(fields)} fields in run {newest}:')
print('   ', ' '.join(fields))
unparsed = sorted(n for n in names if n.startswith('IE_') and newest in n
                  and not products.ICON.name_re.match(n))
print('    IE_ names of that run the parser REJECTS:', unparsed or 'none')

geo = [f for f in files if f.run == newest and f.field == 'geopot']
if not geo:
    sys.exit('\n(3) no geopot in the parsed listing -- that is the bug')
g = geo[0]
print(f'\n(3) {g.name}  size={g.size}  url={g.url}')
resp = s.get(g.url, headers={'Range': 'bytes=0-4194303'}, stream=True, timeout=120)
print('    range request ->', resp.status_code)
raw = resp.raw.read(4 * 1024 * 1024)
print('    first bytes:', raw[:4])
try:
    data = bz2.BZ2Decompressor().decompress(raw)
except Exception as exc:
    sys.exit(f'    not bz2: {exc!r}')
print('    decompressed prefix:', len(data), 'B, magic', data[:4])
tmp = Path(os.environ.get('TMPDIR', '/tmp')) / 'geopot_prefix.nc'
tmp.write_bytes(data)
try:
    hdr = nc3.parse(str(tmp))
except Exception as exc:
    sys.exit(f'    nc3.parse failed: {exc!r}')
print('    version', hdr['version'], 'numrecs', hdr['numrecs'], 'declared', hdr['declared_numrecs'])
for name, var in hdr['vars'].items():
    print('   ', name, {k: v for k, v in var.items() if k != 'attrs'}, var.get('attrs'))
try:
    from imsicon.dataset import EnsembleFile
    ds = EnsembleFile(str(tmp))
    print('    opens as', ds.field, ds.units, 'axis', ds.axis.kind, ds.n_members,
          'levels', 'steps', len(ds.times))
except Exception as exc:
    import traceback
    traceback.print_exc()
