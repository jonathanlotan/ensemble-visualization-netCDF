"""V2.1 -- settle the units string of every IMS field without downloading 3.9 GB.

A `.nc.bz2` is one bz2 stream and bz2 decompresses incrementally, so a *prefix* of the
compressed file yields the NetCDF-3 header plus the first few time records (v2.md 1.2,
re-measured 2026-08-24: 4 MiB -> 8,177,912 B / 2 steps; 16 MiB -> 27,221,957 B / 8 steps).

    4 MiB per field  settles the units string and the t=0 value range
    16 MiB per field settles the accumulation kind (G14) and the cloud encoding (G22)

Dev-only: not imported by the app. Credentials come from IMS_USER/IMS_PASS or keyring,
are never hardcoded, never logged, and never placed in a URL (G7: the server is NTLM,
plain Basic returns 401).

    python tools/sniff_headers.py --local data/*.nc.bz2
    python tools/sniff_headers.py --all --deep          # needs credentials + network
"""
import argparse
import bz2
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from imsicon import nc3                                            # noqa: E402

BASE = 'https://data.israel-meteo-service.org/ims/IMS_ICON_ENSEMBLE/'
PREFIX_SHALLOW = 4 << 20
PREFIX_DEEP = 16 << 20

FIELDS = ('CAPE_ML', 'T_2M', 'T_S', 'RELHUM_2M', 'TOT_PREC', 'U_10M', 'V_10M',
          'VMAX_10M', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 'ASWDIFD_S', 'ASWDIR_S', 'H_SNOW')
CLOUD_FIELDS = ('CLCT', 'CLCL', 'CLCM', 'CLCH')
ACCUM_CANDIDATES = ('TOT_PREC', 'ASWDIFD_S', 'ASWDIR_S')


class SniffError(Exception):
    pass


# ---- the prefix mechanism -------------------------------------------------------------
def decompress_prefix(raw):
    """Decompress as much of a truncated bz2 stream as it will give up.

    A stream shorter than the prefix is SUCCESS, not an error: H_SNOW.nc.bz2 is
    125,788 B in total (all-zero summer snow) and decompresses completely (v2.md 1.2).
    """
    dec = bz2.BZ2Decompressor()
    try:
        return dec.decompress(raw)
    except EOFError:                      # stream ended cleanly inside the prefix
        return b''
    except OSError as exc:                # not a bz2 stream at all
        raise SniffError(f'not a bz2 stream: {exc}') from exc


def clamp_to_available(hdr, nbytes=None):
    """Records the file actually holds. `nc3.parse` already clamps (G26); this reports it.

    Kept as the sniffer's own name for the concept, because the prefix path is exactly
    what motivated the guard: a 4 MiB prefix parses as 121 time steps while holding 2.
    """
    return hdr['numrecs']


def sniff_bytes(raw, label='<prefix>'):
    """Parse a decompressed NetCDF-3 prefix. -> report dict."""
    if len(raw) < 4 or raw[:3] != b'CDF':
        raise SniffError(f'{label}: prefix does not start a NetCDF-3 file '
                         f'(got {raw[:4]!r}); try a larger prefix')
    tmp = Path(tempfile.mkdtemp(prefix='imssniff')) / 'prefix.nc'
    tmp.write_bytes(raw)
    try:
        hdr = nc3.parse(tmp)
        steps = clamp_to_available(hdr, len(raw))
        # A bz2 prefix ends at an arbitrary byte, so the file almost always stops
        # mid-record. nc3.view casts the tail of the mapping to '>f8'/'>f4', and numpy
        # refuses a cast that does not divide evenly -- so trim to whole records first.
        # (Without this, whether a field sniffs at all is pure luck.)
        record_vars = [v for v in hdr['vars'].values() if v['record']]
        if record_vars:
            record_start = min(v['begin'] for v in record_vars)
            tmp.write_bytes(raw[:record_start + steps * hdr['recsize']])
        var = nc3.field_name(hdr)
        attrs = hdr['vars'][var]['attrs']
        report = {
            'label': label,
            'variable': var,
            'field': var[:-4] if var.endswith('_eps') else var,
            'units': str(attrs.get('units', '')).strip(),
            'long_name': str(attrs.get('long_name', '')).strip(),
            'recsize': hdr['recsize'],
            'begin': hdr['vars'][var]['begin'],
            'declared_steps': None,
            'steps': steps,
            'bytes': len(raw),
            'min': np.nan, 'max': np.nan, 'monotone': None, 'cloud': None,
        }
        if steps:
            data = np.asarray(nc3.view(tmp, hdr, var), dtype=np.float64)
            report['min'] = float(np.nanmin(data))
            report['max'] = float(np.nanmax(data))
            report['monotone'] = monotonicity(data)
            report['cloud'] = cloud_verdict(data) if report['field'] in CLOUD_FIELDS else None
        return report
    finally:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()


def monotonicity(data):
    """G14: accumulated fields never decrease in time; averaged ones do. None = too short."""
    if data.shape[0] < 3:
        return None
    diffs = np.diff(data, axis=0)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        return None
    if np.all(finite >= -1e-6):
        return 'non-decreasing (accumulated: kind="sum" unless it is a running mean)'
    return 'not monotone (instantaneous, or an averaged field: kind="mean")'


def cloud_verdict(data):
    """The three G22 outcomes. Never guesses: 'undecidable' is a real answer."""
    good = data[np.isfinite(data)]
    if good.size == 0:
        return 'undecidable (no finite values)'
    hi = float(good.max())
    if hi > 1.0 + 1e-6:
        return f'PERCENT (max {hi:.3f} > 1) -- no conversion, label %'
    if hi < 1e-6:
        return 'UNDECIDABLE (all ~0, clear sky) -- refuse to convert, sniff another run'
    return f'FRACTION (max {hi:.3f} <= 1, spread over (0,1)) -- offer x100 -> %'


def sniff_local(path, prefix_bytes=PREFIX_SHALLOW):
    """Sniff a `.nc.bz2` (or plain `.nc`) already on disk -- no network needed."""
    path = Path(path)
    raw = path.read_bytes()[:prefix_bytes]
    if path.suffix.lower() == '.bz2':
        raw = decompress_prefix(raw)
    return sniff_bytes(raw, path.name)


# ---- network ---------------------------------------------------------------------------
def credentials():
    user, password = os.environ.get('IMS_USER'), os.environ.get('IMS_PASS')
    if user and password:
        return user, password
    try:
        import keyring
        user = user or keyring.get_password('ims-icon', 'username')
        if user:
            password = keyring.get_password('ims-icon', user)
        if user and password:
            return user, password
    except Exception:
        pass
    raise SniffError('no IMS credentials: set IMS_USER and IMS_PASS in the environment '
                     '(they are never written to disk or logged by this script)')


def session():
    try:
        import requests
        from requests_ntlm import HttpNtlmAuth
    except ImportError as exc:
        raise SniffError('pip install requests requests-ntlm  (G7: the IMS server speaks '
                         f'NTLM/Negotiate, not HTTP Basic) -- {exc}') from exc
    user, password = credentials()
    sess = requests.Session()
    sess.auth = HttpNtlmAuth(user, password)
    return sess


def fetch_prefix(sess, url, prefix_bytes):
    resp = sess.get(url, headers={'Range': f'bytes=0-{prefix_bytes - 1}'}, timeout=60)
    if resp.status_code in (401, 403):
        raise SniffError('authentication failed')      # never echo what was tried
    resp.raise_for_status()
    return resp.content


def latest_run(sess):
    """Newest ICON_ENS run id on the server. The listing is UNTRUSTED input: strict regex."""
    import re
    resp = sess.get(BASE, timeout=60)
    resp.raise_for_status()
    runs = sorted(set(re.findall(r'ICON_ENS_(\d{10})_[A-Z0-9_]+\.nc\.bz2', resp.text)))
    if not runs:
        raise SniffError('no ICON_ENS_<run>_<FIELD>.nc.bz2 entries in the server listing')
    return runs[-1]


# ---- reporting -------------------------------------------------------------------------
def print_report(rows):
    print()
    print(f'{"field":11s} {"units":9s} {"steps":>5s} {"min":>12s} {"max":>12s}  long_name')
    print('-' * 96)
    for r in rows:
        print(f'{r["field"]:11s} {r["units"]:9s} {r["steps"]:5d} '
              f'{r["min"]:12.4g} {r["max"]:12.4g}  {r["long_name"][:38]}')
    print()
    for r in rows:
        if r.get('monotone'):
            print(f'  {r["field"]:11s} time-behaviour: {r["monotone"]}')
        if r.get('cloud'):
            print(f'  {r["field"]:11s} cloud encoding: {r["cloud"]}')
    print()
    print('Paste-ready registry rows (v2.md 1.4):')
    for r in rows:
        print(f'| `{r["field"]}` | `{r["units"]}` | measured {r["steps"]} steps | '
              f'range {r["min"]:.4g}..{r["max"]:.4g} |')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--local', nargs='*', metavar='FILE',
                    help='sniff local .nc.bz2/.nc files instead of the server')
    ap.add_argument('--all', action='store_true', help='sniff all 15 fields from the server')
    ap.add_argument('--field', action='append', help='sniff one field (repeatable)')
    ap.add_argument('--run', help='run id YYYYMMDDHH (default: newest on the server)')
    ap.add_argument('--deep', action='store_true',
                    help=f'fetch {PREFIX_DEEP >> 20} MiB so accumulation kind and cloud '
                         'encoding can be decided')
    args = ap.parse_args(argv)
    prefix = PREFIX_DEEP if args.deep else PREFIX_SHALLOW

    rows = []
    if args.local is not None:
        for path in args.local:
            try:
                rows.append(sniff_local(path, prefix))
            except (SniffError, Exception) as exc:
                print(f'{Path(path).name}: {type(exc).__name__}: {exc}', file=sys.stderr)
        if not rows:
            return 2
        print_report(rows)
        return 0

    fields = args.field or (list(FIELDS) if args.all else None)
    if not fields:
        ap.error('give --local FILE..., or --all, or --field NAME')
    try:
        sess = session()
        run = args.run or latest_run(sess)
    except SniffError as exc:
        print(f'{exc}', file=sys.stderr)
        return 2
    print(f'run {run}: fetching {prefix >> 20} MiB of each of {len(fields)} fields '
          f'({len(fields) * (prefix >> 20)} MiB total, vs {len(fields) * 262} MiB whole)')
    for field in fields:
        name = f'ICON_ENS_{run}_{field}.nc.bz2'
        try:
            raw = decompress_prefix(fetch_prefix(sess, BASE + name, prefix))
            rows.append(sniff_bytes(raw, name))
            print(f'  {field:11s} ok')
        except Exception as exc:
            print(f'  {field:11s} {type(exc).__name__}: {exc}', file=sys.stderr)
    if not rows:
        return 2
    print_report(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
