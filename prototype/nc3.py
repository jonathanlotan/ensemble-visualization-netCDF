"""Validated prototype: NetCDF-3 classic / 64-bit-offset reader for IMS ICON ensemble files.

Header parse in pure Python + zero-copy strided numpy.memmap over the data.
No netCDF4/HDF5/xarray needed. Verified element-for-element against netCDF4.Dataset
on ICON_ENS_2026082300_CAPE_ML.nc (2026-08-23). See CLAUDE.md sections 0.4 and 0.5.
"""
import argparse
import struct
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

# NetCDF-3 type tag -> (numpy dtype, itemsize). Classic format is big-endian.
NC_TYPE = {1: ('i1', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4),
           5: ('>f4', 4), 6: ('>f8', 8)}


def parse(path, header_bytes=1 << 20):
    """Parse the header. Returns dict(version, numrecs, dims, vars, attrs, recsize)."""
    buf = open(path, 'rb').read(header_bytes)
    if buf[:3] != b'CDF':
        raise ValueError(f'not NetCDF-3 classic (magic {buf[:4]!r}); '
                         'HDF5-based NetCDF-4 needs the netCDF4 fallback')
    version = buf[3]
    pos = [4]

    def u32():
        v = struct.unpack_from('>I', buf, pos[0])[0]
        pos[0] += 4
        return v

    def name():
        n = u32()
        s = buf[pos[0]:pos[0] + n].decode('utf8', 'replace')
        pos[0] += n + ((4 - n % 4) % 4)          # 4-byte padded
        return s

    def attrs():
        u32()                                     # NC_ATTRIBUTE tag or ABSENT
        out = {}
        for _ in range(u32()):
            k = name()
            t, ln = u32(), u32()
            dtype, size = NC_TYPE[t]
            nbytes = size * ln
            raw = buf[pos[0]:pos[0] + nbytes]
            pos[0] += nbytes + ((4 - nbytes % 4) % 4)
            out[k] = raw.decode('utf8', 'replace') if t == 2 else np.frombuffer(raw, dtype)
        return out

    numrecs = u32()
    u32()                                         # NC_DIMENSION
    dims = [(name(), u32()) for _ in range(u32())]
    gattrs = attrs()
    u32()                                         # NC_VARIABLE
    variables = {}
    for _ in range(u32()):
        vname = name()
        ids = [u32() for _ in range(u32())]
        vattrs = attrs()
        vtype, vsize = u32(), u32()
        if version == 2:                          # 64-bit offset
            begin = struct.unpack_from('>q', buf, pos[0])[0]
            pos[0] += 8
        else:
            begin = u32()
        shape = [dims[i][1] for i in ids]
        is_record = bool(ids) and dims[ids[0]][1] == 0
        if is_record:
            shape[0] = numrecs                    # unlimited dim reports 0 in the header
        variables[vname] = dict(dims=[dims[i][0] for i in ids], shape=shape,
                                dtype=NC_TYPE[vtype][0], itemsize=NC_TYPE[vtype][1],
                                begin=begin, vsize=vsize, record=is_record, attrs=vattrs)

    # GOTCHA G2: every record variable contributes to the per-record stride.
    recsize = sum(v['vsize'] for v in variables.values() if v['record'])
    return dict(version=version, numrecs=numrecs, dims=dict(dims),
                vars=variables, attrs=gattrs, recsize=recsize)


def view(path, hdr, varname):
    """Zero-copy big-endian ndarray view of a variable. Slicing it reads from disk lazily."""
    v = hdr['vars'][varname]
    base = np.memmap(path, dtype=np.uint8, mode='r')[v['begin']:].view(v['dtype'])
    shape = tuple(v['shape'])
    it = v['itemsize']
    inner = [int(np.prod(shape[i + 1:])) * it for i in range(len(shape))]
    strides = tuple([hdr['recsize']] + inner[1:]) if v['record'] else tuple(inner)
    return as_strided(base, shape=shape, strides=strides)


def _default_path():
    """Newest data/*.nc, so the smoke test runs with no arguments."""
    here = Path(__file__).resolve().parent.parent
    candidates = sorted((here / 'data').glob('*.nc'), key=lambda f: f.stat().st_mtime)
    return candidates[-1] if candidates else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Dump the header of an IMS ICON ensemble NetCDF-3 file and time a read.')
    ap.add_argument('path', nargs='?', default=None,
                    help='.nc file (default: newest file in data/)')
    ap.add_argument('-y', type=int, default=130, help='lat index for the timing probe')
    ap.add_argument('-x', type=int, default=80, help='lon index for the timing probe')
    args = ap.parse_args(argv)

    path = Path(args.path) if args.path else _default_path()
    if path is None:
        ap.error('no file given and no *.nc found in data/ -- pass a path explicitly')
    if not path.exists():
        ap.error(f'no such file: {path}')

    hdr = parse(path)
    print(f'{path.name}: recsize={hdr["recsize"]} numrecs={hdr["numrecs"]}')
    for name, var in hdr['vars'].items():
        print(f'  {name:<14} {var["dims"]} {tuple(var["shape"])} {var["dtype"]}')
    field = next(k for k in hdr['vars'] if k.endswith('_eps'))
    arr = view(path, hdr, field)
    t0 = time.perf_counter()
    series = np.asarray(arr[:, :, args.y, args.x])
    dt_ms = (time.perf_counter() - t0) * 1e3
    print(f'{field} series {series.shape} in {dt_ms:.2f} ms, max {series.max():.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
