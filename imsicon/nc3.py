"""NetCDF-3 classic / 64-bit-offset reader for IMS ICON ensemble files.

Header parse in pure Python + zero-copy strided numpy.memmap over the data.
No netCDF4/HDF5/xarray needed. Verified element-for-element against netCDF4.Dataset
on ICON_ENS_2026082300_CAPE_ML.nc (2026-08-23). See CLAUDE.md sections 0.4 and 0.5.
"""
import datetime as dt
import re
import struct

import numpy as np
from numpy.lib.stride_tricks import as_strided

class UnsupportedFormat(Exception):
    """File is not a NetCDF-3 classic / 64-bit-offset file."""


# NetCDF-3 type tag -> (numpy dtype, itemsize). Classic format is big-endian.
NC_TYPE = {1: ('i1', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4),
           5: ('>f4', 4), 6: ('>f8', 8)}


def parse(path, header_bytes=1 << 20):
    """Parse the header. Returns dict(version, numrecs, dims, vars, attrs, recsize)."""
    buf = open(path, 'rb').read(header_bytes)
    if buf[:3] != b'CDF':
        kind = 'HDF5-based NetCDF-4' if buf[:4] == b'\x89HDF' else 'unrecognised'
        raise UnsupportedFormat(
            f'{path}: {kind} file (magic {buf[:4]!r}). This reader handles NetCDF-3 '
            'classic / 64-bit-offset only, which is what IMS ICON ensemble files use.')
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


def field_name(hdr):
    """Name of the ensemble field variable, e.g. 'CAPE_ML_eps'.

    IMS names it '<FIELD>_eps'; fall back to the only 4-D variable so the app still
    opens a file whose naming convention changed.
    """
    eps = [k for k in hdr['vars'] if k.endswith('_eps')]
    if len(eps) == 1:
        return eps[0]
    four_d = [k for k, v in hdr['vars'].items() if len(v['dims']) == 4]
    if len(four_d) == 1:
        return four_d[0]
    raise UnsupportedFormat(
        f'expected one ensemble field variable, found {eps or four_d or list(hdr["vars"])}')


# G4: units carry a NON zero-padded date -- "minutes since 2026-8-23 00:00:00".
# datetime.fromisoformat() rejects that, so parse it tolerantly.
_UNITS_RE = re.compile(
    r'(?P<unit>\w+)\s+since\s+(?P<y>\d{4})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})'
    r'(?:[ T](?P<h>\d{1,2}):(?P<mi>\d{1,2})(?::(?P<s>\d{1,2}(?:\.\d+)?))?)?')
_UNIT_SECONDS = {'second': 1, 'seconds': 1, 'minute': 60, 'minutes': 60,
                 'hour': 3600, 'hours': 3600, 'day': 86400, 'days': 86400}


def parse_time_units(units):
    """'minutes since 2026-8-23 00:00:00' -> (epoch datetime UTC, seconds per unit)."""
    m = _UNITS_RE.search(units or '')
    if not m:
        raise ValueError(f'unrecognised time units: {units!r}')
    scale = _UNIT_SECONDS.get(m.group('unit').lower())
    if scale is None:
        raise ValueError(f'unsupported time unit: {m.group("unit")!r}')
    epoch = dt.datetime(int(m.group('y')), int(m.group('mo')), int(m.group('d')),
                        int(m.group('h') or 0), int(m.group('mi') or 0),
                        int(float(m.group('s') or 0)), tzinfo=dt.timezone.utc)
    return epoch, scale


# G1: the 'sfc' axis is the 20 ensemble members, not a vertical level, and all 20 of its
# coordinate values are 0.0. Member identity survives only in the cdo merge command.
_MEMBER_RE = re.compile(r'ICON_ENS_\d{10}_(\d+)_')


def member_labels(history, n_members):
    """Member numbers recovered from the global 'history' attribute, in merge order."""
    seen = []
    for tok in _MEMBER_RE.findall(history or ''):
        if tok not in seen:
            seen.append(tok)
    if len(seen) >= n_members:
        return [f'member {t}' for t in seen[:n_members]]
    return [f'member {i + 1:02d}' for i in range(n_members)]      # positional fallback
