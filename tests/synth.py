"""A genuine NetCDF-3 64-bit-offset writer, for tests only (v2 phase V2.0).

`data/` holds 2 of the 15 IMS fields, and the 407 MB reference file must never become a
CI dependency (CLAUDE.md Phase 10). Temperature, precipitation, radiation and cloud cover
cannot be tested without synthesising them.

The writer is built from the NetCDF-3 spec, independently of `imsicon.nc3`, which was
built from the same spec and validated against `netCDF4`. A writer and a reader written
separately agreeing byte-for-byte is evidence, not a tautology -- and `tests/test_nc3.py`
keeps the reader pinned to netCDF4 on the real file regardless.

Layout mirrors a real IMS file: dims (time unlimited, lon, lat, sfc), variables
(time, lon, lat, sfc, <FIELD>_eps) and the field declared as (time, sfc, lat, lon), so the
non-monotonic dimension-id order of the real product is exercised too.
"""
import struct
from pathlib import Path

import numpy as np

NC_BYTE, NC_CHAR, NC_SHORT, NC_INT, NC_FLOAT, NC_DOUBLE = 1, 2, 3, 4, 5, 6
NC_DIMENSION, NC_VARIABLE, NC_ATTRIBUTE = 10, 11, 12
ABSENT = b'\x00' * 8

# A real IMS history line, so member_labels (G1) is exercised by the fixtures.
HISTORY_TEMPLATE = ('Sun Aug 23 08:37:43 2026: cdo -O -L merge ' +
                    ' '.join(f'SSN_PubMod_ICON_ENS_2026082300_{i:02d}_{{field}}.nc'
                             for i in range(1, 21)) +
                    ' SSN_PubMod_ICON_ENS_2026082300_{field}.nc')


def _pad4(n):
    return (4 - n % 4) % 4


def _padded(raw):
    return raw + b'\x00' * _pad4(len(raw))


def _name(text):
    raw = text.encode('utf8')
    return struct.pack('>I', len(raw)) + _padded(raw)


def _attr(key, value):
    if isinstance(value, str):
        raw = value.encode('utf8')
        return _name(key) + struct.pack('>II', NC_CHAR, len(raw)) + _padded(raw)
    arr = np.asarray(value)
    if arr.dtype.kind == 'i':
        return _name(key) + struct.pack('>II', NC_INT, arr.size) + \
            _padded(arr.astype('>i4').tobytes())
    return _name(key) + struct.pack('>II', NC_DOUBLE, arr.size) + \
        _padded(arr.astype('>f8').tobytes())


def _attr_list(attrs):
    if not attrs:
        return ABSENT
    out = struct.pack('>II', NC_ATTRIBUTE, len(attrs))
    for key, value in attrs.items():
        out += _attr(key, value)
    return out


def write_nc3(path, field, units, data, *, history=None, long_name=None,
              standard_name=None, lat=None, lon=None, time_minutes=None,
              time_units='minutes since 2026-8-23 00:00:00', global_attrs=None,
              field_attrs=None):
    """Write `data` (n_times, n_members, ny, nx) as ICON_ENS_<run>_<field>.nc.

    Returns the path. The variable is named `<field>_eps`, matching IMS.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f'expected (time, member, lat, lon), got {data.shape}')
    n_times, n_members, ny, nx = data.shape

    lat = np.arange(ny, dtype=float) * 0.025 + 28.0 if lat is None else np.asarray(lat, float)
    lon = np.arange(nx, dtype=float) * 0.025 + 33.0 if lon is None else np.asarray(lon, float)
    time_minutes = (np.arange(n_times, dtype=float) * 60.0 if time_minutes is None
                    else np.asarray(time_minutes, float))
    sfc = np.zeros(n_members, dtype=float)          # G1: all 20 values really are 0.0

    var_name = f'{field}_eps'
    gattrs = {'CDI': 'Climate Data Interface version 2.4.0',
              'Conventions': 'CF-1.6',
              'source': 'icon-2025.04-dwd',
              'institution': 'Max Planck Institute for Meteorology/Deutscher Wetterdienst',
              'history': history if history is not None
                         else HISTORY_TEMPLATE.format(field=field)}
    gattrs.update(global_attrs or {})

    fattrs = {'standard_name': standard_name or field.lower(),
              'long_name': long_name or field.replace('_', ' ').lower(),
              'units': units}
    fattrs.update(field_attrs or {})

    dims = [('time', 0), ('lon', nx), ('lat', ny), ('sfc', n_members)]
    # (name, dimids, attrs, nc_type, itemsize, per-record element count, is_record)
    specs = [
        ('time', [0], {'standard_name': 'time', 'units': time_units,
                       'calendar': 'gregorian', 'axis': 'T'}, NC_DOUBLE, 8, 1, True),
        ('lon', [1], {'standard_name': 'longitude', 'long_name': 'longitude',
                      'units': 'degrees_east', 'axis': 'X'}, NC_DOUBLE, 8, nx, False),
        ('lat', [2], {'standard_name': 'latitude', 'long_name': 'latitude',
                      'units': 'degrees_north', 'axis': 'Y'}, NC_DOUBLE, 8, ny, False),
        ('sfc', [3], {'long_name': 'surface', 'axis': 'Z'}, NC_DOUBLE, 8, n_members, False),
        (var_name, [0, 3, 2, 1], fattrs, NC_FLOAT, 4, n_members * ny * nx, True),
    ]

    def vsize(count, itemsize):
        nbytes = count * itemsize
        return nbytes + _pad4(nbytes)

    def build(begins):
        out = b'CDF\x02' + struct.pack('>I', n_times)
        out += struct.pack('>II', NC_DIMENSION, len(dims))
        for dname, dlen in dims:
            out += _name(dname) + struct.pack('>I', dlen)
        out += _attr_list(gattrs)
        out += struct.pack('>II', NC_VARIABLE, len(specs))
        for (vname, ids, vattrs, vtype, item, count, _rec), begin in zip(specs, begins):
            out += _name(vname) + struct.pack('>I', len(ids))
            out += b''.join(struct.pack('>I', i) for i in ids)
            out += _attr_list(vattrs)
            out += struct.pack('>I', vtype) + struct.pack('>I', vsize(count, item))
            out += struct.pack('>q', begin)          # 64-bit offset: fixed 8-byte field
        return out

    # `begin` is a fixed-width field, so a pass with placeholders gives the true header size.
    header_size = len(build([0] * len(specs)))
    begins, offset = [], header_size
    for _vname, _ids, _va, _vt, item, count, is_record in specs:
        if not is_record:
            begins.append(offset)
            offset += vsize(count, item)
        else:
            begins.append(None)
    record_start, within = offset, 0
    for i, (_vname, _ids, _va, _vt, item, count, is_record) in enumerate(specs):
        if is_record:
            begins[i] = record_start + within
            within += vsize(count, item)

    values = {'time': time_minutes.astype('>f8'), 'lon': lon.astype('>f8'),
              'lat': lat.astype('>f8'), 'sfc': sfc.astype('>f8')}

    path = Path(path)
    with open(path, 'wb') as fh:
        fh.write(build(begins))
        for vname, _ids, _va, _vt, item, count, is_record in specs:
            if not is_record:
                fh.write(_padded(values[vname].tobytes()))
        for t in range(n_times):
            fh.write(_padded(values['time'][t:t + 1].tobytes()))
            fh.write(_padded(data[t].astype('>f4').tobytes()))
    return path


# ---- fixture factories: one per shape v2 needs ----------------------------------------
SHAPE = (6, 3, 4, 5)            # times, members, lat, lon -- tiny but structurally real


def _member_spread(base, n_members, step=0.5):
    """(t, y, x) -> (t, m, y, x), members offset from each other so spread is non-zero."""
    return np.stack([base + m * step for m in range(n_members)], axis=1)


def temperature(path, n_times=6, n_members=3, ny=4, nx=5, units='K'):
    """A Kelvin field with a diurnal wiggle -- exercises the affine K->degC conversion."""
    t = np.arange(n_times)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    base = 293.15 + 5 * np.sin(t * np.pi / 6) + 0.1 * y + 0.05 * x
    return write_nc3(path, 'T_2M', units, _member_spread(base, n_members),
                     long_name='2m temperature', standard_name='air_temperature')


def accumulated_precip(path, hourly=None, n_members=3, ny=4, nx=5):
    """A monotone `kg m-2` accumulation. Returns (path, hourly) so a test can round-trip.

    `hourly[t]` is the precipitation falling in the hour ending at step t; hourly[0] is 0
    because nothing has fallen at the model's initial time.
    """
    if hourly is None:
        hourly = np.array([0.0, 0.0, 1.5, 4.0, 0.0, 2.25])
    hourly = np.asarray(hourly, dtype=float)
    accum = np.cumsum(hourly)
    base = accum[:, None, None] * np.ones((1, ny, nx))
    write_nc3(path, 'TOT_PREC', 'kg m-2', _member_spread(base, n_members, step=0.0),
              long_name='total precipitation', standard_name='precipitation_amount')
    return path, hourly


def averaged_radiation(path, hourly=None, n_members=3, ny=4, nx=5):
    """`W m-2` stored as a running MEAN since model start (G14). Returns (path, hourly).

    A[t] = (1/h[t]) * sum_{i<=t} hourly[i], with A[0] = 0 by convention.
    """
    if hourly is None:
        hourly = np.array([0.0, 100.0, 400.0, 800.0, 300.0, 50.0])
    hourly = np.asarray(hourly, dtype=float)
    hours = np.arange(len(hourly), dtype=float)
    total = np.cumsum(hourly)                      # each step is exactly one hour wide
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(hours > 0, total / np.where(hours == 0, 1, hours), 0.0)
    base = mean[:, None, None] * np.ones((1, ny, nx))
    write_nc3(path, 'ASWDIR_S', 'W m-2', _member_spread(base, n_members, step=0.0),
              long_name='direct downward sw radiation', standard_name='surface_direct_sw')
    return path, hourly


def cloud(path, encoding='fraction', units='1', n_times=6, n_members=3, ny=4, nx=5):
    """Cloud cover in one of the three G22 encodings: 'fraction', 'percent', 'zero'."""
    rng = np.random.default_rng(7)
    base = rng.random((n_times, ny, nx))
    if encoding == 'percent':
        base = base * 100.0
    elif encoding == 'zero':
        base = np.zeros((n_times, ny, nx))
    return write_nc3(path, 'CLCT', units, _member_spread(base, n_members, step=0.0),
                     long_name='total cloud cover', standard_name='cloud_area_fraction')
