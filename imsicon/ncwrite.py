"""A NetCDF-3 64-bit-offset WRITER, so a derived field can be saved as a real file.

`nc3.py` reads the IMS product; this writes the same format back, which is what lets the
dew point be *written* and not merely drawn: the output is an ordinary
`ICON_ENS_<run>_<FIELD>.nc` that this app, `netCDF4`, `cdo` and `ncdump` all open.

It was built from the NetCDF-3 spec independently of `nc3.py` (which was built from the
same spec and validated against `netCDF4`), and it started life as `tests/synth.py`'s
fixture writer -- so a writer and a reader written separately agreeing byte-for-byte is
evidence rather than a tautology. `tests/test_synth.py` keeps that pinned by having
`netCDF4` read what this produces.

Two things the reader's gotchas make load-bearing here:

* **G2** -- `time` is a record variable too, so the record stride is the sum over ALL
  record variables. The layout below interleaves one `time` value with one field record,
  which is what produces the 3,361,688-byte stride the reader asserts.
* **G1** -- member identity survives only in the global `history` attribute. A derived
  file that omits it silently loses which member is which, so `history_for_members`
  rebuilds a history line the reader's `member_labels` can parse back.
"""
import datetime as dt
import re
import struct
from pathlib import Path

import numpy as np

NC_BYTE, NC_CHAR, NC_SHORT, NC_INT, NC_FLOAT, NC_DOUBLE = 1, 2, 3, 4, 5, 6
NC_DIMENSION, NC_VARIABLE, NC_ATTRIBUTE = 10, 11, 12
ABSENT = b'\x00' * 8


class Cancelled(Exception):
    """The caller's `cancel()` returned True. The partial file is removed."""


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


class FrameSource:
    """A (n_times, n_members, ny, nx) stack whose frames are computed on demand.

    A full IMS field is 407 MB, and a derived one is computed from two of those. Holding
    the result in memory to write it would triple the peak footprint for no reason, so
    `write_nc3` streams: it asks for one time step, writes it, and drops it.
    """

    def __init__(self, shape, frame):
        self.shape = tuple(shape)
        if len(self.shape) != 4:
            raise ValueError(f'expected (time, member, lat, lon), got {self.shape}')
        self._frame = frame

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, t):
        values = np.asarray(self._frame(t), dtype=np.float32)
        if values.shape != self.shape[1:]:
            raise ValueError(f'frame {t} is {values.shape}, expected {self.shape[1:]}')
        return values


def _as_stack(data):
    """Accept an ndarray or any (shape, __getitem__) stack, e.g. a FrameSource."""
    if isinstance(data, FrameSource):
        return data
    if hasattr(data, 'shape') and hasattr(data, '__getitem__') \
            and not isinstance(data, (list, tuple)):
        if len(tuple(data.shape)) != 4:
            raise ValueError(f'expected (time, member, lat, lon), got {tuple(data.shape)}')
        return data
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f'expected (time, member, lat, lon), got {data.shape}')
    return data


def history_for_members(member_labels, run, field, prefix=''):
    """A `history` line whose member numbers `nc3.member_labels` reads back (G1).

    The reader matches `ICON_ENS_<10 digits>_(\\d+)_`, so the token from each label is
    re-emitted in merge order. Without this a written file opens with member identity
    replaced by position, which looks identical and is not the same claim.
    """
    tokens = []
    for index, label in enumerate(member_labels):
        tail = str(label).split()[-1]
        tokens.append(tail if tail.isdigit() else f'{index + 1:02d}')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%a %b %d %H:%M:%S %Y')
    members = ' '.join(f'ICON_ENS_{run}_{token}_{field}.nc' for token in tokens)
    head = f'{stamp}: {prefix}' if prefix else f'{stamp}: imsicon'
    return f'{head} {members} ICON_ENS_{run}_{field}.nc'


def write_nc3(path, field, units, data, *, history=None, long_name=None,
              standard_name=None, lat=None, lon=None, time_minutes=None,
              time_units='minutes since 2026-8-23 00:00:00', global_attrs=None,
              field_attrs=None, levels=None, level_units='hPa', variable=None,
              progress=None, cancel=None):
    """Write `data` (n_times, n_levels, ny, nx) as an `ICON_ENS_<run>_<FIELD>.nc`.

    `data` may be an ndarray or a `FrameSource`, in which case frames are pulled one at a
    time. The variable is named `<field>_eps`, matching IMS.

    `levels` chooses what the second axis of the file IS:

    * `None` -- the ensemble's `sfc` axis, which is what G1 describes: 20 values that are
      all 0.0, with member identity carried only by `history`;
    * a sequence of pressures in hPa -- a real `plev` coordinate, so the file comes back
      out of `levels.axis_for` as the same levels it went in as;
    * `False` -- no vertical dimension at all, i.e. the `(time, lat, lon)` shape the
      deterministic run uses for its surface fields. `data` must then have one plane.

    Writes to `<path>.part` and renames, so an interrupted or cancelled write never leaves
    a file that `nc3.parse` would happily open as a truncated dataset (G26).
    """
    data = _as_stack(data)
    n_times, n_members, ny, nx = tuple(data.shape)

    lat = np.arange(ny, dtype=float) * 0.025 + 28.0 if lat is None else np.asarray(lat, float)
    lon = np.arange(nx, dtype=float) * 0.025 + 33.0 if lon is None else np.asarray(lon, float)
    time_minutes = (np.arange(n_times, dtype=float) * 60.0 if time_minutes is None
                    else np.asarray(time_minutes, float))
    if len(lat) != ny or len(lon) != nx or len(time_minutes) != n_times:
        raise ValueError('lat/lon/time lengths do not match the data shape')
    flat = levels is False
    if flat and n_members != 1:
        raise ValueError(f'a file with no vertical dimension needs one plane, '
                         f'got {n_members}')
    if levels is None or flat:
        level_name, level_values = 'sfc', np.zeros(n_members, dtype=float)
        level_attrs = {'long_name': 'surface', 'axis': 'Z'}
    else:
        level_name = 'plev'
        level_values = np.asarray(levels, dtype=float)
        if level_values.size != n_members:
            raise ValueError(f'{level_values.size} levels for {n_members} data planes')
        level_attrs = {'standard_name': 'air_pressure', 'long_name': 'pressure',
                       'units': level_units, 'positive': 'down', 'axis': 'Z'}

    # The ensemble names its variable `<FIELD>_eps`; the deterministic run names it after
    # the field itself, which `nc3.field_name` also accepts. `variable` is what lets a
    # fixture or a save produce either spelling.
    var_name = variable or f'{field}_eps'
    gattrs = {'CDI': 'Climate Data Interface version 2.4.0',
              'Conventions': 'CF-1.6',
              'source': 'icon-2025.04-dwd',
              'institution': 'Max Planck Institute for Meteorology/Deutscher Wetterdienst',
              'history': history if history is not None else
                         history_for_members([f'{i + 1:02d}' for i in range(n_members)],
                                             '0000000000', field)}
    gattrs.update(global_attrs or {})

    fattrs = {'standard_name': standard_name or field.lower(),
              'long_name': long_name or field.replace('_', ' ').lower(),
              'units': units}
    fattrs.update(field_attrs or {})

    dims = [('time', 0), ('lon', nx), ('lat', ny)]
    # (name, dimids, attrs, nc_type, itemsize, per-record element count, is_record)
    specs = [
        ('time', [0], {'standard_name': 'time', 'units': time_units,
                       'calendar': 'gregorian', 'axis': 'T'}, NC_DOUBLE, 8, 1, True),
        ('lon', [1], {'standard_name': 'longitude', 'long_name': 'longitude',
                      'units': 'degrees_east', 'axis': 'X'}, NC_DOUBLE, 8, nx, False),
        ('lat', [2], {'standard_name': 'latitude', 'long_name': 'latitude',
                      'units': 'degrees_north', 'axis': 'Y'}, NC_DOUBLE, 8, ny, False),
    ]
    if flat:
        # (time, lat, lon), the deterministic run's surface shape. The record bytes are
        # identical to a one-plane vertical file's, so only the header differs.
        specs.append((var_name, [0, 2, 1], fattrs, NC_FLOAT, 4, ny * nx, True))
    else:
        dims.append((level_name, n_members))
        specs.append((level_name, [3], level_attrs, NC_DOUBLE, 8, n_members, False))
        specs.append((var_name, [0, 3, 2, 1], fattrs, NC_FLOAT, 4,
                      n_members * ny * nx, True))

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
              'lat': lat.astype('>f8')}
    if not flat:
        values[level_name] = level_values.astype('>f8')

    path = Path(path)
    partial = path.with_name(path.name + '.part')
    try:
        with open(partial, 'wb') as fh:
            fh.write(build(begins))
            for vname, _ids, _va, _vt, item, count, is_record in specs:
                if not is_record:
                    fh.write(_padded(values[vname].tobytes()))
            for t in range(n_times):
                if cancel is not None and cancel():
                    raise Cancelled
                fh.write(_padded(values['time'][t:t + 1].tobytes()))
                fh.write(_padded(np.asarray(data[t], dtype='>f4').tobytes()))
                if progress is not None:
                    progress(t + 1, n_times)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def nc_variable_name(field):
    """A NetCDF-3 variable name for a field label. `T_2M-TD_2M` is not a legal name."""
    safe = re.sub(r'[^A-Za-z0-9_]', '_', str(field)).strip('_')
    return safe if safe and not safe[0].isdigit() else f'F_{safe}'


def write_view(path, view, *, field=None, units=None, long_name=None, frames=None,
               progress=None, cancel=None):
    """Write what a `FieldView` / derived view shows, as a real NetCDF-3 file.

    Grid, time axis, run init and member order are carried over from the view, so the
    result reopens in this app as an ordinary ensemble file at the same coordinates.

    `frames(t) -> (n_members, ny, nx)` overrides the source of values, and `units` labels
    them. The pair exists for one reason: a dew point belongs in a file in **Kelvin**
    however the screen is currently set, so that reopening it gets the units registry's
    ordinary treatment of a temperature rather than a "units look stale" warning. Use
    `write_canonical` rather than assembling that by hand.
    """
    field = field or view.field
    units = units if units is not None else view.units
    frames = frames if frames is not None else view.ens_frame
    run = f'{view.run_init:%Y%m%d%H}'
    axis = getattr(view, 'axis', None)

    def frame(t):
        return np.asarray(frames(t), dtype=np.float32)

    source = FrameSource((view.n_times, view.n_members, view.ny, view.nx), frame)
    provenance = getattr(view, 'provenance', None) or f'derived from {view.field}'
    # G1's history line names ENSEMBLE MEMBERS, so it is written only for a file that has
    # them. A pressure-level field carries its identity in a real `plev` coordinate
    # instead, and stamping it with twenty member names would be a false claim that
    # `nc3.member_labels` would happily read back.
    if axis is not None and axis.is_pressure:
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%a %b %d %H:%M:%S %Y')
        history = f'{stamp}: imsicon: {provenance}'
        level_values = list(axis.values)
    elif axis is not None and axis.kind == 'single':
        # A surface field goes back out as a surface field: (time, lat, lon), with no
        # vertical dimension invented for it and no member history to misread.
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%a %b %d %H:%M:%S %Y')
        history = f'{stamp}: imsicon: {provenance}'
        level_values = False
    else:
        history = history_for_members(view.member_labels, run, field,
                                      prefix=f'imsicon: {provenance};')
        level_values = None
    # `<FIELD>_eps` is the ensemble product's spelling and a claim about what the file
    # holds, so it is written only for a file that holds an ensemble. A pressure-level or
    # surface field gets the deterministic run's plain variable name instead.
    variable = (None if axis is None or axis.aggregatable
                else nc_variable_name(field))
    return write_nc3(
        path, nc_variable_name(field), units, source, variable=variable,
        lat=np.asarray(view.lat, dtype=float), lon=np.asarray(view.lon, dtype=float),
        time_minutes=np.asarray(view.forecast_hours, dtype=float) * 60.0,
        time_units=f'minutes since {view.run_init:%Y-%m-%d %H:%M:%S}',
        long_name=long_name or getattr(view, 'long_name', field),
        standard_name=field.lower(), levels=level_values,
        history=history, global_attrs={'imsicon_provenance': provenance},
        progress=progress, cancel=cancel)


def write_canonical(path, view, **kwargs):
    """Write a derived view in its OWN canonical units, not the screen's.

    For the dew point that means Kelvin, matching every temperature IMS publishes; for a
    difference there is no unit-independent space, so it is whatever the operands show.
    Either way the file says which units it holds, so nothing has to be remembered.
    """
    kwargs.setdefault('units', view.canonical_units)
    kwargs.setdefault('frames', view.canonical_ens_frame)
    return write_view(path, view, **kwargs)
