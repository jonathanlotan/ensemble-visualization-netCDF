"""Synthetic IMS-shaped NetCDF-3 fixtures, for tests only (v2 phase V2.0).

`data/` holds 2 of the 15 IMS fields, and the 407 MB reference file must never become a
CI dependency (CLAUDE.md Phase 10). Temperature, precipitation, radiation and cloud cover
cannot be tested without synthesising them.

The writer itself now lives in `imsicon.ncwrite`, because a *derived* field has to be
written to a real file too and the app cannot import from `tests/`. It is still a writer
built from the NetCDF-3 spec independently of `imsicon.nc3`, which was built from the same
spec and validated against `netCDF4` -- a writer and a reader written separately agreeing
byte-for-byte is evidence, not a tautology, and `tests/test_nc3.py` keeps the reader pinned
to netCDF4 on the real file regardless.

Layout mirrors a real IMS file: dims (time unlimited, lon, lat, sfc), variables
(time, lon, lat, sfc, <FIELD>_eps) and the field declared as (time, sfc, lat, lon), so the
non-monotonic dimension-id order of the real product is exercised too.
"""
import numpy as np

from imsicon.ncwrite import write_nc3                      # noqa: F401  (re-exported)

# A real IMS history line, so member_labels (G1) is exercised by the fixtures.
HISTORY_TEMPLATE = ('Sun Aug 23 08:37:43 2026: cdo -O -L merge ' +
                    ' '.join(f'SSN_PubMod_ICON_ENS_2026082300_{i:02d}_{{field}}.nc'
                             for i in range(1, 21)) +
                    ' SSN_PubMod_ICON_ENS_2026082300_{field}.nc')


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
                     history=HISTORY_TEMPLATE.format(field='T_2M'),
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
              history=HISTORY_TEMPLATE.format(field='TOT_PREC'),
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
              history=HISTORY_TEMPLATE.format(field='ASWDIR_S'),
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
                     history=HISTORY_TEMPLATE.format(field='CLCT'),
                     long_name='total cloud cover', standard_name='cloud_area_fraction')


def humidity(path, n_times=6, n_members=3, ny=4, nx=5, units='%', encoding=None):
    """RELHUM_2M in %, shaped to pair with `temperature()` on the same grid.

    `encoding` injects the two values the dew point formula has to defend against:
    'supersaturated' puts a few cells above 100 %, 'dry' puts a few at 0 %.
    """
    t = np.arange(n_times)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    base = 55.0 + 20 * np.cos(t * np.pi / 6) + 1.5 * y + 0.5 * x
    if encoding == 'supersaturated':
        base = base.copy()
        base[0, 0, 0] = 100.4
    elif encoding == 'dry':
        base = base.copy()
        base[0, 0, 0] = 0.0
    return write_nc3(path, 'RELHUM_2M', units, _member_spread(base, n_members, step=0.0),
                     history=HISTORY_TEMPLATE.format(field='RELHUM_2M'),
                     long_name='relative humidity in 2m', standard_name='relative_humidity')


def pair(tmp_path, n_times=6, n_members=3, ny=4, nx=5, humidity_encoding=None,
         run='2026082300'):
    """A matching (T_2M, RELHUM_2M) pair -- what the dew point is built from."""
    return (temperature(tmp_path / f'ICON_ENS_{run}_T_2M.nc', n_times, n_members, ny, nx),
            humidity(tmp_path / f'ICON_ENS_{run}_RELHUM_2M.nc', n_times, n_members, ny, nx,
                     encoding=humidity_encoding))


# ---- the deterministic ICON-LAM shapes (v7) -------------------------------------------
# `IE_<run>_<field>.nc`, the variable named after the field rather than `<FIELD>_eps`, and
# either a real `plev` coordinate or no vertical dimension at all. Both are written by the
# same production writer the app saves with, so a fixture cannot drift from what the app
# produces -- and neither is a guess about the IMS file: the app decides the axis from
# what a file actually carries (`levels.axis_for`), and these are the shapes that decision
# has to get right.
ICON_LEVELS = (1000, 925, 850, 700, 500)


def _run_units(run):
    """`minutes since <run>` -- so a fixture's header agrees with the run in its name."""
    return (f'minutes since {run[:4]}-{int(run[4:6])}-{int(run[6:8])} '
            f'{int(run[8:10])}:00:00')


def pressure_field(path, field='temp', units='K', n_times=4, ny=4, nx=5, levels=None,
                   values=None, level_units='hPa', variable=None, run='2026083100'):
    """A 3-D field on pressure levels: (time, plev, lat, lon).

    The default pattern falls with height the way an atmosphere does, so a test can tell
    one level from another by its values alone.
    """
    levels = ICON_LEVELS if levels is None else tuple(levels)
    t = np.arange(n_times)[:, None, None, None]
    p = np.asarray(levels, dtype=float)[None, :, None, None]
    y = np.arange(ny)[None, None, :, None]
    x = np.arange(nx)[None, None, None, :]
    if values is None:
        # ~6.5 K per km, roughly hydrostatic, plus a diurnal wiggle and a spatial tilt.
        values = 220.0 + 0.06 * p + 2.0 * np.sin(t * np.pi / 6) + 0.1 * y + 0.05 * x
        values = np.broadcast_to(values, (n_times, len(levels), ny, nx))
    written = np.asarray(levels, dtype=float)
    if level_units == 'Pa':
        written = written * 100.0
    return write_nc3(path, field, units, values, levels=written,
                     variable=variable or field, time_units=_run_units(run),
                     long_name=f'{field} on pressure levels', standard_name=field,
                     global_attrs={'history': 'icon-lam deterministic run'},
                     level_units=level_units)


def surface_field(path, field='t_2m', units='K', n_times=4, ny=4, nx=5, values=None,
                  variable=None, run='2026083100'):
    """A deterministic surface field: (time, lat, lon), with no vertical dimension."""
    t = np.arange(n_times)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    if values is None:
        values = 293.15 + 4 * np.sin(t * np.pi / 6) + 0.2 * y + 0.1 * x
        values = np.broadcast_to(values, (n_times, ny, nx))
    return write_nc3(path, field, units, np.asarray(values, float)[:, None],
                     levels=False, variable=variable or field, time_units=_run_units(run),
                     long_name=f'{field} at the surface', standard_name=field,
                     global_attrs={'history': 'icon-lam deterministic run'})


def icon_run(tmp_path, run='2026083100', fields=('temp', 't_2m'), **kwargs):
    """One deterministic run on disk: 3-D fields get levels, surface fields do not."""
    out = {}
    for field in fields:
        path = tmp_path / f'IE_{run}_{field}.nc'
        if field in ('temp', 'rh', 'u', 'v', 'omega', 'geopot'):
            out[field] = pressure_field(path, field, _ICON_UNITS.get(field, 'K'),
                                        run=run, **kwargs)
        else:
            out[field] = surface_field(path, field, _ICON_UNITS.get(field, 'K'),
                                       run=run, **kwargs)
    return out


_ICON_UNITS = {'temp': 'K', 'rh': '%', 'u': 'm s-1', 'v': 'm s-1', 'omega': 'Pa s-1',
               'geopot': 'm2 s-2', 't_2m': 'K', 'td_2m': 'K', 'rh_2m': '%',
               'u_10m': 'm s-1', 'v_10m': 'm s-1', 'tot_prec': 'kg m-2',
               'pres_msl': 'Pa', 'clct': '%'}


def wind_component(path, field, base, n_members=3, step=0.5, units='m s-1'):
    """One component of the wind, `m s-1`, with the members offset from each other."""
    long_names = {'U_10M': 'zonal wind in 10m', 'V_10M': 'meridional wind in 10m'}
    return write_nc3(path, field, units, _member_spread(np.asarray(base, float),
                                                        n_members, step),
                     history=HISTORY_TEMPLATE.format(field=field),
                     long_name=long_names.get(field, field), standard_name=field.lower())


def wind_pair(tmp_path, n_times=6, n_members=3, ny=4, nx=5, run='2026082300',
              u=None, v=None, units='m s-1'):
    """A matching (U_10M, V_10M) pair -- what the wind map and its barbs are built from.

    The default pattern sweeps the speed across the domain from calm to about 39 kt, so a
    single frame exercises the calm circle, half feathers and full feathers at once, and
    turns the wind through the compass as time advances.
    """
    t = np.arange(n_times)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    if u is None:
        u = 0.0 * t + 0.0 * y + 5.0 * x                  # 0, 5, 10, 15, 20 m s-1 eastward
    if v is None:
        v = -(2.0 + 0.5 * t) + 0.0 * y + 0.0 * x         # northerly, freshening with time
    u, v = np.broadcast_to(u, (n_times, ny, nx)), np.broadcast_to(v, (n_times, ny, nx))
    return (wind_component(tmp_path / f'ICON_ENS_{run}_U_10M.nc', 'U_10M', u,
                           n_members, step=0.5, units=units),
            wind_component(tmp_path / f'ICON_ENS_{run}_V_10M.nc', 'V_10M', v,
                           n_members, step=-0.25, units=units))
