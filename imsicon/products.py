"""The two IMS ICON product families, and everything that differs between them.

IMS publishes the same model twice (`IMS_ICON_manual.pdf`, and CLAUDE.md 0.1 for the
ensemble):

* **the ensemble** -- `ICON_ENS_<YYYYMMDDHH>_<FIELD>.nc.bz2`, one 00Z run a day, 15
  surface fields, and a second data axis of **20 ensemble members** (G1);
* **the deterministic run** (ICON-LAM) -- `IE_<YYYYMMDDHH>_<field>.nc.bz2`, two runs a
  day (00Z and 12Z), 48 fields, six of which are 3-D on **20 pressure levels**.

Everything that differs between them is here, once: the file-name grammar, the server
folder, the catalogue, and which field plays which role in a derived field. `download.py`
fetches, `ingest.py` finds files on disk and `ui/` presents them, and none of the three
carries a second copy of the naming rules.

The pressure levels below are the manual's list, in the manual's order. They are a
**fallback**: `levels.py` reads the axis out of the file whenever the file carries one,
and only falls back to this table when it does not (see `levels.axis_for`).
"""
import re
from pathlib import Path

# MEASURED 2026-08-31 against run 2026083012, from the `plev` coordinate of all six 3-D
# files: 22 levels, stored in Pa, ASCENDING in pressure -- 15000 Pa (150 hPa) first and
# 100000 Pa (1000 hPa) last. The spacing is not uniform (50 hPa apart at the top, 25 at
# the bottom), so a level index is not a linear function of pressure.
#
# The manual's Table 1 footnote lists TWENTY levels in the OPPOSITE order, and it is
# stale on both counts: the files add 450 and 550 hPa, and store the ladder the other way
# up. Both are kept here because the difference is the point -- it is why the axis is read
# out of each file's own coordinate (G39) and why "up" is defined by pressure rather than
# by index (G40). Either would have been enough on its own to label every map wrongly.
PRESSURE_LEVELS = (150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650,
                   700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000)
MANUAL_PRESSURE_LEVELS = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 750,
                          700, 650, 600, 500, 400, 350, 300, 250, 200, 150)


class Product:
    """One downloadable field. `levels` marks the 3-D ones (the manual's asterisk)."""
    __slots__ = ('field', 'label', 'group', 'note', 'levels')

    def __init__(self, field, label, group, note='', levels=False):
        self.field, self.label, self.group = field, label, group
        self.note, self.levels = note, levels

    def __repr__(self):
        return f'<Product {self.field}>'


# ---- the ensemble catalogue (CLAUDE.md 0.2, measured 2026-08-24) -----------------------
ENSEMBLE_PRODUCTS = (
    Product('CAPE_ML', 'CAPE - instability', 'Convection',
            'CAPE of the mean surface layer parcel [J kg-1] - the "cape index"'),
    Product('TOT_PREC', 'Precipitation - total', 'Convection',
            'accumulated since model start [kg m-2 = mm]; the viewer can show 1 h / 3 h rates'),
    Product('T_2M', 'Temperature - 2 m', 'Temperature and humidity',
            'stored in K, shown in °C by default'),
    Product('T_S', 'Temperature - surface', 'Temperature and humidity',
            'weighted surface temperature, stored in K'),
    Product('RELHUM_2M', 'Relative humidity - 2 m', 'Temperature and humidity',
            'needed, with T_2M, to derive the dew point'),
    Product('U_10M', 'Wind - zonal component (U) 10 m', 'Wind', 'm s-1, positive eastward'),
    Product('V_10M', 'Wind - meridional component (V) 10 m', 'Wind',
            'm s-1, positive northward'),
    Product('VMAX_10M', 'Wind - gust at 10 m', 'Wind',
            'max since the previous full hour - already per-interval, never de-accumulate'),
    Product('CLCT', 'Cloud cover - total', 'Cloud', '%'),
    Product('CLCL', 'Cloud cover - low', 'Cloud', '%'),
    Product('CLCM', 'Cloud cover - medium', 'Cloud', '%'),
    Product('CLCH', 'Cloud cover - high', 'Cloud', '%'),
    Product('ASWDIFD_S', 'Solar radiation - diffuse downward', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('ASWDIR_S', 'Solar radiation - direct downward', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('H_SNOW', 'Snow depth', 'Surface', 'm; ~all zero in summer'),
)


# ---- the deterministic catalogue (IMS_ICON_manual.pdf, Table 1) ------------------------
# Field names, descriptions, units and grouping are the manual's, transcribed verbatim;
# the menu labels are this app's. Rows 49-51 of the manual's table are deliberately absent:
# `surf_feilds` [sic] and `use_cams` are marked "not yet implemented", and
# `topo_icon_web` is a constant field that lives in MANUALS/IMS_ICON, not in a run.
ICON_PRODUCTS = (
    # *3-D fields on pressure levels -- the manual's asterisked group.
    Product('temp', 'Temperature - on pressure levels', 'Pressure levels',
            'air temperature [K] on the 20 pressure levels, 1000 to 150 hPa', levels=True),
    Product('rh', 'Relative humidity - on pressure levels', 'Pressure levels',
            'relative humidity [%] on the 20 pressure levels', levels=True),
    Product('u', 'Wind - zonal component (U) on pressure levels', 'Pressure levels',
            'eastward wind [m s-1] on the 20 pressure levels', levels=True),
    Product('v', 'Wind - meridional component (V) on pressure levels', 'Pressure levels',
            'northward wind [m s-1] on the 20 pressure levels', levels=True),
    Product('omega', 'Vertical velocity - on pressure levels', 'Pressure levels',
            'omega [Pa s-1]; NEGATIVE is rising air, because pressure falls upward',
            levels=True),
    Product('geopot', 'Geopotential - on pressure levels', 'Pressure levels',
            'geopotential at full level cell centre [m2 s-2]; divide by 9.80665 for '
            'geopotential height in m', levels=True),
    # Surface: the fields a forecaster reads first.
    Product('t_2m', 'Temperature - 2 m', 'Surface', 'stored in K, shown in °C by default'),
    Product('td_2m', 'Dew point - 2 m', 'Surface',
            'stored in K; with t_2m this gives the depression T-Td directly'),
    Product('t_g', 'Temperature - surface (grid mean)', 'Surface', 'K'),
    Product('tmax_2m', 'Temperature - 2 m maximum', 'Surface', 'K'),
    Product('tmin_2m', 'Temperature - 2 m minimum', 'Surface', 'K'),
    Product('rh_2m', 'Relative humidity - 2 m', 'Surface', '%'),
    Product('qv_s', 'Specific humidity - at the surface', 'Surface', 'kg kg-1'),
    Product('pres_msl', 'Pressure - mean sea level', 'Surface',
            'Pa in the file; shown in hPa by default, which is what a chart is drawn in'),
    Product('pres_sfc', 'Pressure - surface', 'Surface', 'Pa'),
    Product('u_10m', 'Wind - zonal component (U) 10 m', 'Surface', 'm s-1, positive eastward'),
    Product('v_10m', 'Wind - meridional component (V) 10 m', 'Surface',
            'm s-1, positive northward'),
    Product('gust10', 'Wind - gust at 10 m', 'Surface', 'm s-1'),
    # Convection, cloud and rain.
    Product('cape', 'CAPE - instability', 'Convection', 'J kg-1'),
    Product('cape_ml', 'CAPE - mean surface layer parcel', 'Convection',
            'J kg-1 - the "cape index"'),
    Product('cin_ml', 'CIN - convective inhibition', 'Convection',
            'J kg-1, of the mean surface layer parcel'),
    Product('hzerocl', 'Freezing level - height of 0 °C', 'Convection', 'm'),
    Product('hbas_con', 'Convective cloud - base height', 'Convection', 'm'),
    Product('htop_con', 'Convective cloud - top height', 'Convection', 'm'),
    Product('clct', 'Cloud cover - total', 'Cloud', '%'),
    Product('clcl', 'Cloud cover - low', 'Cloud', '%'),
    Product('clcm', 'Cloud cover - medium', 'Cloud', '%'),
    Product('clch', 'Cloud cover - high', 'Cloud', '%'),
    Product('tot_prec', 'Precipitation - total', 'Precipitation',
            'accumulated since model start [kg m-2 = mm]; the viewer can show 1 h / 3 h rates'),
    Product('rain_gsp', 'Precipitation - large scale rain', 'Precipitation',
            'kg m-2, accumulated'),
    Product('rain_con', 'Precipitation - convective rain', 'Precipitation',
            'kg m-2, accumulated'),
    Product('snow_gsp', 'Precipitation - large scale snow', 'Precipitation',
            'kg m-2, accumulated'),
    Product('snow_con', 'Precipitation - convective snow', 'Precipitation',
            'kg m-2, accumulated'),
    Product('graupel_gsp', 'Precipitation - graupel', 'Precipitation',
            'kg m-2, accumulated'),
    Product('h_snow', 'Snow depth', 'Surface', 'm'),
    # Column integrals.
    Product('tqv', 'Column - water vapour (precipitable water)', 'Column integrals',
            'kg m-2'),
    Product('tqc', 'Column - cloud water', 'Column integrals', 'kg m-2'),
    Product('tqi', 'Column - cloud ice', 'Column integrals', 'kg m-2'),
    Product('tqr', 'Column - rain', 'Column integrals', 'kg m-2'),
    Product('tqs', 'Column - snow', 'Column integrals', 'kg m-2'),
    Product('tqg', 'Column - graupel', 'Column integrals', 'kg m-2'),
    # Radiation. The five "a"-prefixed fields are MEANS since model start (G14).
    Product('asodifd_s', 'Solar radiation - diffuse downward (mean)', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('asodifu_s', 'Solar radiation - diffuse upward (mean)', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('asodird_s', 'Solar radiation - direct downward (mean)', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('sodifd_s', 'Solar radiation - diffuse downward flux', 'Radiation',
            'W m-2, instantaneous'),
    # Measured on the server 2026-08-31 and absent from the manual's Table 1: the
    # catalogue follows what is published, not only what is documented.
    Product('sob_s', 'Radiation - net solar at the surface', 'Radiation',
            'W m-2, instantaneous; on the server, not in the manual'),
    Product('sou_s', 'Radiation - upward solar at the surface', 'Radiation',
            'W m-2, instantaneous; on the server, not in the manual'),
    Product('sob_t', 'Radiation - net solar at top of atmosphere', 'Radiation',
            'W m-2, instantaneous'),
    Product('asob_t', 'Radiation - net solar at TOA (mean)', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('athb_t', 'Radiation - net thermal at TOA (mean)', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
)


class Family:
    """One product family: how its files are named, where they live, and what is in them.

    `roles` names the field that plays each part in a derived field, so `derived.py` can
    ask for "this family's temperature" rather than hard-coding `T_2M`. A role a family
    does not have simply is not in the dict -- the deterministic run publishes `td_2m`
    directly, so it needs no computed dew point.
    """

    def __init__(self, key, title, short, prefix, base_url, products, roles,
                 default_axis='member', lower=False):
        self.key = key
        self.title = title
        self.short = short
        self.prefix = prefix
        self.base_url = base_url
        self.products = tuple(products)
        self.by_field = {p.field: p for p in self.products}
        self.groups = tuple(dict.fromkeys(p.group for p in self.products))
        self.roles = dict(roles)
        self.default_axis = default_axis
        # Field names are upper-case in the ensemble product and lower-case in the
        # deterministic one, which is what keeps the two apart in one scan of a directory.
        self.lower = lower
        letters = 'a-z' if lower else 'A-Z'
        self.name_re = re.compile(
            rf'^{prefix}(?P<run>\d{{10}})_(?P<field>[{letters}0-9_]+)\.nc\.bz2$')
        self.file_re = re.compile(
            rf'^{prefix}(?P<run>\d{{10}})_(?P<field>[{letters}0-9_]+)'
            r'\.nc(?P<bz2>\.bz2)?$')

    def local_name(self, run, field, compressed=True):
        """G27: a name on disk is REBUILT from validated parts, never echoed back."""
        return f'{self.prefix}{run}_{field}.nc' + ('.bz2' if compressed else '')

    def label_for(self, field):
        product = self.by_field.get(field)
        return product.label if product else field

    def has_levels(self, field):
        """Does the catalogue say this field is 3-D on pressure levels?"""
        product = self.by_field.get(field)
        return bool(product and product.levels)

    def match(self, name):
        """-> re.Match for a file of this family (compressed or not), else None."""
        return self.file_re.match(str(name))

    def __repr__(self):
        return f'<Family {self.key}>'


BASE = 'https://data.israel-meteo-service.org/ims/'

# The ensemble folder is measured (CLAUDE.md 0.1). The deterministic one is INFERRED from
# the server's own layout -- the ensemble lives in /ims/IMS_ICON_ENSEMBLE/ and the ICON
# manuals and topography in /ims/MANUALS/IMS_ICON/ -- and could not be confirmed from here:
# the server answers 401 to an unauthenticated request for every path, real or not, so a
# probe cannot tell a wrong folder from a private one. `IMS_ICON_URL` overrides it without
# a code change, and the download dialog says so when the listing comes back empty.
ENSEMBLE = Family(
    'ens', 'IMS ICON ensemble', 'ensemble', 'ICON_ENS_', BASE + 'IMS_ICON_ENSEMBLE/',
    ENSEMBLE_PRODUCTS,
    roles={'temperature': 'T_2M', 'humidity': 'RELHUM_2M',
           'zonal': 'U_10M', 'meridional': 'V_10M'},
    default_axis='member')

ICON = Family(
    'icon', 'IMS ICON deterministic (ICON-LAM)', 'deterministic', 'IE_',
    BASE + 'IMS_ICON/', ICON_PRODUCTS,
    # Two wind pairs: the 10 m components, and the 3-D `u`/`v` on pressure levels -- so
    # the wind map can be drawn at the surface or at 850 hPa out of the same machinery.
    roles={'temperature': 't_2m', 'humidity': 'rh_2m',
           'zonal': 'u_10m', 'meridional': 'v_10m',
           'zonal_upper': 'u', 'meridional_upper': 'v',
           # R9: the geopotential, read as the height of every pressure level.
           'height': 'geopot'},
    default_axis='pressure', lower=True)

FAMILIES = (ENSEMBLE, ICON)
BY_KEY = {family.key: family for family in FAMILIES}


def family_of(name, default=None):
    """Which family a file name belongs to, or `default` when it belongs to neither."""
    stem = Path(str(name)).name
    for family in FAMILIES:
        if family.match(stem):
            return family
    return default


def parse_name(name):
    """-> (family, run, field) for an IMS product file name, else None."""
    stem = Path(str(name)).name
    for family in FAMILIES:
        match = family.match(stem)
        if match:
            return family, match.group('run'), match.group('field')
    return None


def field_key(field):
    """The registry key for a field name, so `t_2m` and `T_2M` are one quantity.

    The two families spell the same field differently (`TOT_PREC` against `tot_prec`) and
    they are the same physical quantity out of the same model, so the units registry, the
    accumulation table and the isoline intervals are all keyed on this rather than on the
    literal name. What is NOT folded together is anything a name is parsed back OUT of --
    the file name, the NetCDF variable, the settings key -- because those identify a file.
    """
    return str(field).upper()


def role_field(family, role):
    """The field that plays `role` in this family, or None."""
    return (family.roles.get(role) if family is not None else None)
