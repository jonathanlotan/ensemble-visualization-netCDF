"""The user's configuration file: credentials, map points, and per-map defaults (R10).

One TOML file a person can edit by hand in any text editor, and which the Settings dialog
(`ui/settingsdialog.py`) reads and writes. It lives in the per-user config directory --
never in the repository, because it can hold the IMS password:

    Windows   %APPDATA%\\IMSIconViewer\\config.toml
    macOS     ~/Library/Application Support/IMSIconViewer/config.toml
    Linux     $XDG_CONFIG_HOME/imsicon/config.toml   (~/.config/imsicon/config.toml)

`IMSICON_CONFIG=/some/path.toml` overrides the location, which is also how the test suite
keeps a developer's real file out of a test run (the G25/G47 lesson, applied up front).

Everything in it is a DEFAULT. The toolbar still changes units, colours, the scale, the
isolines and the profile exactly as before; a choice made in the app wins for the rest of
that session, and the next launch starts from the file again.

Rules this module keeps, all of them load-bearing:

* **Loading never raises.** A typo in a hand-edited file must cost that one setting, not
  the app -- the same rule G21 set for a units string. Every problem becomes a sentence in
  `Config.warnings`, and the rest of the file still applies.
* **Field names are folded like the registry folds them** (`products.field_key`), so
  `[fields.T_2M]` covers the ensemble's `T_2M` and the deterministic run's `t_2m` alike,
  and the derived maps are addressed by the names on screen: `TD_2M`, `T-Td`, `WSPD_10M`.
* **The password is written only to a file only its owner can read** (0600 on POSIX), and
  never logged. `download.stored_credentials` reads it after the environment and before the
  system keychain.
* **No Qt here.** The dialog and the window are the Qt half; this is plain data, so it can
  be tested and used from the command line without a display.
"""
import json
import math
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field as dc_field, fields as dc_fields, replace
from pathlib import Path

from . import timefmt
from .products import field_key

ENV = 'IMSICON_CONFIG'
FILE_NAME = 'config.toml'
DEFAULT_POINT_COLOUR = 'blue'
SCALE_MODES = ('dataset', 'frame')
# The degree sign is awkward to type, in a text file as much as at a shell prompt.
UNIT_ALIASES = {'C': '°C', 'c': '°C', 'degC': '°C', 'deg C': '°C',
                'F': '°F', 'f': '°F', 'degF': '°F', 'deg F': '°F'}


def config_dir():
    """The per-user configuration directory (not created here)."""
    if sys.platform.startswith('win'):
        root = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        return root / 'IMSIconViewer'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'IMSIconViewer'
    return Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'imsicon'


def config_path():
    override = os.environ.get(ENV, '').strip()
    return Path(override).expanduser() if override else config_dir() / FILE_NAME


# ---- the data ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    """A dot on the map. `colour` is any name Qt knows ('blue', 'orange') or '#rrggbb'."""
    lat: float
    lon: float
    colour: str = DEFAULT_POINT_COLOUR
    name: str = ''

    def as_line(self):
        """`31.7683, 35.2137, red, Jerusalem` -- the form a person writes in the file."""
        parts = [f'{self.lat:g}', f'{self.lon:g}', self.colour]
        if self.name:
            parts.append(self.name)
        return ', '.join(parts)


@dataclass
class FieldDefaults:
    """What a map opens with. Every attribute None means "the app's own default"."""
    name: str = ''                 # as the user spelled it, so the file round-trips
    units: str | None = None       # a label from the Units combo: '°C', 'kt', 'mm', 'ft'
    colours: str | None = None     # a colour ramp from the Colours combo: 'turbo', 'viridis'
    scale: object = None           # 'dataset' | 'frame' | (lo, hi)
    scale_units: str | None = None  # units (lo, hi) are stated in; default: `units`
    isolines: bool | None = None   # lines on or off when the map opens
    isoline_step: float | None = None   # spacing, in the field's isoline unit
    profile: bool | None = None    # right-hand panel as a vertical profile

    @property
    def fixed_range(self):
        return self.scale if isinstance(self.scale, tuple) else None

    def is_empty(self):
        return all(getattr(self, f.name) is None for f in dc_fields(self) if f.name != 'name')


EMPTY = FieldDefaults()


@dataclass
class Config:
    user: str = ''
    password: str = ''
    show_points: bool = True
    time_zone: str = timefmt.ZULU  # R11: the clock every time on screen is written in
    # R12: the colour scale every map opens with, unless its [fields] table names one
    map_colours: str | None = None         # sequential maps; None = turbo
    difference_colours: str | None = None  # difference maps; None = CET-D1A
    custom_colours: tuple = ()             # the "custom" ramp, bottom to top
    points: list = dc_field(default_factory=list)
    fields: dict = dc_field(default_factory=dict)      # field_key -> FieldDefaults
    warnings: list = dc_field(default_factory=list)
    path: Path | None = None
    unreadable: bool = False       # the file exists but could not be parsed at all

    def credentials(self):
        """-> (user, password), or None unless both are set."""
        return (self.user, self.password) if self.user and self.password else None

    def for_names(self, *names):
        """The defaults for the first of `names` the file mentions, else `EMPTY`."""
        for name in names:
            if name:
                found = self.fields.get(field_key(name))
                if found is not None:
                    return found
        return EMPTY

    def for_view(self, ds):
        """A view is looked up by its identity first, then by the name on screen -- so
        `T-Td` finds `[fields.T-Td]` although its machine name is `T_2M-TD_2M`."""
        if ds is None:
            return EMPTY
        return self.for_names(getattr(ds, 'field', None), getattr(ds, 'display_name', None))

    def custom_isolines(self):
        """-> {field_key: step} for every field the file gives a spacing for."""
        return {key: d.isoline_step for key, d in self.fields.items()
                if d.isoline_step is not None}


# ---- reading -----------------------------------------------------------------------------
def _number(value):
    if isinstance(value, bool):
        raise ValueError('a number, not true/false')
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip())
        except ValueError:
            raise ValueError(f'{value!r} is not a number') from None
    if not math.isfinite(number):
        raise ValueError('a finite number')
    return number


def parse_point(value):
    """`"lat, lon[, colour[, name]]"` or `{lat=, lon=, colour=, name=}` -> Point.

    An empty colour is blue, as asked. Raises ValueError with a readable reason.
    """
    if isinstance(value, dict):
        lat, lon = value.get('lat'), value.get('lon')
        colour = value.get('colour', value.get('color', ''))
        name = value.get('name', '')
    elif isinstance(value, str):
        parts = [p.strip() for p in value.split(',')]
        if len(parts) < 2:
            raise ValueError('needs at least "lat, lon"')
        lat, lon = parts[0], parts[1]
        colour = parts[2] if len(parts) > 2 else ''
        # A name may itself contain a comma ("Tel Aviv, port"): everything after the third.
        name = ', '.join(parts[3:]) if len(parts) > 3 else ''
    else:
        raise ValueError('expected "lat, lon, colour" or a table')
    try:
        lat, lon = _number(lat), _number(lon)
    except (TypeError, ValueError):
        raise ValueError('latitude and longitude must be numbers') from None
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f'latitude {lat:g} is outside -90..90')
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f'longitude {lon:g} is outside -180..180')
    colour = str(colour or '').strip() or DEFAULT_POINT_COLOUR
    return Point(lat, lon, colour, str(name or '').strip())


def normalise_units_label(label):
    label = str(label).strip()
    return UNIT_ALIASES.get(label, label)


def _tristate(value, what):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('on', 'true', 'yes', '1'):
        return True
    if text in ('off', 'false', 'no', '0'):
        return False
    raise ValueError(f'{what} must be true or false')


def parse_field(name, table):
    """One `[fields.<name>]` table -> (FieldDefaults, [warnings])."""
    warnings = []
    out = FieldDefaults(name=str(name))
    if not isinstance(table, dict):
        return out, [f'[fields.{name}] is not a table; ignored']

    def bad(key, why):
        warnings.append(f'[fields.{name}] {key}: {why}; ignored')

    known = {f.name for f in dc_fields(FieldDefaults)} - {'name'}
    aliases = {'colors': 'colours', 'colour': 'colours', 'color': 'colours',
               'isoline_spacing': 'isoline_step', 'spacing': 'isoline_step'}
    for raw_key, value in table.items():
        key = aliases.get(raw_key, raw_key)
        if key not in known:
            bad(raw_key, 'not a setting this version knows')
            continue
        try:
            if key in ('units', 'scale_units'):
                text = normalise_units_label(value)
                if text:
                    setattr(out, key, text)
            elif key == 'colours':
                text = str(value).strip()
                if text:
                    out.colours = text
            elif key == 'scale':
                if isinstance(value, (list, tuple)):
                    if len(value) != 2:
                        raise ValueError('a range is two numbers, [min, max]')
                    lo, hi = _number(value[0]), _number(value[1])
                    if not hi > lo:
                        raise ValueError(f'max ({hi:g}) must be above min ({lo:g})')
                    out.scale = (lo, hi)
                else:
                    text = str(value).strip().lower()
                    mode = {'dataset range': 'dataset', 'this frame': 'frame'}.get(text, text)
                    if mode not in SCALE_MODES:
                        raise ValueError('"dataset", "frame" or [min, max]')
                    out.scale = mode
            elif key in ('isolines', 'profile'):
                setattr(out, key, _tristate(value, key))
            elif key == 'isoline_step':
                step = _number(value)
                if step <= 0:
                    raise ValueError('a spacing must be above zero')
                out.isoline_step = step
        except (TypeError, ValueError) as exc:
            bad(raw_key, str(exc))
    return out, warnings


def from_dict(data, path=None):
    """A parsed TOML document -> Config. Total: every problem becomes a warning."""
    cfg = Config(path=path)
    if not isinstance(data, dict):
        cfg.warnings.append('the file is not a TOML table')
        return cfg

    creds = data.get('credentials', {})
    if isinstance(creds, dict):
        cfg.user = str(creds.get('user', '') or '').strip()
        cfg.password = str(creds.get('password', '') or '')
    else:
        cfg.warnings.append('[credentials] is not a table; ignored')

    map_table = data.get('map', {})
    if not isinstance(map_table, dict):
        cfg.warnings.append('[map] is not a table; ignored')
        map_table = {}
    try:
        cfg.show_points = _tristate(map_table.get('show_points', True), 'show_points')
    except ValueError as exc:
        cfg.warnings.append(f'[map] {exc}; using true')
    points = map_table.get('points', [])
    if not isinstance(points, list):
        cfg.warnings.append('[map] points must be a list; ignored')
        points = []
    for i, value in enumerate(points, 1):
        try:
            cfg.points.append(parse_point(value))
        except ValueError as exc:
            cfg.warnings.append(f'[map] point {i} ({value!r}): {exc}; skipped')

    display = data.get('display', {})
    if not isinstance(display, dict):
        cfg.warnings.append('[display] is not a table; ignored')
        display = {}
    if 'time' in display:
        try:
            cfg.time_zone = timefmt.parse(display['time'])
        except ValueError as exc:
            cfg.warnings.append(f'[display] time: {exc}; using Z')
    for key, attr in (('colours', 'map_colours'), ('colors', 'map_colours'),
                      ('difference_colours', 'difference_colours'),
                      ('difference_colors', 'difference_colours')):
        if key in display:
            text = str(display[key] or '').strip()
            if text:
                setattr(cfg, attr, text)
    custom = display.get('custom_colours', display.get('custom_colors'))
    if custom is not None:
        if isinstance(custom, str):
            custom = custom.split(',')
        if not isinstance(custom, list):
            cfg.warnings.append('[display] custom_colours must be a list of colours; '
                                'ignored')
        else:
            stops = tuple(str(c).strip() for c in custom if str(c).strip())
            if len(stops) == 1:
                cfg.warnings.append('[display] custom_colours needs at least two '
                                    'colours; ignored')
            else:
                cfg.custom_colours = stops

    tables = data.get('fields', {})
    if not isinstance(tables, dict):
        cfg.warnings.append('[fields] is not a table; ignored')
        tables = {}
    for name, table in tables.items():
        defaults, problems = parse_field(name, table)
        cfg.warnings.extend(problems)
        if field_key(name) in cfg.fields:
            cfg.warnings.append(f'[fields.{name}] repeats a field already given '
                                f'(names ignore case); the later one wins')
        cfg.fields[field_key(name)] = defaults
    return cfg


def loads(text, path=None):
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        cfg = Config(path=path, unreadable=True)
        cfg.warnings.append(f'{Path(path).name if path else "config"} is not valid TOML '
                            f'({exc}); nothing in it was applied')
        return cfg
    return from_dict(data, path)


def load(path=None):
    """The configuration, or an empty one. Never raises -- see the module docstring."""
    path = Path(path) if path else config_path()
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return Config(path=path)
    except (OSError, UnicodeDecodeError) as exc:
        cfg = Config(path=path, unreadable=True)
        cfg.warnings.append(f'could not read {path}: {exc}')
        return cfg
    return loads(text, path)


# ---- writing -----------------------------------------------------------------------------
HEADER = """\
# IMS ICON Viewer -- settings
#
# Edit this file by hand or with Settings... in the app (Ctrl+, / Cmd+,). Everything here
# is a DEFAULT: the toolbar still changes all of it, and a choice made there lasts for the
# rest of that session. Anything left out uses the app's own default. A mistake in one
# line is reported in the app and skips that line only.

[credentials]
# The IMS server account (from the IMS product PDF). Kept in plain text in this file,
# which only your user account can read. Leave both empty to use IMS_USER / IMS_PASS
# from the environment, or the system keychain.
"""

DISPLAY_HELP = """
[display]
# The clock every time on screen is written in: "Z" (UTC, as the model files are) or
# "Israel" (IDT in summer, IST in winter -- decided per time step, so a forecast that
# crosses the change shows it). The "Time" combo on the toolbar changes it for a session.
#
# The colour scale every map opens with, instead of the usual one (turbo on an ordinary
# map, CET-D1A on a difference map). A [fields.<NAME>] colours line still wins for its
# own map, and the "Colours" combo still changes it on screen. Leave empty for the usual.
#   colours            = "viridis"   turbo viridis inferno plasma magma CET-L17, or custom
#   difference_colours = "CET-D9"    CET-D1A CET-D9 CET-D3, or custom
# A scale of your own, called "custom" in the Colours combo: two or more colours from the
# bottom of the scale to the top, as names (white, gold, red...) or #rrggbb.
#   custom_colours = ["white", "gold", "orange", "red", "purple"]
"""

MAP_HELP = """
[map]
# Points drawn as dots on the map, toggled with "Points" on the toolbar.
# Each is "latitude, longitude, colour[, name]". An empty colour is blue; a colour is any
# common name (red, orange, black...) or #rrggbb.
#   points = [
#     "31.7683, 35.2137, red, Jerusalem",
#     "32.0853, 34.7818, , Tel Aviv",
#   ]
"""

FIELDS_HELP = """
# ---- per-map defaults ----------------------------------------------------------------
# One [fields.<NAME>] table per map. Names ignore case, so [fields.T_2M] also covers the
# deterministic run's t_2m. Derived maps go by their names on screen: TD_2M, T-Td,
# WSPD_10M (10 m wind) and WSPD (wind on a pressure level).
#
#   units        = "°C"          a choice from the Units combo ("C" and "F" also work)
#   colours      = "viridis"     turbo viridis inferno plasma magma CET-L17
#                                (diverging: CET-D1A CET-D9 CET-D3), or custom
#   scale        = [0, 3000]     a fixed colour scale, in `units` (or `scale_units`);
#                                or "dataset" / "frame" to pick that Scale entry
#   isolines     = true          contour lines on or off when the map opens
#   isoline_step = 2             their spacing: degrees C on a temperature, ft on a
#                                geopotential height, the file's units on anything else
#                                (setting it on CAPE, say, makes CAPE contourable)
#   profile      = true          pressure-level maps: vertical profile instead of the
#                                time graph
#
# Example:
#   [fields.CAPE_ML]
#   colours = "inferno"
#   scale = [0, 3000]
#   isoline_step = 500
#   isolines = false
"""


def _toml_string(text):
    # JSON's string escapes are a subset of TOML's basic-string escapes.
    return json.dumps(str(text), ensure_ascii=False)


def _toml_key(key):
    return key if re.fullmatch(r'[A-Za-z0-9_-]+', key) else _toml_string(key)


def _toml_number(value):
    value = float(value)
    return str(int(value)) if value.is_integer() and abs(value) < 1e15 else repr(value)


def dumps(cfg):
    """Config -> TOML text, with the explanatory comments that make it hand-editable."""
    out = [HEADER,
           f'user = {_toml_string(cfg.user)}',
           f'password = {_toml_string(cfg.password)}',
           DISPLAY_HELP,
           f'time = {_toml_string("Israel" if cfg.time_zone == timefmt.ISRAEL else "Z")}',
           f'colours = {_toml_string(cfg.map_colours or "")}',
           f'difference_colours = {_toml_string(cfg.difference_colours or "")}',
           'custom_colours = [' + ', '.join(_toml_string(c) for c in cfg.custom_colours)
           + ']',
           MAP_HELP,
           f'show_points = {"true" if cfg.show_points else "false"}']
    if cfg.points:
        out.append('points = [')
        out.extend(f'  {_toml_string(p.as_line())},' for p in cfg.points)
        out.append(']')
    else:
        out.append('points = []')
    out.append(FIELDS_HELP)
    for defaults in cfg.fields.values():
        if defaults.is_empty():
            continue
        out.append(f'[fields.{_toml_key(defaults.name)}]')
        if defaults.units:
            out.append(f'units = {_toml_string(defaults.units)}')
        if defaults.colours:
            out.append(f'colours = {_toml_string(defaults.colours)}')
        if isinstance(defaults.scale, tuple):
            lo, hi = defaults.scale
            out.append(f'scale = [{_toml_number(lo)}, {_toml_number(hi)}]')
        elif defaults.scale:
            out.append(f'scale = {_toml_string(defaults.scale)}')
        if defaults.scale_units:
            out.append(f'scale_units = {_toml_string(defaults.scale_units)}')
        if defaults.isolines is not None:
            out.append(f'isolines = {"true" if defaults.isolines else "false"}')
        if defaults.isoline_step is not None:
            out.append(f'isoline_step = {_toml_number(defaults.isoline_step)}')
        if defaults.profile is not None:
            out.append(f'profile = {"true" if defaults.profile else "false"}')
        out.append('')
    return '\n'.join(out).rstrip() + '\n'


def save(cfg, path=None):
    """Write atomically, readable by the owner only. -> the path written.

    A file that could not be parsed is kept as `<name>.bak` first: the dialog cannot show
    what it could not read, so saving over it would silently discard someone's edits.
    """
    path = Path(path or cfg.path or config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.unreadable and path.exists():
        backup = path.with_name(path.name + '.bak')
        backup.write_bytes(path.read_bytes())
    tmp = path.with_name(path.name + '.tmp')
    text = dumps(cfg)
    # Created 0600 rather than chmod-ed after: there is no moment at which the password
    # sits in a world-readable file. (os.open's mode is ignored on Windows, where the
    # per-user %APPDATA% is what keeps it private.)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
    os.replace(tmp, path)
    if os.name == 'posix':
        os.chmod(path, 0o600)
    return path


def with_fields(cfg, entries):
    """A copy of `cfg` whose per-field table is `entries` (a list of FieldDefaults)."""
    table = {}
    for entry in entries:
        if entry.name and not entry.is_empty():
            table[field_key(entry.name)] = entry
    return replace(cfg, fields=table, warnings=[], unreadable=cfg.unreadable)
