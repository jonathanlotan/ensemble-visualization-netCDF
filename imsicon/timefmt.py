"""How a time is written on screen: Zulu, or Israel local time (R11).

Every time in a file is UTC (`nc3.valid_times` attaches `timezone.utc`), and that stays the
truth underneath: the slider, the time index, the forecast hour and the run id never change
with this setting. Only the *text* does, and it all goes through `stamp` so the map title,
the time bar, the readout, the graph axis and the status bar cannot disagree about which
clock they are reading.

Israel is not one fixed offset. It is UTC+3 (IDT) in summer and UTC+2 (IST) in winter, and
a 120 h forecast issued in the last week of October crosses the change -- so the offset is
decided per timestamp, never once per file, and the suffix names which of the two applies
(`17:00 IDT`, then `16:00 IST` an hour of model time later is correct, not a bug).

`zoneinfo` gives the official rules. A Windows Python has no system time-zone database, so
without the `tzdata` package `ZoneInfo('Asia/Jerusalem')` fails; the fallback is the rule in
force since 2013 (Israel Standard Time Law amendment): IDT from the Friday before the last
Sunday of March at 02:00 local, to the last Sunday of October at 02:00 local.
"""
import datetime as dt

ZULU = 'Z'
ISRAEL = 'IL'
ZONES = (ZULU, ISRAEL)
# What the toolbar and the settings file call them.
LABELS = {ZULU: 'Zulu (UTC)', ISRAEL: 'Israel (IDT/IST)'}
ALIASES = {'z': ZULU, 'zulu': ZULU, 'utc': ZULU, 'gmt': ZULU,
           'il': ISRAEL, 'israel': ISRAEL, 'idt': ISRAEL, 'ist': ISRAEL, 'local': ISRAEL,
           'asia/jerusalem': ISRAEL}

_IST = dt.timezone(dt.timedelta(hours=2), 'IST')
_IDT = dt.timezone(dt.timedelta(hours=3), 'IDT')

_current = ZULU


def _load_israel():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo('Asia/Jerusalem')
    except Exception:           # ZoneInfoNotFoundError, or no zoneinfo at all
        return None


_ISRAEL_TZ = _load_israel()


def parse(text):
    """'Z' / 'zulu' / 'UTC' / 'IL' / 'Israel' / 'IDT' ... -> ZULU | ISRAEL. Raises ValueError."""
    key = ALIASES.get(str(text).strip().lower())
    if key is None:
        raise ValueError(f'{text!r} is not a time zone this app shows; use "Z" or "Israel"')
    return key


def zone():
    return _current


def set_zone(name):
    """Choose the clock every label is written in. Accepts anything `parse` does."""
    global _current
    _current = parse(name)
    return _current


def _last_sunday(year, month):
    day = dt.date(year, month, 31 if month in (3, 10) else 30)
    return day - dt.timedelta(days=(day.weekday() + 1) % 7)


def _israel_fallback(when):
    """The post-2013 rule, for a Python without a time-zone database."""
    utc = when.astimezone(dt.timezone.utc)
    year = utc.year
    # Friday before the last Sunday of March, 02:00 IST = 00:00 UTC.
    start = dt.datetime.combine(_last_sunday(year, 3) - dt.timedelta(days=2),
                                dt.time(0, 0), dt.timezone.utc)
    # Last Sunday of October, 02:00 IDT = 23:00 UTC the Saturday before.
    end = dt.datetime.combine(_last_sunday(year, 10), dt.time(0, 0),
                              dt.timezone.utc) - dt.timedelta(hours=1)
    return utc.astimezone(_IDT if start <= utc < end else _IST)


def localise(when, zone_name=None):
    """A UTC datetime -> the same instant in the chosen clock."""
    zone_name = _current if zone_name is None else parse(zone_name)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)     # every file time is UTC (G4)
    if zone_name == ZULU:
        return when.astimezone(dt.timezone.utc)
    if _ISRAEL_TZ is not None:
        return when.astimezone(_ISRAEL_TZ)
    return _israel_fallback(when)


def stamp(when, date=True, zone_name=None):
    """'2026-08-27 14:00Z' or '2026-08-27 17:00 IDT'; `date=False` gives the clock only."""
    zone_name = _current if zone_name is None else parse(zone_name)
    local = localise(when, zone_name)
    if zone_name == ZULU:
        text = f'{local:%H:%M}Z'
    else:
        text = f'{local:%H:%M} {local.tzname()}'
    return f'{local:%Y-%m-%d} {text}' if date else text
