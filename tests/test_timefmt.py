"""R11: times written in Zulu or Israel time, and only the text changes."""
import datetime as dt

import numpy as np
import pytest

import synth
from imsicon import config, timefmt
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView
from imsicon.ncwrite import write_nc3

UTC = dt.timezone.utc


def at(*args):
    return dt.datetime(*args, tzinfo=UTC)


def test_zulu_is_the_default_and_reads_as_before():
    assert timefmt.zone() == timefmt.ZULU
    assert timefmt.stamp(at(2026, 8, 27, 14)) == '2026-08-27 14:00Z'
    assert timefmt.stamp(at(2026, 8, 27, 14), date=False) == '14:00Z'


def test_israel_is_idt_in_summer_and_ist_in_winter():
    assert timefmt.stamp(at(2026, 8, 27, 14), zone_name='IL') == '2026-08-27 17:00 IDT'
    assert timefmt.stamp(at(2026, 1, 15, 14), zone_name='IL') == '2026-01-15 16:00 IST'


def test_the_date_follows_the_local_clock_across_midnight():
    assert timefmt.stamp(at(2026, 8, 27, 22), zone_name='IL') == '2026-08-28 01:00 IDT'


def test_the_offset_is_decided_per_time_step_across_the_october_change():
    # Last Sunday of October 2026 is the 25th: 02:00 IDT falls back to 01:00 IST.
    assert timefmt.stamp(at(2026, 10, 24, 22), zone_name='IL') == '2026-10-25 01:00 IDT'
    assert timefmt.stamp(at(2026, 10, 24, 23), zone_name='IL') == '2026-10-25 01:00 IST'
    # ...and in March it springs forward on the Friday before the last Sunday (27th).
    assert timefmt.stamp(at(2026, 3, 26, 23), zone_name='IL') == '2026-03-27 01:00 IST'
    assert timefmt.stamp(at(2026, 3, 27, 0), zone_name='IL') == '2026-03-27 03:00 IDT'


def test_the_built_in_rule_agrees_with_the_time_zone_database():
    """The fallback is what a Windows build without `tzdata` uses; it must not drift."""
    if timefmt._ISRAEL_TZ is None:
        pytest.skip('no time-zone database here to compare against')
    hour = at(2024, 1, 1)
    while hour < at(2028, 1, 1):
        ours = timefmt._israel_fallback(hour)
        theirs = hour.astimezone(timefmt._ISRAEL_TZ)
        assert (ours.utcoffset(), ours.tzname()) == (theirs.utcoffset(), theirs.tzname()), hour
        hour += dt.timedelta(hours=1)


def test_a_naive_time_is_taken_as_utc():
    assert timefmt.stamp(dt.datetime(2026, 8, 27, 14), zone_name='IL').endswith('17:00 IDT')


@pytest.mark.parametrize('text, zone', [('Z', 'Z'), ('zulu', 'Z'), ('UTC', 'Z'),
                                        ('IL', 'IL'), ('Israel', 'IL'), ('IDT', 'IL'),
                                        (' ist ', 'IL')])
def test_names_a_person_would_write(text, zone):
    assert timefmt.parse(text) == zone


def test_an_unknown_zone_is_refused():
    with pytest.raises(ValueError):
        timefmt.parse('EST')
    assert timefmt.zone() == timefmt.ZULU


# ---- the views ---------------------------------------------------------------------------
def test_the_time_readout_follows_the_clock_and_keeps_the_forecast_hour(tmp_path):
    ds = EnsembleFile(synth.temperature(tmp_path / 'ICON_ENS_2026082300_T_2M.nc'))
    assert ds.label_for(2) == '2026-08-23 02:00Z  (+2 h)'
    timefmt.set_zone('Israel')
    assert ds.label_for(2) == '2026-08-23 05:00 IDT  (+2 h)'
    assert 'run 2026-08-23 03:00 IDT' in ds.summary()
    assert ds.times[2] == at(2026, 8, 23, 2)          # the data's own time is untouched


def test_a_rate_window_names_its_end_in_the_same_clock(tmp_path):
    path, _hourly = synth.accumulated_precip(tmp_path / 'ICON_ENS_2026082300_TOT_PREC.nc')
    view = FieldView(EnsembleFile(path))
    view.set_rate(1)
    timefmt.set_zone('IL')
    assert view.label_for(3).endswith('[1 h to 06:00 IDT]')


def test_a_forecast_across_the_change_shows_both_abbreviations(tmp_path):
    hours = np.arange(48)
    data = np.zeros((48, 2, 3, 4), dtype=np.float32)
    path = write_nc3(tmp_path / 'ICON_ENS_2026102400_T_2M.nc', 'T_2M', 'K', data + 290,
                     time_minutes=hours * 60.0,
                     time_units='minutes since 2026-10-24 00:00:00')
    ds = EnsembleFile(path)
    timefmt.set_zone('IL')
    assert ds.label_for(22).startswith('2026-10-25 01:00 IDT')
    assert ds.label_for(23).startswith('2026-10-25 01:00 IST')
    assert '(+23 h)' in ds.label_for(23)


# ---- the settings file ----------------------------------------------------------------
def test_the_settings_file_carries_the_clock(tmp_path):
    cfg = config.loads('[display]\ntime = "Israel"\n')
    assert cfg.time_zone == timefmt.ISRAEL and cfg.warnings == []
    path = config.save(cfg, tmp_path / 'c.toml')
    assert config.load(path).time_zone == timefmt.ISRAEL
    assert config.load(tmp_path / 'missing.toml').time_zone == timefmt.ZULU


def test_a_bad_clock_in_the_file_is_a_warning_not_a_crash():
    cfg = config.loads('[display]\ntime = "Mars"\n')
    assert cfg.time_zone == timefmt.ZULU
    assert any('[display] time' in w for w in cfg.warnings)
