"""The configuration file itself (R10): reading, writing, and never failing to open.

No Qt here, like `config.py`: the window and the dialog are tested in
`test_ui_settings.py`, and what is tested here is the contract between a person typing
into a text file and the app reading it.
"""
import os
import tomllib

import pytest

from imsicon import config, download, isolines


def write(path, text):
    path.write_text(text, encoding='utf-8')
    return path


# ---- reading ------------------------------------------------------------------------------
def test_a_missing_file_is_an_empty_configuration_not_an_error(tmp_path):
    cfg = config.load(tmp_path / 'nope.toml')
    assert cfg.points == [] and cfg.fields == {} and cfg.warnings == []
    assert cfg.credentials() is None and cfg.show_points


def test_the_location_follows_the_environment_override(isolated_config):
    assert config.config_path() == isolated_config


def test_points_are_read_in_the_form_they_were_asked_for(tmp_path):
    cfg = config.load(write(tmp_path / 'c.toml', '''
[map]
points = [
  "31.1234, 32.1234, blue",
  "31.7683, 35.2137, red, Jerusalem",
  "32.0853, 34.7818, , Tel Aviv",
  "30.5, 34.9",
  { lat = 29.55, lon = 34.95, colour = "#ff8800", name = "Eilat" },
]
'''))
    assert cfg.warnings == []
    first, jerusalem, tel_aviv, bare, eilat = cfg.points
    assert (first.lat, first.lon, first.colour) == (31.1234, 32.1234, 'blue')
    assert (jerusalem.colour, jerusalem.name) == ('red', 'Jerusalem')
    # "If color is empty, default to blue" -- an empty field, and a missing one.
    assert tel_aviv.colour == 'blue' and tel_aviv.name == 'Tel Aviv'
    assert bare.colour == 'blue' and bare.name == ''
    assert (eilat.colour, eilat.name) == ('#ff8800', 'Eilat')


def test_a_name_may_contain_a_comma():
    point = config.parse_point('31.0, 35.0, green, Haifa, port')
    assert point.name == 'Haifa, port'


@pytest.mark.parametrize('line, why', [
    ('31.0', 'at least'),
    ('north, 35', 'numbers'),
    ('95, 35', 'latitude'),
    ('31, 200, red', 'longitude'),
])
def test_a_bad_point_is_skipped_with_a_reason_and_the_rest_survive(tmp_path, line, why):
    cfg = config.load(write(tmp_path / 'c.toml',
                            f'[map]\npoints = ["{line}", "31.5, 35.5, red"]\n'))
    assert len(cfg.points) == 1 and cfg.points[0].colour == 'red'
    assert len(cfg.warnings) == 1 and why in cfg.warnings[0]


def test_field_defaults_are_read_and_folded_like_the_registry(tmp_path):
    cfg = config.load(write(tmp_path / 'c.toml', '''
[fields.t_2m]
units = "F"
colours = "viridis"
scale = [10, 40]
scale_units = "C"
isolines = false
isoline_step = 2

[fields.CAPE_ML]
scale = "frame"

[fields.rh]
profile = "off"

[fields.T-Td]
colors = "inferno"
'''))
    assert cfg.warnings == []
    t2m = cfg.for_names('T_2M')           # the ensemble's spelling finds t_2m's table
    assert t2m.units == '°F' and t2m.scale_units == '°C'      # "F" and "C" are aliases
    assert t2m.colours == 'viridis' and t2m.fixed_range == (10.0, 40.0)
    assert t2m.isolines is False and t2m.isoline_step == 2.0
    assert cfg.for_names('cape_ml').scale == 'frame'
    assert cfg.for_names('RH').profile is False
    assert cfg.for_names('T-Td').colours == 'inferno'          # 'colors' is accepted
    assert cfg.for_names('U_10M') is config.EMPTY


def test_a_derived_view_is_found_by_the_name_on_screen():
    cfg = config.from_dict({'fields': {'T-Td': {'colours': 'magma'}}})

    class Depression:
        field = 'T_2M-TD_2M'
        display_name = 'T-Td'
    assert cfg.for_view(Depression()).colours == 'magma'


@pytest.mark.parametrize('entry, why', [
    ('scale = [40, 10]', 'above min'),
    ('scale = [1, 2, 3]', 'two numbers'),
    ('scale = "biggest"', 'dataset'),
    ('isoline_step = -1', 'above zero'),
    ('isoline_step = "close"', 'number'),
    ('isolines = "maybe"', 'true or false'),
    ('shading = true', 'not a setting'),
])
def test_one_bad_setting_costs_that_setting_only(tmp_path, entry, why):
    cfg = config.load(write(tmp_path / 'c.toml',
                            f'[fields.T_2M]\ncolours = "plasma"\n{entry}\n'))
    assert cfg.for_names('T_2M').colours == 'plasma'
    assert len(cfg.warnings) == 1 and why in cfg.warnings[0], cfg.warnings


def test_a_file_that_is_not_toml_opens_the_app_anyway(tmp_path):
    cfg = config.load(write(tmp_path / 'c.toml', '[map\npoints = ['))
    assert cfg.unreadable and cfg.points == []
    assert 'not valid TOML' in cfg.warnings[0]


# ---- writing ------------------------------------------------------------------------------
def full_config(path):
    cfg = config.Config(user='forecaster', password='p"ss\\word', show_points=False,
                        path=path)
    cfg.points = [config.Point(31.1234, 32.1234), config.Point(31.77, 35.21, 'red',
                                                               'Jerusalem')]
    cfg.fields = {
        'T_2M': config.FieldDefaults('T_2M', units='°C', colours='viridis',
                                     scale=(-5.0, 45.0), isolines=True, isoline_step=2.0),
        'T-TD': config.FieldDefaults('T-Td', scale='frame'),
        'RH': config.FieldDefaults('rh', profile=False),
        'CAPE_ML': config.FieldDefaults('CAPE_ML', scale=(0.0, 3000.0),
                                        isoline_step=500.0),
    }
    return cfg


def test_what_is_saved_is_what_is_read_back(tmp_path):
    path = tmp_path / 'deep' / 'config.toml'
    config.save(full_config(path), path)
    back = config.load(path)
    assert back.warnings == []
    assert back.credentials() == ('forecaster', 'p"ss\\word')
    assert back.show_points is False
    assert back.points == full_config(path).points
    for key, expected in full_config(path).fields.items():
        assert back.fields[key] == expected


def test_the_written_file_explains_itself(tmp_path):
    path = config.save(full_config(tmp_path / 'c.toml'))
    text = path.read_text(encoding='utf-8')
    tomllib.loads(text)                       # valid TOML, comments and all
    for phrase in ('latitude, longitude, colour', 'An empty colour is blue',
                   'isoline_step', 'scale', 'IMS_USER'):
        assert phrase in text
    assert '"31.1234, 32.1234, blue",' in text      # one point per line, as typed


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permissions')
def test_the_password_file_is_readable_by_its_owner_only(tmp_path):
    path = config.save(full_config(tmp_path / 'c.toml'))
    assert (path.stat().st_mode & 0o777) == 0o600


def test_saving_over_an_unreadable_file_keeps_a_copy_of_it(tmp_path):
    path = write(tmp_path / 'c.toml', 'this is [not toml')
    cfg = config.load(path)
    config.save(cfg, path)
    assert (tmp_path / 'c.toml.bak').read_text() == 'this is [not toml'
    assert config.load(path).warnings == []


# ---- the two consumers outside the window ---------------------------------------------------
def test_credentials_come_from_the_environment_then_the_file(isolated_config, monkeypatch):
    monkeypatch.delenv('IMS_USER', raising=False)
    monkeypatch.delenv('IMS_PASS', raising=False)
    monkeypatch.setattr(download, 'keyring_module', lambda: None)
    assert download.stored_credentials() is None
    config.save(config.Config(user='file-user', password='file-pass'), isolated_config)
    assert download.stored_credentials() == ('file-user', 'file-pass')
    monkeypatch.setenv('IMS_USER', 'env-user')
    monkeypatch.setenv('IMS_PASS', 'env-pass')
    assert download.stored_credentials() == ('env-user', 'env-pass')


def test_a_half_filled_account_is_not_used(isolated_config, monkeypatch):
    monkeypatch.delenv('IMS_USER', raising=False)
    monkeypatch.delenv('IMS_PASS', raising=False)
    monkeypatch.setattr(download, 'keyring_module', lambda: None)
    config.save(config.Config(user='only-a-user'), isolated_config)
    assert download.stored_credentials() is None


def test_a_spacing_makes_an_uncontoured_field_contourable():
    assert isolines.interval_for('CAPE_ML') is None
    isolines.set_custom({'CAPE_ML': 500.0, 'T_2M': 3.0}, {'CAPE_ML': 'J kg-1'})
    interval = isolines.interval_for('cape_ml')
    assert interval.step == 500.0 and interval.anchor == 0.0
    assert interval.ladder.steps == (250.0, 500.0, 1000.0, 2000.0)
    assert interval.ladder.unit == 'J kg-1'
    # A field the app already contours keeps its own interval: the configured step is a
    # default spacing on its ladder, applied by the window, not a new interval.
    assert isolines.interval_for('T_2M') is isolines.STEPS['T_2M']
    isolines.set_custom({})
    assert isolines.interval_for('CAPE_ML') is None


# ---- R12: the map's colour scale --------------------------------------------------------
def test_the_display_colour_scale_is_read_and_round_trips(tmp_path):
    cfg = config.loads('[display]\ncolours = "viridis"\ndifference_colours = "CET-D9"\n'
                       'custom_colours = ["white", "#ffcc00", "red"]\n')
    assert cfg.warnings == []
    assert (cfg.map_colours, cfg.difference_colours) == ('viridis', 'CET-D9')
    assert cfg.custom_colours == ('white', '#ffcc00', 'red')
    path = config.save(cfg, tmp_path / 'c.toml')
    again = config.load(path)
    assert (again.map_colours, again.difference_colours, again.custom_colours) == \
        (cfg.map_colours, cfg.difference_colours, cfg.custom_colours)


def test_an_empty_colour_scale_means_the_usual_one(tmp_path):
    cfg = config.loads('[display]\ncolours = ""\ncustom_colours = []\n')
    assert cfg.map_colours is None and cfg.custom_colours == () and cfg.warnings == []
    again = config.load(config.save(config.Config(), tmp_path / 'c.toml'))
    assert again.map_colours is None and again.custom_colours == () and not again.warnings


def test_a_custom_scale_can_be_one_comma_separated_line():
    cfg = config.loads('[display]\ncustom_colours = "white, gold , red"\n')
    assert cfg.custom_colours == ('white', 'gold', 'red')


def test_a_one_colour_custom_scale_is_a_warning_not_a_scale():
    cfg = config.loads('[display]\ncustom_colours = ["red"]\n')
    assert cfg.custom_colours == () and any('two' in w for w in cfg.warnings)
    cfg = config.loads('[display]\ncustom_colours = 3\n')
    assert cfg.custom_colours == () and cfg.warnings


def test_the_custom_ramp_skips_what_is_not_a_colour():
    from imsicon.ui import colors
    assert colors.set_custom_ramp(['white', 'nonsense', 'red']) == ['nonsense']
    assert colors.custom_stops() == ('white', 'red')
    assert colors.ramp_named('CUSTOM') == 'custom' and colors.ramps()[-1] == 'custom'
    assert colors.set_custom_ramp(['red']) == [] and colors.custom_stops() == ()
    assert colors.ramp_named('custom') is None
