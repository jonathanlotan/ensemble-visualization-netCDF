"""V2.2: the affine layer, the normaliser and the registry guard.

Property-based invariants (round-trip, totality) are preferred over golden numbers: they
keep holding when IMS changes a value, and they are what catches the G15/G21 class of bug.
"""
import numpy as np
import pytest

from imsicon import transform
from imsicon.transform import Affine, normalise_units


# ---- the affine itself ----------------------------------------------------------------
@pytest.mark.parametrize('affine', [
    Affine('°C', 1.0, -273.15), Affine('°F', 1.8, -459.67),
    Affine('kt', 3600.0 / 1852.0), Affine('cm', 100.0), Affine('m'),
])
def test_invert_round_trips_apply(affine):
    x = np.array([0.0, 1.0, 273.15, 300.0, -12.5, 1e4])
    assert np.allclose(affine.invert(affine.apply(x)), x, rtol=0, atol=1e-9)


def test_apply_delta_ignores_the_offset():
    """G15: a 5 K rise is a 5 degC rise, not -268.15 degC."""
    celsius = Affine('°C', 1.0, -273.15)
    assert celsius.apply_delta(5.0) == 5.0
    assert celsius.apply(5.0) == pytest.approx(-268.15)
    fahrenheit = Affine('°F', 1.8, -459.67)
    assert fahrenheit.apply_delta(5.0) == pytest.approx(9.0)     # scale still applies
    assert fahrenheit.apply(5.0) == pytest.approx(-450.67)


def test_identity_is_a_no_op_not_a_copy():
    x = np.arange(5.0)
    assert Affine('m s-1').apply(x) is x
    assert Affine('m s-1').apply_delta(x) is x


def test_apply_range_swaps_the_ends_for_a_negative_scale():
    """G19: transforming a cached range must not produce min > max."""
    assert Affine('x', 2.0, 1.0).apply_range(0.0, 10.0) == (1.0, 21.0)
    assert Affine('x', -1.0, 0.0).apply_range(0.0, 10.0) == (-10.0, 0.0)


def test_kelvin_celsius_kelvin_is_stable_on_a_real_shaped_array():
    rng = np.random.default_rng(3)
    kelvin = rng.uniform(250.0, 320.0, size=(121, 20)).astype(np.float32)
    celsius = Affine('°C', 1.0, -273.15)
    back = celsius.invert(celsius.apply(kelvin.astype(np.float64)))
    assert np.allclose(back, kelvin, atol=1e-6)


# ---- G21: the normaliser must be TOTAL ------------------------------------------------
@pytest.mark.parametrize('spelling,canonical', [
    ('K', 'K'), ('k', 'K'), ('kelvin', 'K'), ('degK', 'K'), ('deg_K', 'K'),
    ('J kg-1', 'J kg-1'), ('j/kg', 'J kg-1'), ('J kg**-1', 'J kg-1'), ('J kg^-1', 'J kg-1'),
    ('m s-1', 'm s-1'), ('m/s', 'm s-1'), ('ms-1', 'm s-1'), ('m s**-1', 'm s-1'),
    ('kg m-2', 'kg m-2'), ('kg/m2', 'kg m-2'), ('kg m**-2', 'kg m-2'),
    ('W m-2', 'W m-2'), ('w/m2', 'W m-2'),
    ('%', '%'), ('percent', '%'),
    ('1', '1'), ('-', '1'), ('', '1'),
    ('  M  ', 'm'), ('MM', 'mm'),
])
def test_normaliser_maps_every_documented_alias(spelling, canonical):
    assert normalise_units(spelling) == canonical


@pytest.mark.parametrize('junk', [None, 42, b'bytes', object(), np.float64(3.5),
                                  'completely made up', '\x00\x01', 'kg m-2 kg m-2'])
def test_normaliser_never_raises(junk):
    """A units string must not be able to crash a file open."""
    out = normalise_units(junk)
    assert isinstance(out, str)


def test_an_unrecognised_string_comes_back_unaliased():
    assert normalise_units('furlongs per fortnight') == 'furlongs per fortnight'


# ---- the registry guard (v2 1.3) ------------------------------------------------------
def test_temperature_defaults_to_celsius():
    choices, note = transform.choices_for('T_2M', 'K')
    assert note is None
    assert [c.label for c in choices] == ['°C', 'K', '°F']
    assert choices[0].b == pytest.approx(-273.15)


def test_precipitation_defaults_to_mm_and_the_number_does_not_change():
    choices, _ = transform.choices_for('TOT_PREC', 'kg m-2')
    assert choices[0].label == 'mm'
    assert choices[0].is_identity          # 1 kg m-2 of water IS 1 mm, exactly


def test_knots_are_exact_by_the_definition_of_the_nautical_mile():
    choices, _ = transform.choices_for('U_10M', 'm/s')
    knots = {c.label: c for c in choices}['kt']
    assert knots.a == pytest.approx(3600.0 / 1852.0)
    assert knots.a == pytest.approx(1.9438445, abs=1e-7)


def test_vmax_stays_in_file_units_by_default():
    choices, _ = transform.choices_for('VMAX_10M', 'm s-1')
    assert choices[0].label == 'm s-1'


def test_fields_with_no_sensible_conversion_offer_only_their_own_units():
    for field, units in (('CAPE_ML', 'J kg-1'), ('RELHUM_2M', '%'),
                         ('ASWDIR_S', 'W m-2')):
        choices, note = transform.choices_for(field, units)
        assert note is None
        assert [c.label for c in choices] == [units]
        assert choices[0].is_identity


def test_an_unrecognised_units_string_is_shown_verbatim():
    """Only a RECOGNISED string is prettified; an unknown one is repeated as the file
    wrote it, because we have no grounds to reinterpret it."""
    choices, _ = transform.choices_for('SOMETHING_NEW', 'furlongs/fortnight')
    assert choices[0].label == 'furlongs/fortnight'


def test_a_units_mismatch_disables_conversion_rather_than_guessing():
    """The registry may be stale; converting anyway would be a silent physical error."""
    choices, note = transform.choices_for('T_2M', 'J kg-1')
    assert [c.label for c in choices] == ['J kg-1']
    assert choices[0].is_identity
    assert note and 'stale' in note


def test_an_unknown_field_still_opens_with_no_conversion():
    choices, note = transform.choices_for('SOMETHING_NEW', 'widgets')
    assert [c.label for c in choices] == ['widgets']
    assert note is None


def test_snow_defaults_to_centimetres():
    choices, _ = transform.choices_for('H_SNOW', 'm')
    assert choices[0].label == 'cm' and choices[0].a == 100.0


# ---- measured against the live IMS server, run 2026082400, on 2026-08-24 ---------------
MEASURED = {
    'CAPE_ML': 'J kg-1', 'T_2M': 'K', 'T_S': 'K', 'RELHUM_2M': '%',
    'TOT_PREC': 'kg m-2', 'U_10M': 'm s-1', 'V_10M': 'm s-1', 'VMAX_10M': 'm s-1',
    'CLCT': '%', 'CLCL': '%', 'CLCM': '%', 'CLCH': '%',
    'ASWDIFD_S': 'W/m**2', 'ASWDIR_S': 'W/m**2', 'H_SNOW': 'm',
}


@pytest.mark.parametrize('field,units', sorted(MEASURED.items()))
def test_every_real_units_string_is_accepted_by_the_registry(field, units):
    """The whole point of V2.1: no row of the registry is a guess any more.

    If IMS changes a spelling, this fails here rather than silently disabling conversion
    for a forecaster.
    """
    entry = transform.UNITS[field]
    assert normalise_units(units) in entry.expected, (
        f'{field}: server says {units!r} -> {normalise_units(units)!r}, '
        f'registry expects {entry.expected}')


def test_the_radiation_spelling_only_works_because_of_the_normaliser():
    """`W/m**2` is what the server actually sends; `W m-2` is what the registry holds."""
    assert normalise_units('W/m**2') == 'W m-2'
    choices, note = transform.choices_for('ASWDIR_S', 'W/m**2')
    assert note is None
    assert [c.label for c in choices] == ['W m-2']      # canonical, not the raw spelling
    assert choices[0].is_identity                        # the NUMBER is untouched
