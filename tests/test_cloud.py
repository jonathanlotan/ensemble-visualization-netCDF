"""V2.2 / G22: cloud-cover encoding is undecidable from the units string alone.

CLAUDE.md 0.2 says CLCT/CLCL/CLCM/CLCH are 0-1; ICON commonly publishes them as 0-100.
Offering a x100 against a field already in % is a silent 100x error, so the decision is
made from the DATA RANGE -- and "cannot tell" is a real answer, not a reason to guess.
"""
import numpy as np
import pytest

import synth
from imsicon import transform
from imsicon.dataset import EnsembleFile
from imsicon.fieldview import FieldView


@pytest.mark.parametrize('field', ['CLCT', 'CLCL', 'CLCM', 'CLCH'])
def test_all_four_cloud_fields_are_gated(field):
    assert transform.UNITS[field].gate == 'cloud'


def test_verdicts_from_the_range_alone():
    assert transform.cloud_encoding(np.array([0.0, 0.4, 1.0])) == 'fraction'
    assert transform.cloud_encoding(np.array([0.0, 40.0, 100.0])) == 'percent'
    assert transform.cloud_encoding(np.zeros(50)) == 'undecidable'
    assert transform.cloud_encoding(None) == 'undecidable'
    assert transform.cloud_encoding(np.full(5, np.nan)) == 'undecidable'


def test_a_fraction_file_is_offered_the_x100_and_defaults_to_percent(tmp_path):
    path = synth.cloud(tmp_path / 'frac.nc', encoding='fraction', units='1')
    view = FieldView(EnsembleFile(path))
    assert view.unit_labels == ['%', 'fraction']
    assert view.units == '%'
    assert view.units_note is None
    raw = EnsembleFile(path).ens_frame(0)
    assert np.allclose(view.ens_frame(0), raw * 100.0)
    assert view.ens_frame(0).max() > 1.0


def test_a_percent_file_is_never_multiplied_again(tmp_path):
    """The 100x error this whole test file exists to prevent."""
    path = synth.cloud(tmp_path / 'pct.nc', encoding='percent', units='1')
    view = FieldView(EnsembleFile(path))
    assert view.unit_labels == ['%']
    assert view.units == '%'
    assert view.units_note and 'exceed 1' in view.units_note
    raw = EnsembleFile(path).ens_frame(0)
    assert np.array_equal(view.ens_frame(0), raw)          # untouched
    assert view.ens_frame(0).max() <= 100.0 + 1e-6


def test_a_percent_file_that_says_so_needs_no_warning(tmp_path):
    path = synth.cloud(tmp_path / 'pct2.nc', encoding='percent', units='%')
    view = FieldView(EnsembleFile(path))
    assert view.unit_labels == ['%'] and view.units_note is None


def test_an_all_zero_file_refuses_to_convert(tmp_path):
    """Clear sky: 0 is 0 in both encodings, so nothing can be concluded."""
    path = synth.cloud(tmp_path / 'zero.nc', encoding='zero', units='1')
    view = FieldView(EnsembleFile(path))
    assert not view.can_convert_units
    assert view.units_note and 'cannot be decided' in view.units_note
    assert np.array_equal(view.ens_frame(0), EnsembleFile(path).ens_frame(0))


def test_the_range_check_is_a_permanent_guard_not_a_one_off(tmp_path):
    """v2 1.5: if a future file exceeds 1.0 while the registry says fraction, disable."""
    path = synth.cloud(tmp_path / 'future.nc', encoding='percent', units='1')
    choices, note = transform.choices_for('CLCT', '1', sample=np.array([0.0, 87.0]))
    assert [c.label for c in choices] == ['%']
    assert note is not None
