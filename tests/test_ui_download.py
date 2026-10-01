"""The download dialog's map search: find a map by typing instead of scrolling."""
import pytest
from PySide6 import QtCore
from PySide6.QtTest import QTest

from imsicon import ingest
from imsicon.ui import downloaddialog

CHECKED = QtCore.Qt.CheckState.Checked
FIELDS = ('CAPE_ML', 'TOT_PREC', 'T_2M', 'RELHUM_2M', 'U_10M', 'V_10M', 'VMAX_10M')


@pytest.fixture(autouse=True)
def no_real_environment(monkeypatch, tmp_path):
    # G30: the dialog prefills from the keychain, and lists what is on disk.
    monkeypatch.setattr(downloaddialog.download, 'stored_credentials', lambda: None)
    monkeypatch.setattr(downloaddialog.download, 'keyring_module', lambda: None)
    monkeypatch.setattr(ingest, 'search_roots', lambda near=None: [tmp_path])


@pytest.fixture
def dialog(qapp):
    d = downloaddialog.DownloadDialog(None)
    d._on_listed((object(), downloaddialog.download.parse_listing(
        ''.join(f'<a href="ICON_ENS_2026082300_{f}.nc.bz2">'
                f'ICON_ENS_2026082300_{f}.nc.bz2</a>' for f in FIELDS))))
    yield d
    d.close()


def shown(d):
    return [item.text(1) for item in d.visible_items()]


def search(d, text):
    d.search_edit.setText(text)
    return shown(d)


def test_an_empty_search_shows_every_map(dialog):
    assert shown(dialog) == list(FIELDS)
    assert dialog.match_label.text() == ''


def test_a_search_matches_the_name_on_the_menu_and_ignores_case(dialog):
    assert search(dialog, 'PRECIP') == ['TOT_PREC']
    assert search(dialog, 'cape') == ['CAPE_ML']
    assert dialog.match_label.text() == f'1 of {len(FIELDS)} maps'


def test_a_search_matches_the_field_name_with_or_without_underscores(dialog):
    assert search(dialog, 'tot_prec') == ['TOT_PREC']
    assert search(dialog, 'tot prec') == ['TOT_PREC']
    assert search(dialog, 'u10m') == ['U_10M']


def test_every_word_must_match(dialog):
    assert set(search(dialog, '10m')) == {'U_10M', 'V_10M', 'VMAX_10M'}
    assert search(dialog, '10m vmax') == ['VMAX_10M']
    assert search(dialog, 'nothing like this') == []
    assert dialog.match_label.text() == 'no map matches'


def test_a_ticked_map_stays_ticked_and_downloads_while_it_is_hidden(dialog):
    dialog._select_dew_point()
    search(dialog, 'cape')
    assert {e.field for e in dialog.selected()} == {'T_2M', 'RELHUM_2M'}
    assert '2 ticked maps hidden' in dialog.match_label.text()
    search(dialog, '')
    assert shown(dialog) == list(FIELDS)


def test_enter_ticks_the_only_match(dialog):
    search(dialog, 'precip')
    QTest.keyClick(dialog.search_edit, QtCore.Qt.Key.Key_Return)
    assert [e.field for e in dialog.selected()] == ['TOT_PREC']
    search(dialog, '10m')                              # three left: Enter does nothing
    QTest.keyClick(dialog.search_edit, QtCore.Qt.Key.Key_Return)
    assert [e.field for e in dialog.selected()] == ['TOT_PREC']


def test_the_search_survives_choosing_another_run(dialog):
    search(dialog, 'cape')
    dialog._populate_fields()                          # what a run or product change does
    assert shown(dialog) == ['CAPE_ML']


def test_enter_in_the_search_never_presses_the_dialog_s_buttons(dialog, monkeypatch):
    # Connect is the dialog's default button, so an Enter that leaked would press it.
    pressed = []
    monkeypatch.setattr(dialog, 'connect_to_server', lambda: None)
    dialog.connect_button.clicked.connect(lambda: pressed.append('connect'))
    dialog.download_button.clicked.connect(lambda: pressed.append('download'))
    dialog.show()
    dialog.search_edit.setFocus()
    search(dialog, 'precip')
    QTest.keyClick(dialog.search_edit, QtCore.Qt.Key.Key_Return)
    QTest.keyClick(dialog.search_edit, QtCore.Qt.Key.Key_Enter)
    assert pressed == []
