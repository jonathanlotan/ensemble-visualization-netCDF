import os
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

DATA = Path(__file__).resolve().parent.parent / 'data'
CAPE = DATA / 'ICON_ENS_2026082300_CAPE_ML.nc'


@pytest.fixture(scope='session')
def cape_path():
    if not CAPE.exists():
        pytest.skip(f'reference file missing: {CAPE}')
    return CAPE


@pytest.fixture(scope='session', autouse=True)
def isolated_settings(tmp_path_factory):
    """Keep QSettings out of the developer's real preferences.

    v2 persists a units choice per field, so without this a test run would silently
    change what the app shows on the next real launch -- and read back whatever a previous
    run happened to leave behind.
    """
    from PySide6 import QtCore
    root = tmp_path_factory.mktemp('settings')
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat,
                             QtCore.QSettings.Scope.UserScope, str(root))
    yield root


@pytest.fixture(autouse=True)
def clean_settings(isolated_settings):
    """Each test starts from a fresh preferences file, so a units choice saved by one
    test cannot decide what the next test's window opens in."""
    from PySide6 import QtCore
    QtCore.QSettings('IMS', 'IconEnsembleViewer').clear()
    yield


@pytest.fixture(scope='session')
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
