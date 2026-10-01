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
def clean_settings(isolated_settings, monkeypatch, tmp_path):
    """Each test starts from a fresh preferences file, so a units choice saved by one
    test cannot decide what the next test's window opens in.

    G47: the `setDefaultFormat`/`setPath` redirection above is NOT enough on macOS, where
    `QSettings(org, app)` keeps its native format -- the plist in ~/Library/Preferences --
    whatever the default format says. MEASURED: with only the session fixture in place,
    a marker written natively before the suite was gone after it. So the app opens its
    preferences through one seam, `ui.main.app_settings`, and every test points that seam
    at an `.ini` of its own.
    """
    from PySide6 import QtCore
    from imsicon.ui import main as ui_main
    ini = tmp_path / 'IconEnsembleViewer.ini'

    def test_settings():
        return QtCore.QSettings(str(ini), QtCore.QSettings.Format.IniFormat)

    monkeypatch.setattr(ui_main, 'app_settings', test_settings)
    test_settings().clear()
    yield


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    """R10: the user's settings file is per test as well -- the same lesson as G25/G47,
    applied before it could bite. Without it a developer's real config (their points,
    their CAPE colours, their IMS password) would decide what every UI test sees.
    Custom isoline intervals and the display clock are module state, so they are reset too."""
    from imsicon import isolines, timefmt
    path = tmp_path / 'imsicon-config.toml'
    monkeypatch.setenv('IMSICON_CONFIG', str(path))
    isolines.CUSTOM.clear()
    timefmt.set_zone(timefmt.ZULU)      # R11: the clock is module state too
    yield path
    isolines.CUSTOM.clear()
    timefmt.set_zone(timefmt.ZULU)


@pytest.fixture(scope='session')
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
