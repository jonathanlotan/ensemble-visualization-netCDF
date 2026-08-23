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


@pytest.fixture(scope='session')
def qapp():
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
