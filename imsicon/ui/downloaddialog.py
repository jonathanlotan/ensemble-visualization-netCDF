"""The "Download from IMS..." dialog: choose a run, choose the maps, fetch them.

Connect once, then pick from the fields the server actually offers -- CAPE, precipitation,
temperature, humidity and the rest -- with sizes from the listing and a note on which are
already on disk. Several fields can be queued in one go, because the dew point needs two
(`T_2M` and `RELHUM_2M`) and making the user come back for the second is the kind of
friction that stops a feature being used.

Nothing here stores a credential: the password lives in the OS keychain if the user asks
for that, otherwise only in this dialog's lifetime.
"""
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import download, ingest


def human(size):
    if not size:
        return '?'
    if size >= 1 << 20:
        return f'{size / (1 << 20):,.0f} MB'
    return f'{size / (1 << 10):,.0f} kB'


class ListWorker(QtCore.QThread):
    """Authenticate and fetch the directory index, off the UI thread."""

    listed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, user, password, parent=None):
        super().__init__(parent)
        self._user, self._password = user, password

    def run(self):
        try:
            session = download.make_session(self._user, self._password)
            self.listed.emit((session, download.fetch_listing(session)))
        except download.DownloadError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class FetchWorker(QtCore.QThread):
    """Download the queued files one at a time, resuming any partial transfer."""

    progressed = QtCore.Signal(str, int, int)     # name, bytes done, bytes total
    finished_all = QtCore.Signal(object)          # [Path] actually completed
    failed = QtCore.Signal(str)

    def __init__(self, session, entries, target_dir, parent=None):
        super().__init__(parent)
        self.session = session
        self.entries = list(entries)
        self.target_dir = Path(target_dir)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        done = []
        try:
            for entry in self.entries:
                if self._cancel:
                    break
                # G27: the local name is rebuilt from the validated run and field, so a
                # server-supplied string can never choose where this writes.
                target = self.target_dir / entry.local_name()
                download.download(
                    self.session, entry.url, target, expected_size=entry.size,
                    progress=lambda a, b, name=entry.local_name():
                        self.progressed.emit(name, a, b),
                    cancel=lambda: self._cancel)
                done.append(target)
            ingest.evict_downloads()
        except download.Cancelled:
            pass                            # the .part stays: the next attempt resumes it
        except download.DownloadError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self.finished_all.emit(done)


class DownloadDialog(QtWidgets.QDialog):
    """Credentials -> listing -> pick run and fields -> fetch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Download from IMS')
        self.setMinimumSize(660, 520)
        self.session = None
        self.files = []
        self.by_run = {}
        self.worker = None
        self.fetcher = None
        self.downloaded = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_credentials())
        layout.addWidget(self._build_picker(), 1)
        layout.addWidget(self._build_footer())
        self._prefill()

    # ---- construction ------------------------------------------------------------
    def _build_credentials(self):
        box = QtWidgets.QGroupBox('IMS account')
        grid = QtWidgets.QGridLayout(box)
        grid.addWidget(QtWidgets.QLabel('User:'), 0, 0)
        self.user_edit = QtWidgets.QLineEdit()
        grid.addWidget(self.user_edit, 0, 1)
        grid.addWidget(QtWidgets.QLabel('Password:'), 0, 2)
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        grid.addWidget(self.password_edit, 0, 3)
        self.connect_button = QtWidgets.QPushButton('Connect')
        self.connect_button.setDefault(True)
        self.connect_button.clicked.connect(self.connect_to_server)
        grid.addWidget(self.connect_button, 0, 4)

        self.remember = QtWidgets.QCheckBox('Remember in the system keychain')
        self.remember.setEnabled(download.keyring_module() is not None)
        self.remember.setToolTip(
            'Stored in the Windows Credential Manager / macOS Keychain, never in the '
            "app's settings file." if self.remember.isEnabled() else
            'Install the `keyring` package to store the password securely.')
        grid.addWidget(self.remember, 1, 1, 1, 3)
        hint = QtWidgets.QLabel('Credentials come from the IMS product PDF. '
                                'IMS_USER / IMS_PASS in the environment are used if set.')
        hint.setStyleSheet('color:#666;')
        grid.addWidget(hint, 2, 0, 1, 5)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return box

    def _build_picker(self):
        box = QtWidgets.QGroupBox('Maps')
        outer = QtWidgets.QVBoxLayout(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Run:'))
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setMinimumWidth(190)
        self.run_combo.currentIndexChanged.connect(lambda _: self._populate_fields())
        row.addWidget(self.run_combo)
        row.addStretch(1)
        self.dewpoint_button = QtWidgets.QPushButton('Select what the dew point needs')
        self.dewpoint_button.setToolTip('Ticks T_2M and RELHUM_2M, the two fields the '
                                        'derived dew point is computed from.')
        self.dewpoint_button.clicked.connect(self._select_dew_point)
        row.addWidget(self.dewpoint_button)
        outer.addLayout(row)

        self.field_list = QtWidgets.QTreeWidget()
        self.field_list.setHeaderLabels(['Map', 'Field', 'Size', 'On disk'])
        self.field_list.setRootIsDecorated(False)
        self.field_list.setAlternatingRowColors(True)
        self.field_list.itemChanged.connect(lambda *_: self._refresh_footer())
        header = self.field_list.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.field_list, 1)
        return box

    def _build_footer(self):
        box = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        self.status = QtWidgets.QLabel('Enter your IMS credentials and press Connect.')
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setVisible(False)
        column.addWidget(self.bar)
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close)
        self.download_button = self.buttons.addButton(
            'Download', QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.start_download)
        self.buttons.rejected.connect(self.reject)
        column.addWidget(self.buttons)
        return box

    def _prefill(self):
        stored = download.stored_credentials()
        if stored:
            self.user_edit.setText(stored[0])
            self.password_edit.setText(stored[1])
            self.status.setText('Credentials found. Press Connect to list the runs '
                                'available on the server.')

    # ---- listing ------------------------------------------------------------------
    def connect_to_server(self):
        user, password = self.user_edit.text().strip(), self.password_edit.text()
        if not user or not password:
            self.status.setText('A user name and password are needed.')
            return
        self.connect_button.setEnabled(False)
        self.status.setText('Connecting to the IMS server...')
        self.worker = ListWorker(user, password, self)
        self.worker.listed.connect(self._on_listed)
        self.worker.failed.connect(self._on_list_failed)
        self.worker.start()

    def _on_listed(self, result):
        self.connect_button.setEnabled(True)
        self.session, self.files = result
        if self.remember.isChecked():
            if not download.remember_credentials(self.user_edit.text().strip(),
                                                 self.password_edit.text()):
                self.status.setText('Connected, but the password could not be stored '
                                    'in the keychain.')
        self.by_run = download.index_by_run(self.files)
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in download.runs_in(self.files):
            self.run_combo.addItem(f'{run[:4]}-{run[4:6]}-{run[6:8]}  {run[8:]}Z', run)
        self.run_combo.blockSignals(False)
        self._populate_fields()
        self.status.setText(f'{len(self.files)} files across '
                            f'{len(self.by_run)} runs. Tick the maps you want.')

    def _on_list_failed(self, message):
        self.connect_button.setEnabled(True)
        self.status.setText(message)

    # ---- the field list -----------------------------------------------------------
    def _populate_fields(self):
        run = self.run_combo.currentData()
        offered = self.by_run.get(run, {})
        on_disk = ingest.scan_for_fields(ingest.search_roots())
        self.field_list.blockSignals(True)
        self.field_list.clear()
        # Catalogue order first, so CAPE and precipitation lead; anything the server
        # offers that this build has never heard of still appears, at the end.
        known = [p.field for p in download.PRODUCTS]
        fields = known + sorted(set(offered) - set(known))
        for field in fields:
            entry = offered.get(field)
            if entry is None:
                continue
            product = download.BY_FIELD.get(field)
            cached = on_disk.get((run, field))
            target = ingest.download_dir() / entry.local_name()
            resumable = download.resume_state(target)
            state = (cached.name if cached else
                     f'partial, {human(resumable)} done' if resumable else '')
            item = QtWidgets.QTreeWidgetItem(
                [product.label if product else field, field, human(entry.size), state])
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, entry)
            if product and product.note:
                item.setToolTip(0, product.note)
            if cached:
                item.setForeground(3, QtGui.QBrush(QtGui.QColor('#2a7a2a')))
            self.field_list.addTopLevelItem(item)
        for column in (1, 2, 3):
            self.field_list.resizeColumnToContents(column)
        self.field_list.blockSignals(False)
        self._refresh_footer()

    def _items(self):
        return [self.field_list.topLevelItem(i)
                for i in range(self.field_list.topLevelItemCount())]

    def _select_dew_point(self):
        for item in self._items():
            if item.text(1) in download.DEW_POINT_FIELDS:
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)

    def selected(self):
        return [item.data(0, QtCore.Qt.ItemDataRole.UserRole) for item in self._items()
                if item.checkState(0) == QtCore.Qt.CheckState.Checked]

    def _refresh_footer(self):
        chosen = self.selected()
        total = sum(entry.size or 0 for entry in chosen)
        self.download_button.setEnabled(bool(chosen) and self.session is not None)
        if chosen:
            self.status.setText(f'{len(chosen)} file(s), about {human(total)} compressed '
                                f'(roughly {human(int(total * 1.55))} once expanded).')

    # ---- the transfer -------------------------------------------------------------
    def start_download(self):
        chosen = self.selected()
        if not chosen or self.session is None:
            return
        self.download_button.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 100)
        self.fetcher = FetchWorker(self.session, chosen, ingest.download_dir(), self)
        self.fetcher.progressed.connect(self._on_progress)
        self.fetcher.finished_all.connect(self._on_done)
        self.fetcher.failed.connect(self._on_failed)
        self.fetcher.start()

    def _on_progress(self, name, done, total):
        self.status.setText(f'{name}: {human(done)} of {human(total)}')
        self.bar.setValue(int(100 * done / total) if total else 0)

    def _on_done(self, paths):
        self.bar.setVisible(False)
        self.downloaded = list(paths)
        self.download_button.setEnabled(True)
        self._populate_fields()
        if paths:
            self.accept()
        else:
            self.status.setText('Download cancelled. Partly-fetched files are kept, '
                                'so starting again resumes them.')

    def _on_failed(self, message):
        self.bar.setVisible(False)
        self.download_button.setEnabled(True)
        self.status.setText(message)

    def reject(self):
        if self.fetcher is not None and self.fetcher.isRunning():
            self.fetcher.cancel()
            self.fetcher.wait(5000)
        super().reject()

    def closeEvent(self, event):
        if self.fetcher is not None and self.fetcher.isRunning():
            self.fetcher.cancel()
            self.fetcher.wait(5000)
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(5000)
        super().closeEvent(event)
