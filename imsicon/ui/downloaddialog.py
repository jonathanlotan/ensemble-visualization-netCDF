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

from .. import download, ingest, products


def human(size):
    if not size:
        return '?'
    if size >= 1 << 20:
        return f'{size / (1 << 20):,.0f} MB'
    return f'{size / (1 << 10):,.0f} kB'


class ListWorker(QtCore.QThread):
    """Authenticate and fetch one family's directory index, off the UI thread.

    `session` may be an already-authenticated one: switching the product combo re-lists a
    different folder and must not make the user type the password again.
    """

    listed = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, user, password, family=None, session=None, parent=None):
        super().__init__(parent)
        self._user, self._password = user, password
        self._family = family or products.ENSEMBLE
        self._session = session

    def run(self):
        try:
            session = self._session or download.make_session(self._user, self._password)
            self.listed.emit((session,
                              download.fetch_listing(session, family=self._family),
                              self._family))
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
        self.family = products.ENSEMBLE

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
        # Which product. The ensemble gives 20 members of one 00Z run a day; the
        # deterministic ICON-LAM run gives 00Z and 12Z, 48 fields, and six of them on 20
        # pressure levels -- which is what the viewer's Level control steps through.
        row.addWidget(QtWidgets.QLabel('Product:'))
        self.family_combo = QtWidgets.QComboBox()
        for family in products.FAMILIES:
            self.family_combo.addItem(family.title, family.key)
        self.family_combo.setToolTip(
            'IMS publishes the model twice: an ensemble of 20 members (one 00Z run a '
            'day), and the deterministic ICON-LAM run (00Z and 12Z), whose temperature, '
            'humidity, wind, omega and geopotential come on 20 pressure levels.')
        self.family_combo.currentIndexChanged.connect(self._on_family_changed)
        row.addWidget(self.family_combo)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel('Run:'))
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.setMinimumWidth(190)
        self.run_combo.currentIndexChanged.connect(lambda _: self._populate_fields())
        row.addWidget(self.run_combo)
        row.addStretch(1)
        self.dewpoint_button = QtWidgets.QPushButton('Select what the dew point needs')
        self.dewpoint_button.setToolTip(
            'Ticks T_2M and RELHUM_2M, the two fields the derived dew point is computed '
            'from. With both on disk the viewer offers four maps: the temperature, the '
            'humidity, the dew point TD_2M and the depression T-Td.')
        self.dewpoint_button.clicked.connect(self._select_dew_point)
        row.addWidget(self.dewpoint_button)
        self.wind_button = QtWidgets.QPushButton('Select what the wind map needs')
        self.wind_button.setToolTip(
            'Ticks U_10M and V_10M, the two components the wind map is built from. One '
            'of them on its own is half a wind: the barbs need both.')
        self.wind_button.clicked.connect(self._select_wind)
        row.addWidget(self.wind_button)
        outer.addLayout(row)

        # Search, so a map can be found by typing rather than by scrolling 50 rows.
        # It only HIDES rows: a ticked map stays ticked (and downloads) while filtered out.
        search_row = QtWidgets.QHBoxLayout()
        search_row.addWidget(QtWidgets.QLabel('Search:'))
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText('e.g. cape, precip, wind, temp, 2m, levels')
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setToolTip(
            'Shows only the maps whose name, field or description contain every word '
            'typed. Ticked maps stay ticked while hidden. Enter ticks the map when only '
            'one is left; Esc clears the search.')
        self.search_edit.textChanged.connect(lambda _: self._apply_search())
        # Enter is taken here rather than through returnPressed: a QLineEdit lets Enter
        # through to the dialog, whose default button would then connect or download.
        self.search_edit.installEventFilter(self)
        search_row.addWidget(self.search_edit, 1)
        self.match_label = QtWidgets.QLabel('')
        self.match_label.setStyleSheet('color:#666;')
        search_row.addWidget(self.match_label)
        outer.addLayout(search_row)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self,
                        activated=lambda: (self.search_edit.setFocus(),
                                           self.search_edit.selectAll()))
        QtGui.QShortcut(QtGui.QKeySequence('Escape'), self.search_edit,
                        activated=self.search_edit.clear,
                        context=QtCore.Qt.ShortcutContext.WidgetShortcut)

        self.field_list = QtWidgets.QTreeWidget()
        self.field_list.setHeaderLabels(['Map', 'Field', 'Levels', 'Size', 'On disk'])
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
        self.status.setText(f'Connecting to the IMS server ({self.family.short})...')
        self.worker = ListWorker(user, password, self.family, self.session, self)
        self.worker.listed.connect(self._on_listed)
        self.worker.failed.connect(self._on_list_failed)
        self.worker.start()

    def _on_family_changed(self, _index):
        """Switching product re-lists the other folder, reusing the open session."""
        self.family = products.BY_KEY.get(self.family_combo.currentData(),
                                          products.ENSEMBLE)
        self.files, self.by_run = [], {}
        self.run_combo.clear()
        self._populate_fields()
        for button, fields in ((self.dewpoint_button, download.dew_point_fields(self.family)),
                               (self.wind_button, download.wind_fields(self.family))):
            button.setEnabled(bool(fields))
        if self.session is not None:
            self.connect_to_server()
        else:
            self.status.setText(f'{self.family.title}: press Connect to list its runs.')

    def _on_listed(self, result):
        self.connect_button.setEnabled(True)
        # (session, files) or (session, files, family): a listing that does not name a
        # family belongs to the one currently selected.
        self.session, self.files = result[0], result[1]
        if len(result) > 2 and result[2] is not None:
            self.family = result[2]
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
        self.status.setText(f'{self.family.title}: {len(self.files)} files across '
                            f'{len(self.by_run)} runs. Tick the maps you want.')

    def _on_list_failed(self, message):
        self.connect_button.setEnabled(True)
        self.status.setText(message)

    # ---- the field list -----------------------------------------------------------
    def _populate_fields(self):
        run = self.run_combo.currentData()
        offered = self.by_run.get(run, {})
        on_disk = ingest.fields_of_run(
            ingest.scan_for_fields(ingest.search_roots()), self.family, run)
        self.field_list.blockSignals(True)
        self.field_list.clear()
        # Catalogue order first, so CAPE and precipitation lead; anything the server
        # offers that this build has never heard of still appears, at the end.
        known = [p.field for p in self.family.products]
        fields = known + sorted(set(offered) - set(known))
        for field in fields:
            entry = offered.get(field)
            if entry is None:
                continue
            product = self.family.by_field.get(field)
            cached = on_disk.get(field)
            target = ingest.download_dir() / entry.local_name()
            resumable = download.resume_state(target)
            state = (cached.name if cached else
                     f'partial, {human(resumable)} done' if resumable else '')
            # Which maps give the viewer a Level control to step through, said in the list
            # rather than discovered after a 262 MB download.
            depth = (f'{len(products.PRESSURE_LEVELS)} pressure levels'
                     if entry.levels else 'surface')
            item = QtWidgets.QTreeWidgetItem(
                [product.label if product else field, field, depth,
                 human(entry.size), state])
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, entry)
            if product and product.note:
                item.setToolTip(0, product.note)
            if cached:
                item.setForeground(4, QtGui.QBrush(QtGui.QColor('#2a7a2a')))
            self.field_list.addTopLevelItem(item)
        for column in (1, 2, 3, 4):
            self.field_list.resizeColumnToContents(column)
        self.field_list.blockSignals(False)
        self._apply_search()
        self._refresh_footer()

    @staticmethod
    def _haystack(item):
        text = ' '.join([item.text(c) for c in range(item.columnCount())]
                        + [item.toolTip(0)]).lower()
        # 'tot prec' finds TOT_PREC, and 'u10m' finds U_10M.
        return f'{text} {text.replace("_", " ")} {text.replace("_", "")}'

    def visible_items(self):
        return [item for item in self._items() if not item.isHidden()]

    def _apply_search(self):
        """Hide the rows that do not contain every word of the search."""
        words = self.search_edit.text().lower().split()
        items = self._items()
        for item in items:
            haystack = self._haystack(item)
            item.setHidden(not all(word in haystack for word in words))
        shown = len(self.visible_items())
        if not words:
            self.match_label.setText('')
        else:
            hidden_ticked = sum(1 for item in items if item.isHidden()
                                and item.checkState(0) == QtCore.Qt.CheckState.Checked)
            self.match_label.setText(
                f'{shown} of {len(items)} maps'
                + (f' ({hidden_ticked} ticked map{"s" if hidden_ticked != 1 else ""} '
                   'hidden)' if hidden_ticked else '')
                if shown else 'no map matches')

    def eventFilter(self, watched, event):
        if (watched is getattr(self, 'search_edit', None)
                and event.type() == QtCore.QEvent.Type.KeyPress
                and event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter)):
            self._tick_single_match()
            return True
        return super().eventFilter(watched, event)

    def _tick_single_match(self):
        """Enter on a search that leaves exactly one map ticks (or unticks) it."""
        shown = self.visible_items()
        if len(shown) == 1:
            item = shown[0]
            item.setCheckState(0, QtCore.Qt.CheckState.Unchecked
                               if item.checkState(0) == QtCore.Qt.CheckState.Checked
                               else QtCore.Qt.CheckState.Checked)

    def _items(self):
        return [self.field_list.topLevelItem(i)
                for i in range(self.field_list.topLevelItemCount())]

    def _select_dew_point(self):
        self._select_fields(download.dew_point_fields(self.family))

    def _select_wind(self):
        self._select_fields(download.wind_fields(self.family))

    def _select_fields(self, fields):
        for item in self._items():
            if item.text(1) in fields:
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
