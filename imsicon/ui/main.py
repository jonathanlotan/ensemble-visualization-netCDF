"""MainWindow - map (left) | readout + graph (right), wired together."""
import numpy as np
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import geo, ingest, nc3
from ..dataset import EnsembleFile, member_stats
from .mapview import MapView
from .plotview import PlotView
from .readout import ReadoutPanel

FILE_FILTER = 'ICON ensemble (*.nc *.nc.bz2);;NetCDF (*.nc);;Compressed (*.nc.bz2);;All files (*)'
AGG_CHOICES = [('Ensemble mean', 'mean'), ('Ensemble max', 'max'), ('Ensemble min', 'min'),
               ('Ensemble median', 'median'), ('Spread (max-min)', 'spread'),
               ('Single member', 'member')]
COLORMAPS = ['turbo', 'viridis', 'inferno', 'plasma', 'magma', 'CET-L17']


class ScanWorker(QtCore.QThread):
    """Background pass for the dataset-wide min/max (~0.6 s per 407 MB file)."""
    progressed = QtCore.Signal(int, int)
    finished_range = QtCore.Signal(object)

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        result = self.ds.scan_range(progress=self.progressed.emit, cancel=lambda: self._cancel)
        self.finished_range.emit(result)


class DecompressWorker(QtCore.QThread):
    """`.nc.bz2` -> cached `.nc`, off the UI thread (~16 s for 407 MB)."""
    progressed = QtCore.Signal(int, int)
    finished_path = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            target = ingest.decompress(self.path, progress=self.progressed.emit,
                                       cancel=lambda: self._cancel)
            self.finished_path.emit(target)
        except KeyboardInterrupt:
            self.finished_path.emit(None)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle('IMS ICON Ensemble Viewer')
        self.resize(1500, 880)
        self.setAcceptDrops(True)

        self.settings = QtCore.QSettings('IMS', 'IconEnsembleViewer')
        self.ds = None
        self.t = 0
        self.member = 0
        self.point = None            # (iy, ix)
        self.scan = None
        self.decompressor = None
        self._progress = None

        self._build_ui()
        self._pending = path
        if path is None:
            # F1.1: draw the window first, then put the picker on top of it
            QtCore.QTimer.singleShot(0, self.open_dialog)
        else:
            QtCore.QTimer.singleShot(0, lambda: self.open_path(path))

    # ---- construction ----------------------------------------------------------
    def _build_ui(self):
        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        welcome = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(welcome)
        vbox.addStretch(1)
        title = QtWidgets.QLabel('IMS ICON Ensemble Viewer')
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = title.font()
        font.setPointSize(20)
        font.setBold(True)
        title.setFont(font)
        vbox.addWidget(title)
        hint = QtWidgets.QLabel('Open an ICON_ENS_*.nc or .nc.bz2 file to begin '
                                '(you can also drag one onto this window)')
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet('color:#666;')
        vbox.addWidget(hint)
        button = QtWidgets.QPushButton('Open ICON ensemble file...')
        button.setFixedWidth(260)
        button.clicked.connect(self.open_dialog)
        vbox.addWidget(button, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        vbox.addStretch(2)
        self.stack.addWidget(welcome)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.stack.addWidget(self.splitter)

        left = QtWidgets.QWidget()
        left_box = QtWidgets.QVBoxLayout(left)
        left_box.setContentsMargins(0, 0, 0, 0)
        self.map = MapView()
        left_box.addWidget(self.map, 1)
        left_box.addLayout(self._build_time_bar())
        self.splitter.addWidget(left)

        right = QtWidgets.QWidget()
        right_box = QtWidgets.QVBoxLayout(right)
        right_box.setContentsMargins(0, 0, 0, 0)
        self.readout = ReadoutPanel()
        right_box.addWidget(self.readout)
        self.plot = PlotView()
        right_box.addWidget(self.plot, 1)
        self.splitter.addWidget(right)
        self.splitter.setSizes([720, 780])

        self.map.pointPicked.connect(self.select_point)
        self.map.cursorMoved.connect(self._on_map_cursor)
        self.plot.hovered.connect(self._on_hover)
        self.plot.timePicked.connect(self.set_time)

        self._build_toolbar()
        self.status = self.statusBar()
        self.status_left = QtWidgets.QLabel('No file loaded')
        self.status.addWidget(self.status_left, 1)
        self.status_right = QtWidgets.QLabel('')
        self.status.addPermanentWidget(self.status_right)

    def _build_time_bar(self):
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 0, 8, 6)
        self.play_button = QtWidgets.QToolButton()
        self.play_button.setText('Play')
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_play)
        bar.addWidget(self.play_button)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.set_time)
        bar.addWidget(self.slider, 1)
        self.time_label = QtWidgets.QLabel('--')
        self.time_label.setMinimumWidth(230)
        self.time_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                     | QtCore.Qt.AlignmentFlag.AlignVCenter)
        bar.addWidget(self.time_label)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._advance)
        return bar

    def _build_toolbar(self):
        tb = self.addToolBar('Main')
        tb.setMovable(False)
        open_action = QtGui.QAction('Open...', self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_dialog)
        tb.addAction(open_action)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(' Map shows: '))
        self.agg_combo = QtWidgets.QComboBox()
        for label, key in AGG_CHOICES:
            self.agg_combo.addItem(label, key)
        self.agg_combo.currentIndexChanged.connect(self._on_agg_changed)
        tb.addWidget(self.agg_combo)

        self.member_combo = QtWidgets.QComboBox()
        self.member_combo.setMinimumWidth(120)
        self.member_combo.setEnabled(False)
        self.member_combo.currentIndexChanged.connect(self._on_member_changed)
        tb.addWidget(self.member_combo)

        tb.addWidget(QtWidgets.QLabel('  Colours: '))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        tb.addWidget(self.cmap_combo)

        tb.addWidget(QtWidgets.QLabel('  Scale: '))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(['Dataset range', 'This frame'])
        self.scale_combo.currentIndexChanged.connect(lambda _: self.refresh_map())
        tb.addWidget(self.scale_combo)

        reset = QtGui.QAction('Reset view', self)
        reset.setShortcut(QtGui.QKeySequence('Home'))
        reset.triggered.connect(self.map.reset_view)
        tb.addAction(reset)

        for keys, delta in (('Left', -1), ('Right', 1), ('Shift+Left', -6), ('Shift+Right', 6)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.activated.connect(lambda d=delta: self.set_time(self.t + d))

    # ---- F1: opening files -----------------------------------------------------
    def open_dialog(self):
        start = self.settings.value('last_dir', '')
        if not start or not Path(start).exists():
            local = Path.cwd() / 'data'
            start = str(local if local.exists() else Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open IMS ICON ensemble file', start, FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path):
        path = Path(path)
        if not path.exists():
            self._error(f'File not found:\n{path}')
            return
        self.settings.setValue('last_dir', str(path.parent))
        if ingest.is_compressed(path):
            self._start_decompress(path)
        else:
            self._load(path)

    def _start_decompress(self, path):
        self._progress = QtWidgets.QProgressDialog(
            f'Decompressing {path.name}\n(407 MB, about 16 seconds)...', 'Cancel', 0, 100, self)
        self._progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self.decompressor = DecompressWorker(path, self)
        self.decompressor.progressed.connect(self._on_decompress_progress)
        self.decompressor.finished_path.connect(self._on_decompressed)
        self.decompressor.failed.connect(self._on_decompress_failed)
        self._progress.canceled.connect(self.decompressor.cancel)
        self.decompressor.start()

    def _on_decompress_progress(self, done, total):
        if self._progress is not None and total:
            self._progress.setValue(int(min(99, 100 * done / total)))

    def _on_decompressed(self, target):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        if target is not None:
            self._load(Path(target))

    def _on_decompress_failed(self, message):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        self._error(f'Could not decompress the file:\n{message}')

    def _load(self, path):
        try:
            ds = EnsembleFile(path)
        except nc3.UnsupportedFormat as exc:
            self._error(str(exc))
            return
        except Exception as exc:
            self._error(f'Could not open {Path(path).name}:\n{exc}')
            return

        self._stop_scan()
        self.ds = ds
        self.t = 0
        self.member = 0
        self.stack.setCurrentIndex(1)
        self.setWindowTitle(f'IMS ICON Ensemble Viewer - {path.name}')
        self.status_left.setText(ds.summary())

        self.map.set_dataset(ds, geo.coastline_for(path))
        self.plot.set_dataset(ds)
        self.readout.configure(ds)

        self.member_combo.blockSignals(True)
        self.member_combo.clear()
        self.member_combo.addItems(ds.member_labels)
        self.member_combo.blockSignals(False)

        self.slider.blockSignals(True)
        self.slider.setRange(0, ds.n_times - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)

        cached = ds.cached_range()
        if cached is None:
            self.status_right.setText('scanning for dataset range...')
            self.scan = ScanWorker(ds, self)
            self.scan.finished_range.connect(self._on_scan_done)
            self.scan.start()
        else:
            self._apply_range(cached)

        iy, ix = ds.ny // 2, ds.nx // 2
        self.select_point(iy, ix)
        self.set_time(0)

    def _on_scan_done(self, result):
        self.status_right.setText('')
        if result is not None:
            self._apply_range(result)

    def _apply_range(self, value_range):
        lo, hi = value_range
        self.readout.span = hi - lo
        self.plot.set_yrange(min(0.0, lo) if lo >= 0 else lo, hi)   # A1: pinned to dataset max
        self.refresh_map()

    def _stop_scan(self):
        if self.scan is not None and self.scan.isRunning():
            self.scan.cancel()
            self.scan.wait(2000)
        self.scan = None

    # ---- drawing ---------------------------------------------------------------
    def refresh_map(self):
        if self.ds is None:
            return
        mode = self.agg_combo.currentData()
        if mode == 'member':
            frame = self.ds.frame(self.t, self.member)
            what = self.ds.member_labels[self.member]
        else:
            frame = self.ds.agg_frame(self.t, mode)
            what = self.agg_combo.currentText()
        if self.scale_combo.currentIndex() == 0 and self.ds.value_range is not None:
            lo, hi = self.ds.value_range
            if mode == 'spread':
                lo, hi = 0.0, float(np.nanmax(frame)) or 1.0
        else:
            lo, hi = float(np.nanmin(frame)), float(np.nanmax(frame))
        self.map.set_frame(frame, (lo, hi),
                           f'{self.ds.field} - {what} - {self.ds.label_for(self.t)}')

    def set_time(self, t):
        if self.ds is None:
            return
        t = int(np.clip(t, 0, self.ds.n_times - 1))
        self.t = t
        if self.slider.value() != t:
            self.slider.blockSignals(True)
            self.slider.setValue(t)
            self.slider.blockSignals(False)
        self.time_label.setText(self.ds.label_for(t))
        self.plot.set_time(t)
        self.refresh_map()
        self._update_readout(t, hovering=False)

    def select_point(self, iy, ix):
        """F3.3: a click on the map redirects the whole right-hand panel."""
        if self.ds is None:
            return
        self.point = (iy, ix)
        series = self.ds.series(iy, ix)
        self.plot.set_series(series)
        lat, lon = float(self.ds.lat[iy]), float(self.ds.lon[ix])
        self.map.set_marker(lat, lon)
        self.readout.set_point(lat, lon)
        if self.ds.value_range is None:
            self.plot.set_yrange(float(np.nanmin(series)), float(np.nanmax(series)))
        self._update_readout(self.t, hovering=False)

    def _update_readout(self, t, hovering):
        if self.ds is None or self.point is None or self.plot.series is None:
            return
        t = int(np.clip(t, 0, self.ds.n_times - 1))
        self.readout.show_stats(self.ds.label_for(t),
                                member_stats(self.plot.series[t]), hovering)

    # ---- signals ---------------------------------------------------------------
    def _on_hover(self, t):
        """F4.1: -1 means the cursor left the plot, so fall back to the map's time."""
        self._update_readout(self.t if t < 0 else t, hovering=t >= 0)

    def _on_map_cursor(self, lat, lon, value):
        if self.ds is None or not np.isfinite(lat):
            self.status_right.setText('')
            return
        text = f'{lat:.3f}°N  {lon:.3f}°E'
        if np.isfinite(value):
            text += f'   {value:,.1f} {self.ds.units}'
        self.status_right.setText(text)

    def _on_agg_changed(self):
        self.member_combo.setEnabled(self.agg_combo.currentData() == 'member')
        self.refresh_map()

    def _on_member_changed(self, index):
        self.member = max(0, index)
        if self.agg_combo.currentData() == 'member':
            self.refresh_map()

    def _on_cmap_changed(self, name):
        self.map.set_colormap(name)
        self.refresh_map()

    def _toggle_play(self, on):
        self.play_button.setText('Pause' if on else 'Play')
        self.timer.start() if on else self.timer.stop()

    def _advance(self):
        if self.ds is not None:
            self.set_time((self.t + 1) % self.ds.n_times)

    # ---- misc ------------------------------------------------------------------
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith(('.nc', '.nc.bz2')):
                self.open_path(path)
                break

    def closeEvent(self, ev):
        self._stop_scan()
        if self.decompressor is not None and self.decompressor.isRunning():
            self.decompressor.cancel()
            self.decompressor.wait(3000)
        super().closeEvent(ev)

    def _error(self, message):
        QtWidgets.QMessageBox.critical(self, 'IMS ICON Ensemble Viewer', message)
        if self.ds is None:
            self.stack.setCurrentIndex(0)
