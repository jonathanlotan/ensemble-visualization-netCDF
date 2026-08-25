"""MainWindow - map (left) | readout + graph (right), wired together."""
import numpy as np
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import geo, ingest, nc3, ncwrite, transform
from ..dataset import EnsembleFile, member_stats
from ..fieldview import FieldView
from . import derivedialog, downloaddialog
from .mapview import MapView
from .plotview import PlotView
from .readout import ReadoutPanel

FILE_FILTER = 'ICON ensemble (*.nc *.nc.bz2);;NetCDF (*.nc);;Compressed (*.nc.bz2);;All files (*)'
AGG_CHOICES = [('Ensemble mean', 'mean'), ('Ensemble max', 'max'), ('Ensemble min', 'min'),
               ('Ensemble median', 'median'), ('Spread (max-min)', 'spread'),
               ('Single member', 'member')]
COLORMAPS = ['turbo', 'viridis', 'inferno', 'plasma', 'magma', 'CET-L17',
             # Diverging, for a difference map: a single hue ramp cannot show which side
             # of zero a value is on, which is the only thing a difference map is for.
             'CET-D1A', 'CET-D9', 'CET-D3']
DIVERGING_DEFAULT = 'CET-D1A'
SEQUENTIAL_DEFAULT = 'turbo'


def _finite_max(frame, fallback):
    """G20: `float(np.nanmax(f)) or 1.0` is nan for an all-NaN frame, because bool(nan)
    is True -- and the `hi <= lo` guard downstream never fires, since `nan <= 0.0` is
    False. So an all-NaN frame used to reach cbar.setLevels(high=nan)."""
    values = np.asarray(frame)
    good = values[np.isfinite(values)]
    return float(good.max()) if good.size else float(fallback)


def _finite_min(frame, fallback):
    values = np.asarray(frame)
    good = values[np.isfinite(values)]
    return float(good.min()) if good.size else float(fallback)


def _symmetric(lo, hi):
    """The smallest range about 0 containing (lo, hi) -- a difference map's scale."""
    reach = max(abs(float(lo)), abs(float(hi)))
    return (-reach, reach) if reach > 0 else (-1.0, 1.0)


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


class WriteWorker(QtCore.QThread):
    """`ncwrite.write_canonical` off the UI thread: a full field is 407 MB."""
    progressed = QtCore.Signal(int, int)
    finished_path = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, ds, path, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.path = path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            written = ncwrite.write_canonical(
                self.path, self.ds, progress=self.progressed.emit,
                cancel=lambda: self._cancel)
            self.finished_path.emit(written)
        except ncwrite.Cancelled:
            self.finished_path.emit(None)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


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
        self.builder = None
        self.writer = None
        self._progress = None
        # Every file-backed view this window has opened, so building a derived field on
        # the field already on screen does not map another 407 MB of the same bytes.
        self.opened = {}

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

        self.download_action = QtGui.QAction('Download...', self)
        self.download_action.setShortcut(QtGui.QKeySequence('Ctrl+D'))
        self.download_action.setToolTip('Fetch a map for a chosen run straight from the '
                                        'IMS server')
        self.download_action.triggered.connect(self.download_dialog)
        tb.addAction(self.download_action)

        self.derive_action = QtGui.QAction('Derived field...', self)
        self.derive_action.setShortcut(QtGui.QKeySequence('Ctrl+R'))
        self.derive_action.setToolTip('Dew point from temperature and humidity, or the '
                                      'difference between two fields')
        self.derive_action.triggered.connect(self.derive_dialog)
        tb.addAction(self.derive_action)

        self.save_action = QtGui.QAction('Save field...', self)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.setToolTip('Write what is on screen to a NetCDF file')
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self.save_dialog)
        tb.addAction(self.save_action)
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

        self.addToolBarBreak()          # v2: display controls get their own row
        row2 = self.addToolBar('Display')
        row2.setMovable(False)
        row2.addWidget(QtWidgets.QLabel(' Units: '))
        self.units_combo = QtWidgets.QComboBox()
        self.units_combo.setMinimumWidth(110)
        self.units_combo.setEnabled(False)
        self.units_combo.currentTextChanged.connect(self._on_units_changed)
        row2.addWidget(self.units_combo)
        row2.addWidget(QtWidgets.QLabel('   Rate: '))
        self.rate_combo = QtWidgets.QComboBox()
        self.rate_combo.setMinimumWidth(110)
        self.rate_combo.setEnabled(False)
        for hours in transform.RATE_HOURS:
            self.rate_combo.addItem(transform.rate_label(hours), hours)
        self.rate_combo.currentIndexChanged.connect(self._on_rate_changed)
        row2.addWidget(self.rate_combo)

        self.units_warning = QtWidgets.QLabel('')
        self.units_warning.setStyleSheet('color:#a05000;')
        self.units_warning.hide()
        row2.addWidget(self.units_warning)

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
            raw = EnsembleFile(path)
            saved = self.settings.value(f'units/{raw.field}', None)
            ds = FieldView(raw, units_label=saved)
        except nc3.UnsupportedFormat as exc:
            self._error(str(exc))
            return
        except Exception as exc:
            self._error(f'Could not open {Path(path).name}:\n{exc}')
            return
        self.opened[Path(path)] = ds
        self._install(ds, path.name, near=path)

    def _install(self, ds, title, near=None):
        """Put any view -- a file-backed FieldView or a derived one -- on screen.

        A derived field is not a special case anywhere below this line: `derived.py`
        mirrors `FieldView`'s surface precisely so that the map, the graph, the readout
        and the toolbar keep working on it unchanged.
        """
        self._stop_scan()
        self.ds = ds
        self.t = 0
        self.member = 0
        self.stack.setCurrentIndex(1)
        self.setWindowTitle(f'IMS ICON Ensemble Viewer - {title}')
        self.status_left.setText(ds.summary())

        self.map.set_dataset(ds, geo.coastline_for(near or ds.path))
        self.plot.set_dataset(ds)
        self.readout.configure(ds)
        self._sync_colormap(ds)

        self._sync_units_combo()
        self._sync_rate_combo()
        # Cleared as well as set: this runs again for every field the user opens, and a
        # warning left over from the previous one would be pointing at nothing.
        self.status_right.setText('\u26a0 incomplete file' if ds.truncation_note else '')
        self.status_right.setToolTip(ds.truncation_note or '')

        self.member_combo.blockSignals(True)
        self.member_combo.clear()
        self.member_combo.addItems(ds.member_labels)
        self.member_combo.blockSignals(False)
        self.save_action.setEnabled(True)

        self.slider.blockSignals(True)
        self.slider.setRange(0, ds.n_times - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)

        self._ensure_range()

        iy, ix = ds.ny // 2, ds.nx // 2
        self.select_point(iy, ix)
        self.set_time(0)

    # ---- downloading, deriving, saving ------------------------------------------
    def download_dialog(self):
        """Phase 8 / v2.md 6.2: choose a run and a map, fetch it, open it."""
        dialog = downloaddialog.DownloadDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        fetched = list(dialog.downloaded)
        if not fetched:
            return
        # Several fields can be fetched at once (the dew point needs two). Open the first
        # so the download lands the user in the viewer, and say what else arrived.
        if len(fetched) > 1:
            self.status_right.setText(
                f'downloaded {len(fetched)} files: ' + ', '.join(p.name for p in fetched))
        self.open_path(fetched[0])

    def derive_dialog(self):
        """Dew point, dew point depression, or an A-B difference map."""
        near = self.ds.path if self.ds is not None else None
        run = f'{self.ds.run_init:%Y%m%d%H}' if self.ds is not None else None
        dialog = derivedialog.DerivedDialog(self, near=near, run=run)
        if not dialog.available:
            self._error('No ICON ensemble files were found beside the open file, in '
                        './data, or in the cache. Open or download a run first.')
            return
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        request = dialog.request()
        if request is None:
            return
        self._progress = QtWidgets.QProgressDialog(
            f'Building {request.title}...', 'Cancel', 0, 0, self)
        self._progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self.builder = derivedialog.BuildWorker(request, dict(self.opened), self)
        self.builder.progressed.connect(
            lambda text: self._progress and self._progress.setLabelText(text))
        self.builder.finished_view.connect(lambda view: self._on_derived(view, request))
        self.builder.failed.connect(self._on_derive_failed)
        self._progress.canceled.connect(self.builder.cancel)
        self.builder.start()

    def _on_derived(self, view, request):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        if view is None:
            return
        saved = self.settings.value(f'units/{view.field}', None)
        if saved:
            view.set_units(saved)
        self._install(view, request.title)
        if view.note:
            self.status_right.setText('⚠ ' + view.note)

    def _on_derive_failed(self, message):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        self._error(message)

    def save_dialog(self):
        """Write what is on screen to a real NetCDF-3 file (the "writes" half of F5)."""
        if self.ds is None:
            return
        run = f'{self.ds.run_init:%Y%m%d%H}'
        name = f'ICON_ENS_{run}_{ncwrite.nc_variable_name(self.ds.field)}.nc'
        start = self.settings.value('last_dir', '') or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Write field to NetCDF', str(Path(start) / name), 'NetCDF (*.nc)')
        if not path:
            return
        units = self.ds.canonical_units
        self._progress = QtWidgets.QProgressDialog(
            f'Writing {Path(path).name} in {units}...', 'Cancel', 0,
            self.ds.n_times, self)
        self._progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self.writer = WriteWorker(self.ds, path, self)
        self.writer.progressed.connect(self._on_write_progress)
        self.writer.finished_path.connect(self._on_written)
        self.writer.failed.connect(self._on_write_failed)
        self._progress.canceled.connect(self.writer.cancel)
        self.writer.start()

    def _on_write_progress(self, done, total):
        if self._progress is not None:
            self._progress.setValue(done)

    def _on_written(self, path):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        if path is not None:
            self.status_right.setText(f'wrote {Path(path).name}')

    def _on_write_failed(self, message):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        self._error(f'Could not write the field:\n{message}')

    def _sync_colormap(self, ds):
        """A difference wants a diverging ramp; anything else wants a sequential one."""
        wanted = DIVERGING_DEFAULT if getattr(ds, 'diverging', False) else SEQUENTIAL_DEFAULT
        if self.cmap_combo.currentText() != wanted:
            self.cmap_combo.setCurrentText(wanted)      # fires _on_cmap_changed
        else:
            self.map.set_colormap(wanted)

    def _ensure_range(self):
        """Range for the CURRENT transform view: cached, or one background scan (G19).

        A unit change never lands here -- its range is the cached one, transformed.
        """
        if self.ds is None:
            return
        cached = self.ds.cached_range()
        if cached is not None:
            self._apply_range(self.ds.value_range)
            return
        self._stop_scan()
        self.status_right.setText('scanning for dataset range...')
        self.scan = ScanWorker(self.ds, self)
        self.scan.finished_range.connect(self._on_scan_done)
        self.scan.start()

    def _on_scan_done(self, result):
        self.status_right.setText('')
        if result is not None and self.ds is not None:
            self._apply_range(self.ds.value_range)

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
                lo, hi = 0.0, _finite_max(frame, 1.0)
        else:
            lo, hi = _finite_min(frame, 0.0), _finite_max(frame, 1.0)
        if getattr(self.ds, 'diverging', False) and mode != 'spread':
            # A difference map has to be symmetric about zero, or the colour that means
            # "no difference" moves with the data and +2 K reads as the same colour as
            # -2 K did a frame earlier. `spread` is excluded: it is non-negative already.
            lo, hi = _symmetric(lo, hi)
        units = f' [{self.ds.units}]' if self.ds.units else ''
        self.map.set_frame(frame, (lo, hi),
                           f'{self.ds.field}{units} - {what} - {self.ds.label_for(self.t)}')

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

    def _sync_units_combo(self):
        """Reflect what the registry offers for this field (v2 1.3/1.4)."""
        self.units_combo.blockSignals(True)
        self.units_combo.clear()
        self.units_combo.addItems(self.ds.unit_labels)
        self.units_combo.setCurrentText(self.ds.units)
        # Disabled, not hidden: a stable layout beats a jumping toolbar.
        self.units_combo.setEnabled(self.ds.can_convert_units)
        self.units_combo.blockSignals(False)
        note = self.ds.units_note
        self.units_combo.setToolTip(note or f'Display units for {self.ds.field}')
        self.units_warning.setText(' \u26a0 no unit conversion offered' if note else '')
        self.units_warning.setToolTip(note or '')
        self.units_warning.setVisible(bool(note))

    def _on_units_changed(self, label):
        """One switch has to move the map, colorbar, y axis, readout and status bar
        together -- which is exactly why the conversion lives in FieldView and not here."""
        if self.ds is None or not label or not self.ds.set_units(label):
            return
        self.settings.setValue(f'units/{self.ds.field}', label)
        self._refresh_units()

    def _refresh_units(self):
        ds = self.ds
        self.status_left.setText(ds.summary())
        self.plot.getAxis('left').enableAutoSIPrefix(False)   # G12: never 'kJ kg-1'
        self.plot.setLabel('left', ds.field, units=ds.units or None)
        self.readout.configure(ds)          # select_point below restores the point label
        self.time_label.setText(ds.label_for(self.t))
        if ds.value_range is not None:
            self._apply_range(ds.value_range)
        if self.point is not None:
            self.select_point(*self.point)
        else:
            self.refresh_map()

    def _sync_rate_combo(self):
        """Disabled, not hidden, when the field is not accumulated -- a stable layout."""
        self.rate_combo.blockSignals(True)
        self.rate_combo.setCurrentIndex(max(0, self.rate_combo.findData(self.ds.rate_hours)))
        self.rate_combo.setEnabled(self.ds.can_rate)
        self.rate_combo.blockSignals(False)
        self.rate_combo.setToolTip(
            f'{self.ds.field} accumulates since model start ({self.ds.accum_kind}-kind): '
            'show the value over a 1 h or 3 h window instead'
            if self.ds.can_rate else
            f'{self.ds.field} is not an accumulated field, so it has no window rate')

    def _on_rate_changed(self, _index):
        if self.ds is None:
            return
        hours = self.rate_combo.currentData() or 0
        if not self.ds.set_rate(hours):
            self._sync_rate_combo()
            return
        # V2.4.4: landing on a blank map reads as a broken app, so skip past the gap.
        k = self.ds.window_steps
        if self.t < k:
            self.t = k
            self.slider.blockSignals(True)
            self.slider.setValue(k)
            self.slider.blockSignals(False)
        self._refresh_units()
        self._ensure_range()
        if self.ds.rate_note:
            self.status_right.setText('\u26a0 ' + self.ds.rate_note)

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
        for worker in (self.decompressor, self.builder, self.writer):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(3000)
        super().closeEvent(ev)

    def _error(self, message):
        QtWidgets.QMessageBox.critical(self, 'IMS ICON Ensemble Viewer', message)
        if self.ds is None:
            self.stack.setCurrentIndex(0)
