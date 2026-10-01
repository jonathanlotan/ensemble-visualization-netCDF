"""MainWindow - map (left) | readout + graph (right), wired together."""
import numpy as np
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import config as user_config
from .. import (derived, geo, ingest, isolines, nc3, ncwrite, products, terrain, timefmt,
               transform)
from ..dataset import EnsembleFile, level_stats, member_stats
from ..fieldview import FieldView
from . import colors, derivedialog, downloaddialog, settingsdialog
from .mapview import MapView
from .plotview import PlotView
from .profileview import ProfileView
from .readout import ReadoutPanel

FILE_FILTER = ('IMS ICON (*.nc *.nc.bz2);;NetCDF (*.nc);;Compressed (*.nc.bz2);;'
               'All files (*)')
AGG_CHOICES = [('Ensemble mean', 'mean'), ('Ensemble max', 'max'), ('Ensemble min', 'min'),
               ('Ensemble median', 'median'), ('Spread (max-min)', 'spread'),
               ('Single member', 'member')]
# A file whose second axis is pressure levels gets ONE choice, and the map always shows one
# level. Not a restriction for its own sake: a mean, a max or a spread across 1000..150 hPa
# is not a quantity anyone forecasts, and it would look exactly as convincing as one that
# is (`levels.py`, and G16's family of plausible-but-meaningless statistics).
LEVEL_CHOICES = [('Single level', 'member')]
# The colour ramps live in `ui/colors.py` so the Settings dialog can offer the same list.
from .colors import (COLORMAPS, DIVERGING_DEFAULT, DIVERGING_MAPS,  # noqa: E402,F401
                     SEQUENTIAL_DEFAULT, SEQUENTIAL_MAPS)
# The Scale combo's entries, by position. The third is only live on a map the settings
# file gives a fixed range for (R10).
SCALE_DATASET, SCALE_FRAME, SCALE_FIXED = 0, 1, 2


# What the wind barbs mean, said once. The glyph is defined in knots whatever the colour
# scale is set to -- a half feather is 5 kt, not "5 of whatever the toolbar says".
BARB_TOOLTIP = ('Wind barbs on the map, in knots: half feather 5 kt, full feather 10 kt, '
                'pennant 50 kt, open circle calm. The staff points into the wind (the '
                'direction it blows FROM), and the barbs thin out or fill in as you zoom '
                'so they stay about a finger-width apart.')


# R5. Both tooltips say the interval in the units on screen rather than in the abstract,
# because the whole point of the G15 scaling is that "every 1 degree Celsius" survives a
# switch to degF as "every 1.8 degF" -- and a reader has to be able to see that it did.
def _isoline_tooltip(ds):
    interval = getattr(ds, 'isolines', None) if ds is not None else None
    if interval is None:
        name = ds.display_name if ds is not None else 'This field'
        return (f'{name} is not contoured. Isolines are drawn on the temperature maps '
                '(every 1 °C by default) and on a difference between two of them, such '
                'as T-Td (every 0.5 °C). Giving a map an isoline_step under Settings... '
                'makes it contourable too.')
    units = f' {ds.units}' if ds.units else ''
    return (f'Draw contour lines over the map: currently every {interval.step:g}{units}, '
            f'with every {interval.emphasis} ({interval.step * interval.emphasis:g}'
            f'{units}) drawn heavier. Use the slider beside it to choose how close they '
            'are.')


def _ladder_of(ds):
    """The spacings the slider offers for this view: degrees, or gpm on a height chart."""
    ladder = getattr(ds, 'isoline_ladder', None) if ds is not None else None
    return ladder or isolines.DEGREES


def _isoline_step_tooltip(ds, live):
    """The slider's own tooltip: the choice, and the one thing about it worth stating."""
    ladder = _ladder_of(ds)
    choices = ', '.join(f'{step:g}' for step in ladder.steps)
    if not live:
        name = getattr(ds, 'display_name', None) or 'this field'
        why = ('Tick Isolines to space the lines on ' + name
               if getattr(ds, 'isolines', None) is not None
               else f'{name} is not contoured, so there is nothing to space')
        return f'{why}. The spacings on offer are {choices} {ladder.unit}.'
    if ladder is isolines.DEGREES:
        example = ' (2 °C reads as 3.6 °F)'
    elif ladder is isolines.HEIGHT:
        example = ' (500 ft reads as 152.4 gpm)'
    else:
        example = ''
    return (f'How close the isolines are: {choices} {ladder.unit}. The spacing is fixed '
            f'in {ladder.unit}, so switching the display units relabels the lines'
            f'{example} rather than drawing a different set of them.')


ISOLINE_LABELS_TOOLTIP = (
    "Write each isoline's value on it, in the units on screen. Spread out so they never "
    'overlap -- the heavier lines are labelled first -- and re-placed as you zoom, so '
    'zooming in shows more of them.')

TOPO_TOOLTIP = ('Shade the terrain under the map: slopes facing away from a north-west '
                'sun are darkened, flat ground and the sea are left alone, so the hills '
                'show through whatever field is drawn over them. From the bundled ETOPO1 '
                'elevation grid (1 arc-minute, NOAA, public domain), not from the model.')


def _profile_tooltip(ds):
    if ds is None or not getattr(ds.axis, 'is_pressure', False):
        name = ds.display_name if ds is not None else 'This field'
        return (f'{name} is not on pressure levels, so there is no column to draw as a '
                'profile. The graph shows the point through time.')
    return (f'Draw the column at the chosen point as a profile: {ds.display_name} across, '
            'height up, one point per pressure level, at the time step on the slider. '
            'The height is the geopotential height of each level from the run\'s geopot '
            'file; without it the levels are drawn against pressure instead. Unticked, '
            'the graph shows every level through time.')


def _sort_tooltip(ds):
    scale = getattr(ds, 'sort_scale', None) if ds is not None else None
    if scale is None:
        name = ds.display_name if ds is not None else 'This field'
        return (f'Sort applies to the dew point depression (T-Td), not to {name}. '
                'Choose T-Td under "Map shows".')
    units = f' {ds.units}' if ds.units else ''
    stops = ', '.join(f'{value:g}{units} {name}' for value, name in scale.described())
    return (f'Show colour only where the depression is under {scale.top:g}{units} -- '
            f'{stops} -- so the map says where the air is near saturation and stops '
            'colouring everywhere that is not. Drier than that is left white; the '
            'isolines still run through it.')


def _level_tooltip(ds):
    """What the level/member control does, in the words of the file that is open."""
    if ds is None:
        return 'Choose one member or pressure level once a file is open'
    axis = ds.axis
    if axis.is_pressure:
        return (f'Which of the {axis.n} pressure levels the map shows, listed with the '
                f'top of the atmosphere first. The Up and Down arrow keys (and the two '
                f'buttons beside this list) step through them -- Up goes higher, towards '
                f'{axis.labels[axis.upward[-1]]}, whichever order the file stores them in.')
    if axis.kind == 'single':
        return f'{ds.display_name} is a surface field: it has one level and nothing to choose.'
    return ('Which ensemble member the map shows, when "Single member" is selected. '
            'The Up and Down arrow keys step through them.')


# Menu names for the fields the download catalogue does not carry. A `TD_2M` written by
# "Save field..." opens like any other file, and listing it as a bare code would make the
# one map the user built by hand the only one in the menu without a name.
EXTRA_FIELD_LABELS = {derived.DEW_POINT_FIELD: 'dew point'}


def _field_label(field, view=None, family=None):
    """`FIELD - what it is`, for the Map shows combo.

    The catalogue answers this without opening the file, which matters: most of the maps
    in the list are still compressed on disk and expanding one costs 16 s. A file whose
    field the catalogue has never heard of falls back to its own header once it is open.
    """
    family = family or products.ENSEMBLE
    product = family.by_field.get(field)
    if product is not None:
        return f'{field} - {product.label}'[:60]
    if field in EXTRA_FIELD_LABELS:
        return f'{field} - {EXTRA_FIELD_LABELS[field]}'[:60]
    if view is not None:
        return f'{view.display_name} - {view.long_name}'[:60]
    return field


def _field_of(path):
    """The field in an `ICON_ENS_<run>_<FIELD>` / `IE_<run>_<field>` name, else the name."""
    parsed = products.parse_name(Path(path).name)
    return parsed[2] if parsed else Path(path).name


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


def _zero_is_the_floor(lo, hi):
    """Is the bottom of this colour scale the value zero?

    The one question that decides whether the map fades out where the field is zero
    (`ui/colors.py`). Zero at the *floor* means the field cannot go lower, so a zero cell
    is an absence -- no CAPE, no rain, no snow, no spread between the members -- and
    drawing it as nothing is honest. Zero anywhere else on the scale is an ordinary
    reading with colder or drier values below it, and fading it would hide them.

    The tolerance is relative to the span because the floor is a float that has been
    through a units affine: 0 mm of rain converted to inches is not exactly 0.0.
    """
    lo, hi = float(lo), float(hi)
    span = abs(hi - lo)
    return bool(np.isfinite(lo) and np.isfinite(hi)
                and abs(lo) <= 1e-6 * (span or 1.0))


def convert_range(value_range, source, choices, target, difference=False):
    """`(lo, hi)` stated in units `source` -> the same range in the `target` affine.

    `choices` are the view's `Affine`s, all relative to its canonical values; a range in
    units the view does not offer cannot be placed, so it is None rather than a guess. A
    difference view takes the scale only (G15: no difference is no difference in every
    unit).
    """
    lo, hi = value_range
    if target is None or source == target.label:
        return float(lo), float(hi)
    src = next((c for c in choices if c.label == source), None)
    if src is None or not src.a:
        return None

    def convert(value):
        if difference:
            return target.a * (value / src.a)
        return target.a * ((value - src.b) / src.a) + target.b

    a, b = convert(float(lo)), convert(float(hi))
    return (a, b) if a <= b else (b, a)


def app_settings():
    """The one place the app's preferences are opened.

    A function rather than a line in `__init__` so the test suite can swap it for a
    throw-away `.ini` (G47): on macOS `QSettings(org, app)` ignores `setDefaultFormat` and
    goes straight to `~/Library/Preferences`, so without this seam a test run wiped the
    developer's real preferences on every test -- the exact failure G25 was written for.
    """
    return QtCore.QSettings('IMS', 'IconEnsembleViewer')


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

        self.settings = app_settings()
        self.ds = None
        self.t = 0
        # Index on the file's SECOND axis: an ensemble member, or a pressure level. One
        # attribute for both, because everything below it -- `frame(t, member)`,
        # `wind_vectors`, `direction_at` -- takes a position on that axis and does not
        # care what the axis means.
        self.level = 0
        self.point = None            # (iy, ix)
        self.scan = None
        self.decompressor = None
        self.builder = None
        self.writer = None
        # The view that came from a file. A derived map replaces `ds` but not this, so the
        # "Map shows" combo can switch back without reopening 407 MB.
        self.base_ds = None
        self._field_requests = {}       # derived entries -> DerivedRequest
        self._field_files = {}          # file-backed entries -> Path
        # Directories this window has been pointed at, newest first. A decompressed file
        # lives in the cache, so once the user switches to one, the directory they opened
        # from would otherwise be forgotten -- and the rest of the run with it.
        self._roots = []
        self._progress = None
        # What colour scale the map is currently carrying, so a scrub does not rebuild an
        # identical lookup table on every frame. See `_apply_colormap`.
        self._cmap_state = None
        # Every file-backed view this window has opened, so building a derived field on
        # the field already on screen does not map another 407 MB of the same bytes.
        self.opened = {}
        # R8: the run's wind, held beside whatever is on screen, so barbs can be drawn
        # over ANY map. `_overlay_key` is the shape it was built to sit on -- run, axis,
        # grid and number of steps -- so switching between two fields of one run keeps it
        # while it fits, and drops it when it does not (`_overlay_shape`).
        self.wind_overlay = None
        self._overlay_key = None
        self._overlay_builder = None
        # R5.9: the contour spacing chosen per field, in canonical units. Per field
        # because 2 degC on a temperature map and 2 degC on a depression are different
        # readings, so flipping between them under "Map shows" must not carry one choice
        # onto the other -- and must not throw the first one away either.
        self._iso_steps = {}
        # R9: the run's geopotential, held beside a pressure-level map so the readout can
        # say how high the level shown is at the chosen point, and the profile can put
        # each level at its real height. Same arrangement as the wind overlay: opened on
        # demand, kept while it fits the map (`_overlay_shape`), dropped when it does not.
        self.height_companion = None
        self._height_key = None
        self._height_builder = None
        # R9: whether the right-hand panel is the profile or the time graph, per field.
        self._profile_choices = {}
        # R10: the user's configuration file -- credentials, points and per-map defaults.
        # A choice made in the app wins over it for the rest of the session, which is what
        # these per-field dicts remember: units, and whether the isolines are on.
        self.config = user_config.load()
        timefmt.set_zone(self.config.time_zone)     # R11: before any label is written
        self._install_config_isolines()
        self._custom_colour_problems = colors.set_custom_ramp(self.config.custom_colours)
        self._units_chosen = {}
        self._iso_on = {}
        self._iso_last_choice = None

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
        hint = QtWidgets.QLabel(
            'Open an ICON_ENS_*.nc (ensemble) or IE_*.nc (deterministic, on pressure '
            'levels) file to begin - .nc.bz2 works too, and you can drag one onto this '
            'window')
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet('color:#666;')
        vbox.addWidget(hint)
        button = QtWidgets.QPushButton('Open an IMS ICON file...')
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
        # Two readings of the same column share the panel: the time graph, and (R9) the
        # vertical profile at one time. A stack rather than two panels, because the
        # readout above describes whichever one is showing and must sit directly over it.
        self.plot = PlotView()
        self.profile = ProfileView()
        self.graph_stack = QtWidgets.QStackedWidget()
        self.graph_stack.addWidget(self.plot)
        self.graph_stack.addWidget(self.profile)
        right_box.addWidget(self.graph_stack, 1)
        self.splitter.addWidget(right)
        self.splitter.setSizes([720, 780])

        self.map.pointPicked.connect(self.select_point)
        self.map.cursorMoved.connect(self._on_map_cursor)
        self.plot.hovered.connect(self._on_hover)
        self.plot.timePicked.connect(self.set_time)
        self.profile.levelHovered.connect(self._on_profile_hover)
        self.profile.levelPicked.connect(self.set_level)

        self._build_toolbar()
        self.status = self.statusBar()
        self.status_left = QtWidgets.QLabel('No file loaded')
        self.status.addWidget(self.status_left, 1)
        self.status_right = QtWidgets.QLabel('')
        self.status.addPermanentWidget(self.status_right)
        self._apply_points()
        self._report_config_problems()

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
        # R11: which clock the times are written in. The files are UTC and the slider, the
        # forecast hour and the run id stay so; only the text changes.
        self.tz_combo = QtWidgets.QComboBox()
        for zone in timefmt.ZONES:
            self.tz_combo.addItem(timefmt.LABELS[zone], zone)
        self.tz_combo.setCurrentIndex(self.tz_combo.findData(timefmt.zone()))
        self.tz_combo.setToolTip('Show times in Zulu (UTC, as the model files are) or in '
                                 'Israel time -- IDT (UTC+3) in summer, IST (UTC+2) in '
                                 'winter, decided for each time step.\nThe default is '
                                 'in Settings... > Display.')
        self.tz_combo.currentIndexChanged.connect(
            lambda _i: self.set_time_zone(self.tz_combo.currentData()))
        bar.addWidget(self.tz_combo)
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

        self.settings_action = QtGui.QAction('Settings...', self)
        self.settings_action.setShortcut(QtGui.QKeySequence.StandardKey.Preferences)
        self.settings_action.setToolTip('Credentials, points on the map, and the units, '
                                        'colours, scale, isolines and profile each map '
                                        'opens with')
        self.settings_action.triggered.connect(self.settings_dialog)
        tb.addAction(self.settings_action)
        tb.addSeparator()

        tb.addWidget(QtWidgets.QLabel(' Map shows: '))
        # Which field. The derived maps are here rather than only behind a dialog because
        # switching between T_2M and T-Td is something you do while reading a forecast,
        # not a one-off setup step.
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.setMinimumWidth(150)
        # The list is rebuilt whenever a file is opened and a run can hold 15 maps, so it
        # has to widen to whatever is in it -- the default only measures once, at first show.
        self.field_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        tb.addWidget(self.field_combo)

        self.agg_combo = QtWidgets.QComboBox()
        for label, key in AGG_CHOICES:
            self.agg_combo.addItem(label, key)
        self.agg_combo.currentIndexChanged.connect(self._on_agg_changed)
        tb.addWidget(self.agg_combo)

        # The second-axis picker. On an ensemble it names a member; on a pressure-level
        # file it names the level, and the two buttons beside it are the "up and down"
        # that walk the column. Disabled, not hidden, when there is nothing to choose --
        # the rule Rate, Wind barbs and Isolines already follow.
        self.level_title = QtWidgets.QLabel(' Member: ')
        tb.addWidget(self.level_title)
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.setMinimumWidth(120)
        self.level_combo.setEnabled(False)
        self.level_combo.currentIndexChanged.connect(self._on_level_selected)
        tb.addWidget(self.level_combo)
        self.level_up = QtWidgets.QToolButton()
        self.level_up.setText('\u25b2')
        self.level_up.setAutoRepeat(True)
        self.level_up.setEnabled(False)
        self.level_up.clicked.connect(lambda: self.step_level(1))
        tb.addWidget(self.level_up)
        self.level_down = QtWidgets.QToolButton()
        self.level_down.setText('\u25bc')
        self.level_down.setAutoRepeat(True)
        self.level_down.setEnabled(False)
        self.level_down.clicked.connect(lambda: self.step_level(-1))
        tb.addWidget(self.level_down)

        tb.addWidget(QtWidgets.QLabel('  Colours: '))
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(colors.ramps())
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        tb.addWidget(self.cmap_combo)

        tb.addWidget(QtWidgets.QLabel('  Scale: '))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(['Dataset range', 'This frame', 'Fixed'])
        self.scale_combo.model().item(SCALE_FIXED).setEnabled(False)
        self.scale_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.scale_combo.currentIndexChanged.connect(self._on_scale_changed)
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

        # Disabled, not hidden, exactly like Rate: the control is part of the layout
        # whether or not the field on screen has a direction to draw.
        self.barbs_check = QtWidgets.QCheckBox('  Wind barbs')
        self.barbs_check.setChecked(True)
        self.barbs_check.setEnabled(False)
        self.barbs_check.setToolTip(BARB_TOOLTIP)
        self.barbs_check.toggled.connect(self._on_barbs_toggled)
        row2.addWidget(self.barbs_check)

        # R9: shaded relief under the map. Off by default -- it is a backdrop, and a
        # backdrop that appears unasked competes with the field -- and remembered, because
        # a reader who wants the hills wants them on every map, not on this one.
        self.topo_check = QtWidgets.QCheckBox('  Topography')
        self.topo_check.setEnabled(terrain.available())
        self.topo_check.setToolTip(
            TOPO_TOOLTIP if terrain.available() else
            'The bundled elevation grid (imsicon/mapdata/levant_etopo1.npz) is missing, '
            'so there is no relief to draw.')
        remembered = str(self.settings.value('display/topography', 'false')).lower()
        self.topo_check.setChecked(terrain.available() and remembered in ('true', '1'))
        self.topo_check.toggled.connect(self._on_topo_toggled)
        row2.addWidget(self.topo_check)

        # R5. Isolines are on by default where a field has them -- they are what makes a
        # smooth colour ramp readable as numbers -- while Sort is off, because it hides
        # part of the map and that has to be asked for.
        self.isolines_check = QtWidgets.QCheckBox('  Isolines')
        self.isolines_check.setChecked(True)
        self.isolines_check.setEnabled(False)
        self.isolines_check.toggled.connect(self._on_isolines_toggled)
        row2.addWidget(self.isolines_check)

        # R5.9: how close the lines are. A slider rather than a combo because the choice
        # is one-dimensional and ordered -- "closer" and "further apart" is the whole of
        # it -- and because a forecaster tries two or three spacings against one frame
        # before settling, which is a drag rather than three trips through a menu.
        self.isoline_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.isoline_step_slider.setRange(0, len(isolines.STEP_CHOICES) - 1)
        self.isoline_step_slider.setValue(isolines.STEP_CHOICES.index(1.0))
        self.isoline_step_slider.setSingleStep(1)
        self.isoline_step_slider.setPageStep(1)
        self.isoline_step_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.isoline_step_slider.setTickInterval(1)
        self.isoline_step_slider.setFixedWidth(96)
        self.isoline_step_slider.setEnabled(False)
        self.isoline_step_slider.valueChanged.connect(self._on_isoline_step_changed)
        row2.addWidget(self.isoline_step_slider)
        # Fixed width and left-aligned, for the reason the readout's numbers are: this
        # label changes while the slider is being dragged, and a label that resizes as it
        # changes drags the whole toolbar row about under the cursor.
        self.isoline_step_label = QtWidgets.QLabel('')
        self.isoline_step_label.setMinimumWidth(64)
        row2.addWidget(self.isoline_step_label)

        # R15: each line's value written on it. Off by default -- text over a map is
        # something to ask for -- and remembered, like Topography, because a reader who
        # wants the numbers on the lines wants them on every map. Live only while the
        # lines themselves are, for the reason the spacing slider is.
        self.isoline_labels_check = QtWidgets.QCheckBox('  Values')
        self.isoline_labels_check.setEnabled(False)
        remembered = str(self.settings.value('display/isoline_labels', 'false')).lower()
        self.isoline_labels_check.setChecked(remembered in ('true', '1'))
        self.isoline_labels_check.toggled.connect(self._on_isoline_labels_toggled)
        self.map.set_isoline_labels(self.isoline_labels_check.isChecked())
        row2.addWidget(self.isoline_labels_check)

        self.sort_check = QtWidgets.QCheckBox('  Sort')
        self.sort_check.setChecked(False)
        self.sort_check.setEnabled(False)
        self.sort_check.toggled.connect(self._on_sort_toggled)
        row2.addWidget(self.sort_check)

        # R9: the right-hand panel as a vertical profile. Enabled on a column of pressure
        # levels, ticked by default on relative humidity -- the field it was asked for --
        # and remembered per field within the window, like the isoline spacing.
        self.profile_check = QtWidgets.QCheckBox('  Profile')
        self.profile_check.setChecked(False)
        self.profile_check.setEnabled(False)
        self.profile_check.toggled.connect(self._on_profile_toggled)
        row2.addWidget(self.profile_check)

        # R10: the configured points, as dots on the map. Live whenever the settings file
        # has any; ticked as the file says (`show_points`), and the tick lasts the session.
        self.points_check = QtWidgets.QCheckBox('  Points')
        self.points_check.setChecked(bool(self.config.show_points))
        self.points_check.toggled.connect(self.map.show_points)
        row2.addWidget(self.points_check)

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
        # Left/right walk time; up/down walk the second axis. On a pressure-level file that
        # is the column -- up goes higher into the atmosphere, which is not the same as
        # "the next index", because a file may store its levels either way round.
        for keys, delta in (('Up', 1), ('Down', -1)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.activated.connect(lambda d=delta: self.step_level(d))

    # ---- F1: opening files -----------------------------------------------------
    def open_dialog(self):
        start = self.settings.value('last_dir', '')
        if not start or not Path(start).exists():
            local = Path.cwd() / 'data'
            start = str(local if local.exists() else Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open an IMS ICON file', start, FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path):
        path = Path(path)
        if not path.exists():
            self._error(f'File not found:\n{path}')
            return
        self.settings.setValue('last_dir', str(path.parent))
        self._remember_root(path)
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
        if target is None:
            self._sync_field_combo()        # cancelled: name the field still on screen
            return
        self._load(Path(target))

    def _on_decompress_failed(self, message):
        if self._progress is not None:
            self._progress.reset()
            self._progress = None
        self._error(f'Could not decompress the file:\n{message}')
        self._sync_field_combo()

    def _remember_root(self, path):
        """Note the directory a file came from, newest first.

        Capped, so a long session cannot turn the field scan into a walk of everywhere the
        user has ever browsed.
        """
        directory = Path(path).parent
        self._roots = [directory] + [r for r in self._roots if r != directory]
        del self._roots[8:]

    def _search_roots(self):
        """Where to look for the other maps of this run.

        `ingest.search_roots` covers the caches, ./data and whatever is beside the open
        file; the directories the user actually opened from come first, and stay in the
        list after a `.nc.bz2` has been expanded into the cache.
        """
        near = self.base_ds.path if self.base_ds is not None else None
        return list(dict.fromkeys(self._roots + ingest.search_roots(near)))

    def _load(self, path):
        try:
            raw = EnsembleFile(path)
            ds = FieldView(raw)
            self._apply_initial_units(ds)
        except nc3.UnsupportedFormat as exc:
            self._error(str(exc))
            self._sync_field_combo()
            return
        except Exception as exc:
            self._error(f'Could not open {Path(path).name}:\n{exc}')
            self._sync_field_combo()
            return
        self.opened[Path(path)] = ds
        self.base_ds = ds
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
        self.level = ds.axis.default_index
        self.stack.setCurrentIndex(1)
        self.setWindowTitle(f'IMS ICON Ensemble Viewer - {title}')
        self.status_left.setText(ds.summary())

        self.map.set_dataset(ds, geo.overlay_for(near or ds.path))
        self.plot.set_dataset(ds)
        self.profile.set_dataset(ds)
        self.readout.configure(ds)
        # Before the colormap, which can trigger the first redraw: `refresh_map` reads the
        # aggregation combo, and the previous file's mode must not draw this file's frame.
        self._sync_agg_combo(ds)
        self._sync_level_combo(ds)
        self._sync_scale_combo(apply_default=True)
        self._sync_colormap(ds)

        self._sync_field_combo()
        self._sync_units_combo()
        self._sync_rate_combo()
        # Cleared as well as set: this runs again for every field the user opens, and a
        # warning left over from the previous one would be pointing at nothing.
        self._show_standing_note()

        self.save_action.setEnabled(True)
        # An overlay built for another shape -- another run, another axis, another grid --
        # cannot be drawn over this one, so it is dropped rather than silently reused.
        if self._overlay_key is not None and self._overlay_key != self._overlay_shape():
            self.wind_overlay = None
            self._overlay_key = None
        self._sync_barbs_check()
        self._sync_isolines_check()
        self._sync_sort_check()
        self._sync_topo()
        self._sync_profile_check()
        self._sync_height_companion()

        self.slider.blockSignals(True)
        self.slider.setRange(0, ds.n_times - 1)
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.slider.blockSignals(False)

        self._ensure_range()

        iy, ix = ds.ny // 2, ds.nx // 2
        self.select_point(iy, ix)
        self.set_time(0)

    def _standing_note(self):
        """The warning this file carries, if any: `(short text, full text)`.

        Two of them, and both are about a file whose header does not say what it seems to:
        one stops early (G26), the other could not name its own levels. Held as a method
        rather than written into the status bar once, because the bar is also used for
        transient messages -- and a transient one must put the standing warning BACK when
        it is done, not blank it (`_on_scan_done`, where the truncation warning used to
        vanish for any file whose range had not been cached yet).
        """
        notes = [note for note in
                 (getattr(self.ds, 'truncation_note', None),
                  getattr(self.ds, 'axis_note', None)) if note]
        if not notes:
            return '', ''
        short = ('\u26a0 incomplete file'
                 if getattr(self.ds, 'truncation_note', None) else '\u26a0 levels assumed')
        return short, '\n\n'.join(notes)

    def _show_standing_note(self):
        short, full = self._standing_note()
        self.status_right.setText(short)
        self.status_right.setToolTip(full)

    # ---- the "Map shows" field selector -------------------------------------------
    def _field_entries(self):
        """-> [(key, label, target)] for the Map shows combo.

        **Every map of this run that is on disk is listed**, not only the file that was
        opened. Downloading four maps and then being able to look at one of them is not a
        viewer, and "open the other one again" is a file dialog the user should not have
        to visit to compare two fields of the same forecast.

        `target` is the `Path` of a file-backed entry (key `file:<FIELD>`), the
        `DerivedRequest` of a computed one, or None for the field already on screen
        (key `base`). The derived entries are checked against the files actually on disk
        rather than offered blindly, because a menu entry that always fails is worse than
        one that is not there. Whatever is currently on screen is always listed, even when
        it is an ad-hoc `A - B` the standard entries do not cover -- otherwise the combo
        would name one field while the map shows another.

        One run only: fields of another run share neither the valid times nor, in
        principle, the grid, so offering them here would be offering a comparison the
        rest of the app is careful to refuse (v3 R3.6).
        """
        if self.base_ds is None:
            return []
        run = f'{self.base_ds.run_init:%Y%m%d%H}'
        family = self.base_ds.family or products.ENSEMBLE
        available = ingest.scan_for_fields(self._search_roots())
        on_disk = ingest.fields_of_run(available, family, run)
        # The open file itself may sit somewhere the scan does not look.
        on_disk.setdefault(self.base_ds.field, self.base_ds.path)

        entries = []
        for field in sorted(on_disk, key=derivedialog.field_sort_key):
            if field == self.base_ds.field:
                entries.append(('base', _field_label(field, self.base_ds, family), None))
            else:
                entries.append((f'file:{field}',
                                _field_label(field, family=family), on_disk[field]))

        # The derived entries ask the FAMILY which field plays each role: the ensemble's
        # dew point comes from T_2M and RELHUM_2M, the deterministic run's from t_2m and
        # rh_2m -- and the deterministic run also publishes td_2m itself, in which case
        # the depression below is a difference of two files rather than a computation.
        needed = [derived.role_field(family, role)
                  for role in ('temperature', 'humidity')]
        if all(field and field in on_disk for field in needed):
            paths = [on_disk[field] for field in needed]
            for kind, label in ((derivedialog.DEW_POINT,
                                 f'{derived.DEW_POINT_FIELD} - dew point'),
                                (derivedialog.DEPRESSION,
                                 f'{derived.DEPRESSION_NAME} - dew point depression')):
                entries.append((kind, label,
                                derivedialog.DerivedRequest(kind, paths, label)))

        # A run can offer two wind maps: the 10 m components, and (deterministic only)
        # the 3-D ones, which is the wind at whatever level the Level control is on.
        for pair in derived.wind_pairs_in(family, on_disk):
            upper = pair == (family.roles.get('zonal_upper'),
                             family.roles.get('meridional_upper'))
            kind = derivedialog.WIND_UPPER if upper else derivedialog.WIND
            label = (f'{derived.UPPER_WIND_FIELD} - wind speed + barbs (pressure levels)'
                     if upper else f'{derived.WIND_FIELD} - wind speed + barbs')
            entries.append((kind, label, derivedialog.DerivedRequest(
                kind, [on_disk[field] for field in pair], label)))

        current = getattr(self.ds, 'derived_kind', 'base')
        if current != 'base' and not any(key == current for key, _l, _t in entries):
            entries.append((current,
                            f'{self.ds.display_name} - {self.ds.long_name}'[:60],
                            getattr(self.ds, 'derived_request', None)))
        return entries

    def _sync_field_combo(self):
        """Rebuild the field list and select whatever is actually on screen."""
        current = getattr(self.ds, 'derived_kind', 'base')
        entries = self._field_entries()
        self._field_requests = {key: target for key, _label, target in entries
                                if not key.startswith('file:')}
        self._field_files = {key: target for key, _label, target in entries
                             if key.startswith('file:')}
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for key, label, _target in entries:
            self.field_combo.addItem(label, key)
        self.field_combo.setCurrentIndex(max(0, self.field_combo.findData(current)))
        self.field_combo.setEnabled(self.field_combo.count() > 1)
        self.field_combo.blockSignals(False)
        self.field_combo.setToolTip(
            f'{(self.base_ds.family or products.ENSEMBLE).title} maps of run '
            f'{self.base_ds.run_init:%Y-%m-%d %H}Z found beside the open file, in ./data '
            'or in the download cache, plus the fields that can be derived from them.'
            if self.base_ds is not None else '')

    def _on_field_changed(self, _index):
        key = self.field_combo.currentData()
        if key is None or self.base_ds is None:
            return
        if key == 'base':
            if self.ds is not self.base_ds:
                self._install(self.base_ds, self.base_ds.path.name,
                              near=self.base_ds.path)
            return
        if key.startswith('file:'):
            path = self._field_files.get(key)
            if path is None:
                self._sync_field_combo()    # nothing to open: put the label back
                return
            self._open_field_file(path)
            return
        request = self._field_requests.get(key)
        if request is None:
            self._sync_field_combo()        # nothing to rebuild it from: put the label back
            return
        self._start_derive(request)

    def _open_field_file(self, path):
        """Switch the map to another field of this run that is already on disk.

        A view this window has opened before is reinstalled rather than mapped again, so
        flipping between the maps of one run is free after the first look at each; a
        `.nc.bz2` still has to be expanded, on the same worker `Open...` uses.

        Unlike `open_path` this does not move the Open dialog's remembered directory:
        the file usually comes from the download cache, which is not where the user
        browses for the next one.
        """
        path = Path(path)
        self._remember_root(path)
        existing = self.opened.get(path)
        if existing is not None:
            self.base_ds = existing
            self._install(existing, path.name, near=path)
        elif ingest.is_compressed(path):
            self._start_decompress(path)
        else:
            self._load(path)

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
        # so the download lands the user in the viewer, and point at where the rest are:
        # they are all in the "Map shows" list, which is not obvious from a status line
        # that only names files.
        if len(fetched) > 1:
            names = ', '.join(_field_of(path) for path in fetched)
            self.status_right.setText(
                f'downloaded {len(fetched)} maps ({names}) - choose between them under '
                '"Map shows"')
        self.open_path(fetched[0])

    def derive_dialog(self):
        """Dew point, dew point depression, or an A-B difference map."""
        near = self.ds.path if self.ds is not None else None
        run = f'{self.ds.run_init:%Y%m%d%H}' if self.ds is not None else None
        family = getattr(self.base_ds, 'family', None) if self.base_ds is not None else None
        dialog = derivedialog.DerivedDialog(self, near=near, run=run,
                                            roots=self._search_roots(), family=family)
        if not dialog.available:
            self._error('No ICON ensemble files were found beside the open file, in '
                        './data, or in the cache. Open or download a run first.')
            return
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        request = dialog.request()
        if request is not None:
            self._start_derive(request)

    def _start_derive(self, request):
        """Build a derived view off the UI thread -- opening a second file can be 16 s."""
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
        self._apply_initial_units(view)
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
        family = getattr(self.ds, 'family', None) or products.ENSEMBLE
        name = family.local_name(run, ncwrite.nc_variable_name(self.ds.field),
                                 compressed=False)
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
        """A difference wants a diverging ramp; anything else wants a sequential one.

        Only the *combo* is set here. What reaches the map goes through `_apply_colormap`,
        which is also what the sort scale overrides -- so there is one place that decides
        the colours, and choosing a field cannot quietly undo the sort band.
        """
        diverging = getattr(ds, 'diverging', False)
        # The map's own line in the file, then the file's scale for every map of its
        # kind (R12), then the usual one.
        wanted = (colors.ramp_named(self._defaults(ds).colours)
                  or colors.ramp_named(self.config.difference_colours if diverging
                                       else self.config.map_colours)
                  or (DIVERGING_DEFAULT if diverging else SEQUENTIAL_DEFAULT))
        if self.cmap_combo.currentText() != wanted:
            self.cmap_combo.setCurrentText(wanted)      # fires _on_cmap_changed
        else:
            # The new field decides the transparency, and only `refresh_map` knows where
            # its scale lands -- so drop the cached state and let the next frame rebuild.
            self._cmap_state = None

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
        self._show_standing_note()
        if result is not None and self.ds is not None:
            self._apply_range(self.ds.value_range)

    def _apply_range(self, value_range):
        """G46: the range can be None by the time it arrives, and unpacking None crashes.

        `_on_scan_done` re-reads the range from the VIEW rather than trusting the value
        the worker carried, because a scan is per transform signature (G19) -- and if the
        signature moved on while the scan was running (the Rate combo, mostly), the view
        has no range for the new one yet and answers None. Seen for real when `--rate 3h`
        and `--barbs on` were given together: opening the wind held the event loop long
        enough for the two to cross. Doing nothing is right -- `_ensure_range` has already
        started the scan for the signature that is now on screen.
        """
        if value_range is None:
            return
        lo, hi = value_range
        self.readout.span = hi - lo
        self._apply_graph_range(value_range)
        self.refresh_map()

    def _apply_graph_range(self, value_range=None):
        """The graph's value axis: pinned to the dataset max (A1), or to the fixed range
        the settings file gives this map while Scale says Fixed (R10) -- one scale for
        the colours and the curves, so a reading on one is a reading on the other."""
        fixed = self._active_fixed_range()
        if fixed is not None:
            lo, hi = fixed
        elif value_range is not None:
            lo, hi = value_range
            lo = min(0.0, lo) if lo >= 0 else lo
        else:
            return
        self.plot.set_yrange(lo, hi)
        self.profile.set_xrange(lo, hi)

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
            frame = self.ds.frame(self.t, self.level)
            what = self.ds.level_label(self.level)
        else:
            frame = self.ds.agg_frame(self.t, mode)
            what = self.agg_combo.currentText()
        sort = self.sort_scale(mode)
        self._sync_scale_controls(sort)
        scale = self.scale_combo.currentIndex()
        # R10: a fixed range from the settings file. Not on `spread`, which is a width
        # across the members and starts at zero whatever the field's own range is.
        fixed = self._active_map_range() if mode != 'spread' else None
        if sort is not None:
            # The band IS the scale: fixed, so that a cell's colour means the same
            # depression in every frame and at every time step, which is the whole
            # premise of reading it as "under 2 degrees" rather than as "reddest here".
            lo, hi = sort.levels
        elif fixed is not None:
            lo, hi = fixed
        elif scale != SCALE_FRAME and self.ds.value_range is not None:
            lo, hi = self.ds.value_range
            if mode == 'spread':
                lo, hi = 0.0, _finite_max(frame, 1.0)
            elif not self.ds.axis.aggregatable:
                # A column spans the whole troposphere, so one level coloured against the
                # file's range is a single flat shade. Each level has its own cached range
                # (`EnsembleFile.level_range`): still fixed while time is scrubbed, which
                # is what R1's "Dataset range" is for, but fixed to something with contrast.
                per_level = self.ds.level_range(self.level)
                if per_level is not None:
                    lo, hi = per_level
        else:
            lo, hi = _finite_min(frame, 0.0), _finite_max(frame, 1.0)
        if (sort is None and fixed is None and getattr(self.ds, 'diverging', False)
                and mode != 'spread'):
            # A difference map has to be symmetric about zero, or the colour that means
            # "no difference" moves with the data and +2 K reads as the same colour as
            # -2 K did a frame earlier. `spread` is excluded: it is non-negative already.
            lo, hi = _symmetric(lo, hi)
        self._apply_colormap(sort, transparent_zero=_zero_is_the_floor(lo, hi))
        units = f' [{self.ds.units}]' if self.ds.units else ''
        interval = (getattr(self.ds, 'isolines', None)
                    if self.isolines_check.isChecked() else None)
        self.map.set_frame(frame, (lo, hi), interval=interval)
        # After the frame, never before: the title names the interval the lines were
        # actually drawn at, which `levels_for` may have coarsened (G35).
        notes = [note for note in (self._push_wind(mode), self._isoline_note(interval),
                                   self._sort_note(sort), self._terrain_note()) if note]
        self.map.set_title('  |  '.join(
            [f'{self.ds.display_name}{units} - {what} - {self.ds.label_for(self.t)}']
            + notes))

    def _isoline_note(self, interval):
        """`isolines 1 °C`, from what the map drew rather than from what was asked for."""
        if interval is None or not len(self.map.isoline_levels):
            return ''
        units = f' {self.ds.units}' if self.ds.units else ''
        drawn = f'isolines {self.map.isoline_step:g}{units}'
        if self.map.isoline_step > interval.step * 1.000001:
            drawn += f' (too many lines at {interval.step:g}{units})'
        return drawn

    def _terrain_note(self):
        """`terrain shading` -- said in the title because the shadows change what a colour
        looks like, and a reader of a screenshot has to know they are not the field."""
        return 'terrain shading' if self.map.terrain_drawn else ''

    def _sort_note(self, sort):
        """`sorted: colour only below 2 °C`. What the colours mean is on the colorbar
        beside it and in the tooltip; a title is a label, not a legend."""
        if sort is None:
            return ''
        units = f' {self.ds.units}' if self.ds.units else ''
        return f'sorted: colour only below {sort.top:g}{units}'

    def sort_scale(self, mode=None):
        """The R5 sort band if it applies to what is on screen, else None.

        `spread` is excluded for the reason it is excluded from the symmetric scale: a
        max-minus-min across the members is a width, not a depression, so colouring it
        against the fog thresholds would read as a forecast of fog that nobody made.
        """
        if self.ds is None or not self.sort_check.isChecked():
            return None
        if (mode or self.agg_combo.currentData()) == 'spread':
            return None
        return getattr(self.ds, 'sort_scale', None)

    def _apply_colormap(self, sort, transparent_zero=False):
        """Push the colour scale the view and the options ask for, when it has changed.

        `transparent_zero` fades the bottom of the ramp out (`ui/colors.py`) and is passed
        in rather than worked out here, because only `refresh_map` knows where the
        colorbar's low end ended up -- the dataset range, this frame's range, a symmetric
        difference or the sort band all put it somewhere different.

        Guarded by the key rather than by call order: `refresh_map` runs on every frame
        of a scrub, and rebuilding a lookup table 121 times to arrive at the same colours
        is the sort of thing that only shows up as "the slider feels heavy".
        """
        if sort is None:
            name = self.cmap_combo.currentText()
            # A diverging ramp is left alone: its centre is a reading ("no difference"),
            # not an absence, and its neutral colour is already pale.
            sequential = name not in DIVERGING_MAPS
            # The custom ramp's colours are in the key: the same name can mean another
            # ramp after Settings... is saved.
            key = ('map', name, sequential, sequential and transparent_zero,
                   colors.custom_stops() if name == colors.CUSTOM else None)
        else:
            key = ('sort', round(sort.top, 6))
        if key == self._cmap_state:
            return
        self._cmap_state = key
        if sort is None:
            self.map.set_colormap(colors.map_colormap(key[1], transparent_zero=key[3],
                                                      punchy=key[2]))
        else:
            self.map.set_colormap(colors.with_transparent_top(*sort.positions()))

    # ---- wind barbs ------------------------------------------------------------
    def _push_wind(self, mode):
        """Hand the map the vectors behind what it is showing. -> a label, or ''.

        Duck-typed on purpose: any view that can produce `wind_vectors` gets barbs, and
        neither this method nor `MapView` ever learns what a `WindView` is -- the same
        arrangement that lets a derived field be an ordinary dataset everywhere else.

        The label is not decoration. `mean` barbs are the mean *vector* while the colours
        under them are the mean *speed*, and `spread` has no direction at all, so the map
        has to say which of those the feathers are counting.
        """
        source, overlay = self.wind_source()
        if source is None or not self.barbs_check.isChecked():
            self.map.set_wind(None)
            return ''
        vectors = source.wind_vectors
        # `t` and the axis position are read when the map asks, not captured now, so a
        # wheel zoom between two redraws still draws the step that is on screen.
        self.map.set_wind(lambda rows, cols: vectors(self.t, mode, self.level,
                                                     rows, cols))
        return source.barb_label(mode, over=self.ds.display_name if overlay else None)

    def _on_isolines_toggled(self, checked):
        """The tick decides whether the lines are drawn at all; the slider follows it,
        because a spacing control that is live while nothing is spaced is a control that
        does nothing when you move it."""
        if self.ds is not None:
            self._iso_on[self.ds.field] = bool(checked)
        self._iso_last_choice = bool(checked)
        self._sync_isoline_step_slider()
        self.refresh_map()

    def _on_isoline_labels_toggled(self, on):
        self.settings.setValue('display/isoline_labels', bool(on))
        self.map.set_isoline_labels(on)

    def _on_isoline_step_changed(self, index):
        """R5.9: the chosen spacing, in canonical degrees, pushed onto the view.

        Onto the *view*, not into a variable here, for the reason every other transform
        lives there: the map, the title and the tooltip then all read one number, and the
        affine that turns 2 °C into 3.6 °F is applied in exactly one place (G15).
        """
        if self.ds is None:
            return
        ladder = _ladder_of(self.ds)
        index = max(0, min(int(index), len(ladder.steps) - 1))
        step = ladder.canonical(ladder.steps[index])
        if not self.ds.set_isoline_step(step):
            return
        self._iso_steps[self.ds.field] = step
        self._sync_isoline_step_slider()
        self.refresh_map()

    def _sync_isolines_check(self):
        """Disabled, not hidden -- the same rule Rate and Wind barbs follow.

        The spacing this window remembers for the field is pushed onto the view first, so
        that flipping between T_2M and T-Td under "Map shows" gives each of them back the
        spacing it was last read at rather than the other one's.
        """
        field = getattr(self.ds, 'field', None)
        defaults = self._defaults()
        remembered = self._iso_steps.get(field)
        if (remembered is None and defaults.isoline_step is not None
                and getattr(self.ds, 'isolines', None) is not None):
            # R10: the configured spacing, snapped to a notch the slider can show -- in the
            # field's natural unit (degrees, ft), or the file's units on a field the file
            # itself made contourable.
            ladder = _ladder_of(self.ds)
            remembered = ladder.canonical(ladder.nearest(defaults.isoline_step))
        if remembered is not None and self.ds is not None:
            self.ds.set_isoline_step(remembered)
        interval = getattr(self.ds, 'isolines', None)
        if interval is not None:
            # On or off: this session's choice for the field, then the settings file, then
            # whatever was last chosen on any field (R5: the tick carries across), then on.
            wanted = self._iso_on.get(field)
            for fallback in (defaults.isolines, self._iso_last_choice, True):
                if wanted is None:
                    wanted = fallback
            if wanted != self.isolines_check.isChecked():
                self.isolines_check.blockSignals(True)
                self.isolines_check.setChecked(bool(wanted))
                self.isolines_check.blockSignals(False)
        self.isolines_check.setEnabled(interval is not None)
        self.isolines_check.setToolTip(_isoline_tooltip(self.ds))
        self._sync_isoline_step_slider()

    def _sync_isoline_step_slider(self):
        """Put the slider where the view is contoured and say the spacing on screen.

        The label is in DISPLAY units while the slider's notches are canonical degrees,
        which is not a contradiction but the point of it: one notch is one reading of the
        map, and the label says what that reading is called in the units the colorbar and
        the readout are using.
        """
        interval = getattr(self.ds, 'isolines', None)
        canonical = getattr(self.ds, 'isoline_step', None) if self.ds is not None else None
        live = interval is not None and self.isolines_check.isChecked()
        self.isoline_step_slider.setEnabled(live)
        self.isoline_labels_check.setEnabled(live)
        self.isoline_labels_check.setToolTip(
            ISOLINE_LABELS_TOOLTIP if live else
            'Write each isoline\'s value on it. Tick Isolines first'
            + ('' if interval is not None else
               ' -- and this map has none (see the Isolines tooltip)') + '.')
        # The notches are the FIELD's ladder (R9): five degree spacings on a temperature,
        # seven gpm spacings on a height chart. Re-ranged under blocked signals, because
        # a shorter range clamps the value and would otherwise fire a choice nobody made.
        ladder = _ladder_of(self.ds)
        self.isoline_step_slider.blockSignals(True)
        self.isoline_step_slider.setRange(0, len(ladder.steps) - 1)
        if canonical:
            # blockSignals, not a guard on the value: a field whose registry interval is
            # already where the slider sits would otherwise leave the view unset while
            # `_iso_steps` says it was chosen.
            self.isoline_step_slider.setValue(ladder.index_of(canonical))
        self.isoline_step_slider.blockSignals(False)
        units = f' {self.ds.units}' if self.ds is not None and self.ds.units else ''
        self.isoline_step_label.setText(
            f'{interval.step:g}{units}' if live and interval is not None else '')
        tip = _isoline_step_tooltip(self.ds, live)
        self.isoline_step_slider.setToolTip(tip)
        self.isoline_step_label.setToolTip(tip)

    def _sync_sort_check(self):
        """Sort follows the field: available on T-Td, and cleared on anything else.

        Cleared rather than left ticked-but-disabled, because a disabled tick reads as
        "this is on and you cannot change it", which is the opposite of what it means.
        """
        available = getattr(self.ds, 'sort_scale', None) is not None
        self.sort_check.blockSignals(True)
        self.sort_check.setEnabled(available)
        if not available:
            self.sort_check.setChecked(False)
        self.sort_check.blockSignals(False)
        self.sort_check.setToolTip(_sort_tooltip(self.ds))

    def _sync_scale_controls(self, sort):
        """While the sort band is on it IS the colour scale, so the two controls that
        would otherwise claim to set one are disabled instead of silently ignored."""
        for widget, what, idle in ((self.cmap_combo, 'colours', 'Colour ramp for the map'),
                                   (self.scale_combo, 'range',
                                    'Colour the whole dataset range, or just this frame')):
            widget.setEnabled(sort is None)
            # Restored as well as set, for the reason the truncation warning is cleared in
            # `_install`: a tooltip left over from a state the app is no longer in is
            # worse than none at all.
            widget.setToolTip(idle if sort is None else
                              f'Sort sets the {what}: red at 0 to white at '
                              f'{sort.top:g}. Untick Sort to choose again.')

    def _on_sort_toggled(self, _on):
        self.refresh_map()

    # ---- R10: the settings file -------------------------------------------------------
    def _defaults(self, ds=None):
        """What the settings file says this view opens with (`config.EMPTY` if nothing)."""
        return self.config.for_view(ds if ds is not None else self.ds)

    def _install_config_isolines(self):
        """Fields the file gives a spacing for but the app does not contour become
        contourable, in their file's own units (`isolines.set_custom`)."""
        steps = self.config.custom_isolines()
        units = {key: transform.UNITS[key].expected[0] for key in steps
                 if key in transform.UNITS and transform.UNITS[key].expected}
        isolines.set_custom(steps, units)

    def _apply_points(self):
        unknown = self.map.set_points(self.config.points,
                                      visible=self.points_check.isChecked())
        self.points_check.setEnabled(bool(self.config.points))
        count = len(self.config.points)
        self.points_check.setToolTip(
            f'Show the {count} point{"s" if count != 1 else ""} from the settings file '
            'as dots on the map' if count else
            'No points are set. Add them under Settings... (latitude, longitude, colour).')
        if unknown:
            self.config.warnings.append(
                'unknown point colour(s) ' + ', '.join(repr(c) for c in unknown)
                + '; drawn in blue')

    def _sync_ramp_list(self):
        """The Colours combo offers the custom ramp only while the file defines one."""
        wanted = colors.ramps()
        have = [self.cmap_combo.itemText(i) for i in range(self.cmap_combo.count())]
        if have == wanted:
            return
        current = self.cmap_combo.currentText()
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.clear()
        self.cmap_combo.addItems(wanted)
        self.cmap_combo.setCurrentText(current if current in wanted else wanted[0])
        self.cmap_combo.blockSignals(False)

    def _report_config_problems(self):
        """A broken line in the settings file costs that line, and is said out loud."""
        for key, value in (('colours', self.config.map_colours),
                           ('difference_colours', self.config.difference_colours)):
            if value and colors.ramp_named(value) is None:
                problem = (f'[display] {key} {value!r} is not one of '
                           f'{", ".join(colors.ramps())}; using the usual scale')
                if problem not in self.config.warnings:
                    self.config.warnings.append(problem)
        if self._custom_colour_problems:
            problem = ('[display] custom_colours: ' + ', '.join(
                repr(c) for c in self._custom_colour_problems) + ' not a colour; skipped'
                + ('' if colors.custom_stops() else ' (fewer than two left: no custom '
                   'scale)'))
            if problem not in self.config.warnings:
                self.config.warnings.append(problem)
        for defaults in self.config.fields.values():
            if defaults.colours and colors.ramp_named(defaults.colours) is None:
                problem = (f'[fields.{defaults.name}] colours {defaults.colours!r} is not '
                           f'one of {", ".join(colors.ramps())}; ignored')
                if problem not in self.config.warnings:
                    self.config.warnings.append(problem)
        if self.config.warnings:
            count = len(self.config.warnings)
            self.status.showMessage(
                f'\u26a0 settings file: {count} problem{"s" if count != 1 else ""} - '
                f'{self.config.warnings[0]}' + (' (and more: see Settings...)'
                                                 if count > 1 else ''), 15000)

    def _apply_initial_units(self, ds):
        """The units a view opens in: this session's choice for it, then the settings
        file, then the choice remembered from an earlier session (v2), else the
        registry's default. A label the field does not offer is passed over, and a
        configured one is said so rather than silently ignored."""
        configured = self._defaults(ds).units
        for label in (self._units_chosen.get(ds.field), configured,
                      self.settings.value(f'units/{ds.field}', None)):
            if label and ds.set_units(label):
                return
            if label and label == configured:
                offered = ', '.join(getattr(ds, 'unit_labels', []) or []) or 'none'
                self.statusBar().showMessage(
                    f'\u26a0 settings file: {ds.display_name} is not offered in '
                    f'{configured!r} (offered: {offered})', 10000)

    def _fixed_range(self, ds=None, which='scale'):
        """The settings file's fixed range for this view, in the units on screen, or None.

        `which` is 'scale' -- the range for the map AND the graph -- or 'map_scale', the
        map's colours only (R13), stated in the same units.

        Stated in the field's configured units (or `scale_units`), and converted when the
        Units combo says otherwise -- a CAPE scale of [0, 3000] or a temperature scale of
        [10, 40] degC has to stay the same colours in degF. None on a rate view, whose
        values are a different quantity from the stored ones the range was written for.
        """
        ds = ds if ds is not None else self.ds
        if ds is None or getattr(ds, 'rate_hours', 0):
            return None
        defaults = self._defaults(ds)
        wanted = defaults.fixed_range if which == 'scale' else defaults.map_scale
        if wanted is None:
            return None
        choices = list(getattr(ds, 'unit_choices', None) or [])
        source = defaults.scale_units or defaults.units or (choices[0].label if choices
                                                            else ds.units)
        return convert_range(wanted, source, choices,
                             getattr(ds, 'units_affine', None),
                             difference=getattr(ds, 'is_difference_view', False))

    def _active_fixed_range(self):
        """The GRAPH's fixed range while Scale says Fixed: the file's `scale`, if any."""
        if self.scale_combo.currentIndex() != SCALE_FIXED:
            return None
        return self._fixed_range()

    def _active_map_range(self):
        """The MAP's fixed range while Scale says Fixed: its own `map_scale` when the file
        gives one (R13), else the `scale` it shares with the graph."""
        if self.scale_combo.currentIndex() != SCALE_FIXED:
            return None
        return self._fixed_range(which='map_scale') or self._fixed_range()

    def _sync_scale_combo(self, apply_default):
        """Name the fixed range on its entry, and pick the entry the file asks for.

        `apply_default` on opening a map; a units or rate change only re-labels, so it
        never undoes the Scale a reader has just chosen.
        """
        graph = self._fixed_range()
        on_map = self._fixed_range(which='map_scale')
        fixed = on_map or graph
        item = self.scale_combo.model().item(SCALE_FIXED)
        item.setEnabled(fixed is not None)
        units = f' {self.ds.units}' if self.ds is not None and self.ds.units else ''
        if on_map is not None and graph is not None:
            text = (f'Fixed: map {on_map[0]:g} to {on_map[1]:g}, graph {graph[0]:g} to '
                    f'{graph[1]:g}{units}')
        elif on_map is not None:
            text = f'Fixed map {on_map[0]:g} to {on_map[1]:g}{units}'
        elif graph is not None:
            text = f'Fixed {graph[0]:g} to {graph[1]:g}{units}'
        else:
            text = 'Fixed (set one under Settings...)'
        item.setText(text)
        defaults = self._defaults()
        wanted = None
        if apply_default:
            if fixed is not None and (defaults.fixed_range is not None
                                      or defaults.map_scale is not None):
                wanted = SCALE_FIXED
            elif defaults.scale in user_config.SCALE_MODES:
                wanted = SCALE_DATASET if defaults.scale == 'dataset' else SCALE_FRAME
            elif self.scale_combo.currentIndex() == SCALE_FIXED:
                wanted = SCALE_DATASET     # the previous map's range is not this map's
        elif fixed is None and self.scale_combo.currentIndex() == SCALE_FIXED:
            wanted = SCALE_DATASET
        if wanted is not None and wanted != self.scale_combo.currentIndex():
            self.scale_combo.blockSignals(True)
            self.scale_combo.setCurrentIndex(wanted)
            self.scale_combo.blockSignals(False)

    def _on_scale_changed(self, _index):
        if self.ds is None:
            return
        if self.ds.value_range is not None:
            self._apply_range(self.ds.value_range)
        else:
            self._apply_graph_range()
            self.refresh_map()

    def settings_dialog(self):
        dialog = settingsdialog.SettingsDialog(self.config, current=self.ds, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.saved:
            self.apply_config(dialog.saved)

    def apply_config(self, cfg):
        """Use a newly saved configuration at once.

        Points and custom isolines straight away; for the map on screen, every default
        the reader has not overridden in this session -- units, colours, scale, isolines
        and profile -- so that "I set CAPE to inferno" is visible on pressing Save.
        """
        self.config = cfg
        self._install_config_isolines()
        self._custom_colour_problems = colors.set_custom_ramp(cfg.custom_colours)
        self._sync_ramp_list()
        self.set_time_zone(cfg.time_zone)
        self.points_check.blockSignals(True)
        self.points_check.setChecked(bool(cfg.show_points))
        self.points_check.blockSignals(False)
        self._apply_points()
        self._report_config_problems()
        ds = self.ds
        if ds is None:
            return
        if ds.field not in self._units_chosen:
            configured = self._defaults(ds).units
            if configured and configured != ds.units and ds.set_units(configured):
                self._sync_units_combo()
        self._sync_scale_combo(apply_default=True)
        self._sync_colormap(ds)
        self._sync_isolines_check()
        self._sync_profile_check()
        self._refresh_units()

    # ---- R11: Zulu or Israel time ------------------------------------------------------
    def set_time_zone(self, zone):
        """Rewrite every time on screen in another clock. Nothing is re-read or rescaled:
        the index, the slider and the data are the same instant either way."""
        zone = timefmt.set_zone(zone)
        index = self.tz_combo.findData(zone)
        if self.tz_combo.currentIndex() != index:
            self.tz_combo.blockSignals(True)
            self.tz_combo.setCurrentIndex(index)
            self.tz_combo.blockSignals(False)
        ds = self.ds
        if ds is None:
            return
        self.status_left.setText(ds.summary())
        self.plot.label_time_axis()
        self.readout.label_run(ds)
        self.time_label.setText(ds.label_for(self.t))
        self.refresh_map()                          # the map title carries the time
        self._update_readout(self.t, hovering=False)

    # ---- R8: the wind, over any map ------------------------------------------------
    def _overlay_shape(self):
        """What an overlay has to fit: the run, the axis, the grid AND the time axis.

        Compared rather than rebuilt on every field change, so flipping between two maps
        of one run keeps the barbs that are already open. `check_pairable` still has the
        last word when the view is built.

        `n_times` is in the key for a reason found on real data (**G45**): the run's `temp`
        and its `u`/`v` can hold DIFFERENT numbers of steps -- an interrupted download
        stops each file at its own point (G26) -- and barbs from a 5-step wind over a
        7-step map index past the end of the wind the moment the slider passes step 5.
        """
        if self.ds is None:
            return None
        return (f'{self.ds.run_init:%Y%m%d%H}', self.ds.axis.kind, self.ds.n_members,
                self.ds.n_times, self.ds.ny, self.ds.nx)

    def _overlay_request(self):
        """A `DerivedRequest` for the wind that fits what is on screen, or None.

        The pair is chosen by SHAPE (`derived.wind_pair_for`): over an 850 hPa temperature
        the barbs must be the wind at 850 hPa, which is the run's 3-D `u`/`v`, while over
        a surface or ensemble map they must be the 10 m pair. Anything else would draw a
        wind from the wrong place and look entirely normal.
        """
        if self.base_ds is None or self.ds is None:
            return None
        family = self.base_ds.family or products.ENSEMBLE
        run = f'{self.base_ds.run_init:%Y%m%d%H}'
        on_disk = ingest.fields_of_run(ingest.scan_for_fields(self._search_roots()),
                                       family, run)
        pair = derived.wind_pair_for(family, on_disk,
                                     levels=self.ds.axis.is_pressure)
        if pair is None:
            return None
        label = f'Wind barbs from {pair[0]} + {pair[1]}'
        return derivedialog.DerivedRequest(derivedialog.WIND,
                                           [on_disk[field] for field in pair], label)

    def wind_source(self):
        """The view whose vectors the barbs should use: the map's own, or the overlay.

        The overlay is only offered while it still fits the map on screen -- same run,
        axis, grid and number of steps -- so nothing downstream has to defend against
        being handed a wind from a different forecast.
        """
        if getattr(self.ds, 'wind_vectors', None) is not None:
            return self.ds, False
        if self.wind_overlay is not None and self._overlay_key == self._overlay_shape():
            return self.wind_overlay, True
        return None, False

    def _sync_barbs_check(self):
        """Enabled whenever barbs are POSSIBLE -- on the wind map, and over any other map
        of a run whose wind components are on disk."""
        own = getattr(self.ds, 'wind_vectors', None) is not None
        request = None if own else self._overlay_request()
        self.barbs_check.setEnabled(own or request is not None)
        if own:
            tooltip = BARB_TOOLTIP
        elif request is not None:
            fields = ' and '.join(path.name for path in request.paths)
            tooltip = (f'Draw the run\'s wind over the {self.ds.display_name} map.\n\n'
                       f'{BARB_TOOLTIP}\n\nThe barbs come from {fields}, so the colours '
                       'and the feathers are two different quantities -- the map title '
                       'says which wind they are.')
        else:
            tooltip = (f'{self.ds.display_name} has no direction to draw, and the wind '
                       'components of this run are not on disk. Download them '
                       '(Download... -> Select what the wind map needs) and they can be '
                       'drawn over any map.')
        self.barbs_check.setToolTip(tooltip)
        # The tick reflects what is actually DRAWN, so there is never a ticked box with no
        # barbs under it. A wind map is barbs by definition and comes up ticked (R4); an
        # overlay over someone else's map is opt-in, because building it opens two more
        # files -- so a plain field comes up unticked even where the wind is available,
        # and stays ticked once an overlay exists and still fits.
        source, _overlay = self.wind_source()
        self.barbs_check.blockSignals(True)
        self.barbs_check.setChecked(own or source is not None)
        self.barbs_check.blockSignals(False)

    def _on_barbs_toggled(self, on):
        """Ticking Wind barbs over a non-wind map opens the run's wind first."""
        source, _overlay = self.wind_source()
        if on and source is None and self._overlay_request() is not None:
            self._start_overlay()
            return
        self.refresh_map()

    def _start_overlay(self):
        """Open the wind pair off the UI thread: it can mean decompressing 2 x 262 MB."""
        request = self._overlay_request()
        if request is None or (self._overlay_builder is not None
                               and self._overlay_builder.isRunning()):
            return
        self.status_right.setText('opening the wind for the barbs...')
        self._overlay_builder = derivedialog.BuildWorker(request, dict(self.opened), self)
        self._overlay_builder.finished_view.connect(self._on_overlay_built)
        self._overlay_builder.failed.connect(self._on_overlay_failed)
        self._overlay_builder.start()

    def _on_overlay_built(self, view):
        """Accept the wind only if it actually fits the map it is going over.

        The two components pair with each OTHER inside `derived.wind`; that says nothing
        about whether they pair with the field on screen. Grid, run, time axis and level
        ladder all have to agree, or the barbs would be drawn from another forecast --
        and, when the wind is the shorter file, would run off the end of it (**G45**).
        """
        self._show_standing_note()
        if view is None:
            return
        try:
            derived.check_pairable(self.ds, view)
        except derived.PairError as exc:
            self._refuse_overlay(str(exc))
            return
        self.wind_overlay = view
        self._overlay_key = self._overlay_shape()
        self.refresh_map()

    def _refuse_overlay(self, message):
        """Say why the barbs cannot go over this map, and put the tick back.

        A status note rather than a modal: the barbs are a secondary thing the user asked
        for on top of a map that is fine, so this must not interrupt reading it.
        """
        self.wind_overlay = None
        self._overlay_key = None
        self.barbs_check.blockSignals(True)
        self.barbs_check.setChecked(False)
        self.barbs_check.blockSignals(False)
        self.barbs_check.setToolTip(
            f'The wind of this run cannot be drawn over {self.ds.display_name}:\n\n'
            f'{message}')
        self.status_right.setText('\u26a0 no barbs over this map')
        self.status_right.setToolTip(message)
        self.refresh_map()

    def _on_overlay_failed(self, message):
        """A failed overlay unticks the box rather than leaving it on with no barbs."""
        self._show_standing_note()
        self.barbs_check.blockSignals(True)
        self.barbs_check.setChecked(False)
        self.barbs_check.blockSignals(False)
        self._error(f'The wind barbs could not be drawn over this map:\n{message}')

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
        self._refresh_profile()
        self._update_readout(t, hovering=False)

    def select_point(self, iy, ix):
        """F3.3: a click on the map redirects the whole right-hand panel."""
        if self.ds is None:
            return
        self.point = (iy, ix)
        series = self.ds.series(iy, ix)
        self.plot.set_series(series)
        self.plot.set_level(self.level)
        lat, lon = float(self.ds.lat[iy]), float(self.ds.lon[ix])
        self.map.set_marker(lat, lon)
        self.readout.set_point(lat, lon)
        if self.ds.value_range is None:
            self.plot.set_yrange(float(np.nanmin(series)), float(np.nanmax(series)))
        self._refresh_profile()
        self._update_readout(self.t, hovering=False)

    def _update_readout(self, t, hovering, level=None):
        """The six ensemble statistics, or the column's reading at the chosen level.

        Which one is decided by the file's axis, in one place, so the panel and the
        numbers in it can never disagree about what they are describing. `level` lets a
        hover over the profile report another level than the map's, the way a hover over
        the time graph reports another time.
        """
        if self.ds is None or self.point is None or self.plot.series is None:
            return
        t = int(np.clip(t, 0, self.ds.n_times - 1))
        values = self.plot.series[t]
        position = self.level if level is None else int(level)
        if self.ds.axis.aggregatable:
            stats = member_stats(values)
        else:
            stats = level_stats(values, self.ds.member_labels, position)
            stats['height'] = self._height_at(t, position)
        self.readout.show_stats(self.ds.label_for(t), stats, hovering)

    # ---- R9: the vertical profile --------------------------------------------------------
    def _sync_profile_check(self):
        """Live on a column of pressure levels; ticked by default on relative humidity.

        Per field within the window, like the isoline spacing: a reader who switched the
        temperature back to the time graph should not find the humidity switched too.
        """
        ds = self.ds
        column = ds is not None and bool(getattr(ds.axis, 'is_pressure', False))
        remembered = self._profile_choices.get(ds.field) if ds is not None else None
        if remembered is None:
            remembered = self._defaults(ds).profile           # R10: the settings file
        wanted = column and (remembered if remembered is not None
                             else products.field_key(ds.field) == 'RH')
        self.profile_check.blockSignals(True)
        self.profile_check.setEnabled(column)
        self.profile_check.setChecked(bool(wanted))
        self.profile_check.blockSignals(False)
        self.profile_check.setToolTip(_profile_tooltip(ds))
        self._show_graph()

    def _on_profile_toggled(self, on):
        if self.ds is not None:
            self._profile_choices[self.ds.field] = bool(on)
        self._show_graph()

    @property
    def profile_shown(self):
        return self.graph_stack.currentIndex() == 1

    def _show_graph(self):
        """Put the profile or the time graph in front, and draw whichever it is."""
        column = self.ds is not None and bool(getattr(self.ds.axis, 'is_pressure', False))
        self.graph_stack.setCurrentIndex(1 if column and self.profile_check.isChecked()
                                         else 0)
        self._refresh_profile()

    def _refresh_profile(self):
        """The column at the chosen point and the current time, at its real heights."""
        if not self.profile_shown or self.ds is None or self.plot.series is None:
            return
        t = int(np.clip(self.t, 0, self.ds.n_times - 1))
        values = self.plot.series[t]
        heights = self._height_column(t)
        pressures = (np.asarray(self.ds.axis.values, dtype=float)
                     if self.ds.axis.is_pressure else None)
        units = self.height_companion.units if self.height_companion is not None else 'ft'
        self.profile.set_profile(values, heights, pressures, level=self.level,
                                 height_units=units)

    def _on_profile_hover(self, index):
        """A hover over the profile names a LEVEL, as one over the time graph names a time."""
        self._update_readout(self.t, hovering=index >= 0,
                             level=None if index < 0 else index)

    # ---- R9: the geopotential height beside the value ------------------------------------
    def _height_request(self):
        """A request for the run's geopotential, when the map is a column that is not the
        geopotential itself and the file is on disk; else None."""
        if self.base_ds is None or self.ds is None:
            return None
        if not getattr(self.ds.axis, 'is_pressure', False):
            return None
        if products.field_key(self.ds.field) == 'GEOPOT':
            return None
        family = self.base_ds.family or products.ENSEMBLE
        field = family.roles.get('height')
        if not field:
            return None
        run = f'{self.base_ds.run_init:%Y%m%d%H}'
        on_disk = ingest.fields_of_run(ingest.scan_for_fields(self._search_roots()),
                                       family, run)
        if field not in on_disk:
            return None
        return derivedialog.DerivedRequest(derivedialog.FIELD, [on_disk[field]],
                                           f'Heights from {field}')

    def _height_reason(self):
        """Why the height row reads `--`, for its tooltip."""
        if self.ds is None or not getattr(self.ds.axis, 'is_pressure', False):
            return 'Only a column of pressure levels has a level to give the height of.'
        family = (self.base_ds.family if self.base_ds is not None else None) \
            or products.ENSEMBLE
        field = family.roles.get('height') or 'geopot'
        return (f'The run\'s {field} is not on disk, so the height of this level is not '
                f'known. Download it (Download... -> {field}) and the height appears here.')

    def _sync_height_companion(self):
        """Open the run's geopotential beside a pressure-level map, if it is there.

        Automatic rather than opt-in, unlike the wind overlay: the height is a reading of
        the map on screen, not a second quantity drawn over it, and it was asked for next
        to the value. Off the UI thread all the same -- a `.nc.bz2` is 16 s.
        """
        request = self._height_request()
        if request is None:
            self.height_companion, self._height_key = None, None
            self.readout.set_height_source('', note=self._height_reason())
            return
        if self.height_companion is not None and self._height_key == self._overlay_shape():
            self.readout.set_height_source(self.height_companion.units, note='')
            return
        self.height_companion, self._height_key = None, None
        if self._height_builder is not None and self._height_builder.isRunning():
            return
        self.readout.set_height_source('', note='Opening the run\'s geopot for the heights...')
        self.status_right.setText('opening geopot for the heights...')
        self._height_builder = derivedialog.BuildWorker(request, dict(self.opened), self)
        self._height_builder.finished_view.connect(self._on_height_built)
        self._height_builder.failed.connect(self._on_height_failed)
        self._height_builder.start()

    def _on_height_built(self, view):
        """Accept the geopotential only if it fits the map on screen (G45 again)."""
        self._show_standing_note()
        if view is None or self.ds is None:
            return
        try:
            derived.check_pairable(self.ds, view)
        except derived.PairError as exc:
            self.readout.set_height_source('', note=(
                f'The run\'s {view.field} cannot be read against this map:\n\n{exc}'))
            return
        # The same units the geopot MAP would open in -- feet by default, or whatever the
        # Units combo was last set to on it -- so the height row, the profile axis and the
        # chart never disagree. The view is shared with "Map shows", so a later change on
        # the chart moves the row with it.
        self._apply_initial_units(view)
        self.opened.setdefault(Path(view.path), view)
        self.height_companion = view
        self._height_key = self._overlay_shape()
        self.readout.set_height_source(view.units, note='')
        self._refresh_profile()
        self._update_readout(self.t, hovering=False)

    def _on_height_failed(self, message):
        self._show_standing_note()
        self.readout.set_height_source('', note=f'The heights could not be read:\n{message}')

    def _height_column(self, t):
        """(n_levels,) geopotential heights at the chosen point and time, or None."""
        if self.height_companion is None or self.point is None or self.ds is None:
            return None
        try:
            series = self.height_companion.series(*self.point)
        except Exception:
            return None
        if series.ndim != 2 or series.shape[0] <= t or series.shape[1] != self.ds.n_members:
            return None
        return np.asarray(series[t], dtype=float)

    def _height_at(self, t, level):
        column = self._height_column(t)
        if column is None or not (0 <= int(level) < column.size):
            return None
        return float(column[int(level)])

    # ---- R9: shaded relief ---------------------------------------------------------------
    def _sync_topo(self):
        self.map.set_terrain(self.topo_check.isChecked())

    def _on_topo_toggled(self, on):
        self.settings.setValue('display/topography', bool(on))
        self._sync_topo()
        self.refresh_map()                  # the title says when the shading is on

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
            # A wind speed with no direction is half a reading, so the view is asked
            # for one whenever it has it -- including when the barbs are switched off.
            # One value read off the aggregated VECTOR, never a statistic of degrees,
            # which is the thing G16 forbids.
            direction = getattr(self.ds, 'direction_at', None)
            if direction is not None:
                iy, ix = self.ds.nearest_index(lat, lon)
                degrees = direction(self.t, iy, ix, self.agg_combo.currentData(),
                                    self.level)
                text += f'   from {degrees:.0f}°'
        self.status_right.setText(text)

    def _on_agg_changed(self):
        self._sync_level_enabled()
        self.refresh_map()

    # ---- the second axis: members, or pressure levels ------------------------------
    def _sync_agg_combo(self, ds):
        """Offer the aggregations the file's second axis actually supports.

        An ensemble gets all six. A column of pressure levels gets one entry -- the map
        shows a level at a time -- and the combo is disabled with a tooltip saying why,
        rather than silently dropping five entries the user might go looking for.
        """
        across = ds.axis.aggregatable
        choices = AGG_CHOICES if across else LEVEL_CHOICES
        self.agg_combo.blockSignals(True)
        self.agg_combo.clear()
        for label, key in choices:
            self.agg_combo.addItem(label, key)
        self.agg_combo.setCurrentIndex(0)
        self.agg_combo.setEnabled(across)
        self.agg_combo.blockSignals(False)
        self.agg_combo.setToolTip(
            'What the map shows across the 20 ensemble members' if across else
            f'{ds.display_name} is on {ds.axis.describe()}, not on ensemble members. A '
            'mean or a spread across a column of pressure levels is not a quantity '
            'anyone forecasts, so the map shows one level at a time.')

    def _level_entries(self, axis):
        """-> [(label, index)] in the order the combo lists them.

        A pressure axis is listed with the TOP OF THE ATMOSPHERE FIRST, so the list reads
        like a vertical profile and "up" means up in both senses -- up the list and up the
        column. It also means the Up key and the Up button agree with the combo's own
        keyboard behaviour instead of fighting it.
        """
        if axis.is_pressure:
            return [(axis.labels[i], i) for i in reversed(axis.upward)]
        return [(axis.labels[i], i) for i in range(axis.n)]

    def _sync_level_combo(self, ds):
        self.level_title.setText(f' {ds.axis.selector_label}: ')
        self.level_combo.blockSignals(True)
        self.level_combo.clear()
        for label, index in self._level_entries(ds.axis):
            self.level_combo.addItem(label, index)
        self.level_combo.setCurrentIndex(max(0, self.level_combo.findData(self.level)))
        self.level_combo.blockSignals(False)
        for widget in (self.level_combo, self.level_up, self.level_down):
            widget.setToolTip(_level_tooltip(ds))
        self.level_up.setToolTip(self.level_up.toolTip() + '\n\nUp arrow')
        self.level_down.setToolTip(self.level_down.toolTip() + '\n\nDown arrow')
        self._sync_level_enabled()

    def _sync_level_enabled(self):
        """The picker is live whenever the map is showing ONE position on the axis."""
        single = (self.ds is not None and self.ds.n_members > 1
                  and self.agg_combo.currentData() == 'member')
        for widget in (self.level_combo, self.level_up, self.level_down):
            widget.setEnabled(single)

    def _on_level_selected(self, _index):
        data = self.level_combo.currentData()
        if data is not None:
            self.set_level(int(data))

    def set_level(self, index):
        """Show one position on the second axis: `member 03`, or `850 hPa`."""
        if self.ds is None:
            return
        index = int(np.clip(index, 0, self.ds.n_members - 1))
        self.level = index
        position = self.level_combo.findData(index)
        if position >= 0 and self.level_combo.currentIndex() != position:
            self.level_combo.blockSignals(True)
            self.level_combo.setCurrentIndex(position)
            self.level_combo.blockSignals(False)
        self.plot.set_level(index)
        self.profile.set_level(index)
        if self.agg_combo.currentData() == 'member':
            self.refresh_map()
        self._update_readout(self.t, hovering=False)

    def step_level(self, delta):
        """Move one position UP (delta > 0) or DOWN the axis -- the up/down buttons.

        Up is up the atmosphere, not up the index: `LevelAxis.step` orders by pressure, so
        the key does the same thing whichever way round a file stores its levels.

        On an ensemble the map may be showing a mean rather than a member, and stepping
        "to the next member" while the map shows the mean would change nothing visible.
        So the aggregation switches to Single member first: the title then says which
        member is on screen, which is what the key was asking for.
        """
        if self.ds is None or self.ds.n_members < 2:
            return
        if self.agg_combo.currentData() != 'member':
            keys = [self.agg_combo.itemData(i) for i in range(self.agg_combo.count())]
            if 'member' not in keys:
                return
            self.agg_combo.setCurrentIndex(keys.index('member'))    # fires _on_agg_changed
        self.set_level(self.ds.axis.step(self.level, delta))

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
        self._units_chosen[self.ds.field] = label
        self._refresh_units()

    def _refresh_units(self):
        ds = self.ds
        self.status_left.setText(ds.summary())
        self.plot.getAxis('left').enableAutoSIPrefix(False)   # G12: never 'kJ kg-1'
        self.plot.setLabel('left', ds.display_name, units=ds.units or None)
        self.profile.set_dataset(ds)        # relabelled; select_point below redraws it
        self.readout.configure(ds)          # select_point below restores the point label
        self.readout.set_height_source(
            self.height_companion.units if self.height_companion is not None else '',
            note='' if self.height_companion is not None else self._height_reason())
        # The interval and the sort band are stated in the units on screen, so both
        # tooltips are stale the moment those change.
        self._sync_isolines_check()
        self._sync_sort_check()
        self._sync_scale_combo(apply_default=False)    # the fixed range is in these units
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

    def _on_cmap_changed(self, _name):
        self.refresh_map()          # which pushes the colours through _apply_colormap

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
        for worker in (self.decompressor, self.builder, self.writer,
                       self._overlay_builder, self._height_builder):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(3000)
        super().closeEvent(ev)

    def _error(self, message):
        QtWidgets.QMessageBox.critical(self, 'IMS ICON Ensemble Viewer', message)
        if self.ds is None:
            self.stack.setCurrentIndex(0)
