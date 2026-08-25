"""Choosing a derived field: the dew point, the wind map, or the difference of two fields.

The dialog only *chooses*. Opening the files it names can mean decompressing 262 MB, so
`MainWindow` does that on a worker thread -- keeping every long operation in one place
rather than sprinkling `QThread`s through dialogs.
"""
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .. import derived, download, ingest

# Presentation order for the A/B combos: the download catalogue's order, so CAPE and
# precipitation lead rather than whatever sorts first alphabetically, with the derived
# dew point slotted in beside the temperatures it belongs with.
FIELD_ORDER = []
for _product in download.PRODUCTS:
    FIELD_ORDER.append(_product.field)
    if _product.field == 'T_S':
        FIELD_ORDER.append(derived.DEW_POINT_FIELD)
    if _product.field == 'V_10M':
        FIELD_ORDER.append(derived.WIND_FIELD)


def field_sort_key(field):
    return (FIELD_ORDER.index(field) if field in FIELD_ORDER else len(FIELD_ORDER), field)


WIND_FIELDS = {role: field for role, (field, _units) in derived.WIND_INPUTS.items()}

DEW_POINT = 'dewpoint'
DEPRESSION = 'depression'
DIFFERENCE = 'difference'
WIND = 'wind'


class DerivedRequest:
    """What the user asked for: a kind, and the files it needs, in operand order."""

    def __init__(self, kind, paths, title):
        self.kind = kind
        self.paths = [Path(p) for p in paths]
        self.title = title

    def __repr__(self):
        return f'<DerivedRequest {self.kind} {[p.name for p in self.paths]}>'


class DerivedDialog(QtWidgets.QDialog):
    """Pick a dew point / depression / wind map / A-B difference from the files on disk."""

    def __init__(self, parent=None, near=None, run=None, roots=None):
        super().__init__(parent)
        self.setWindowTitle('Derived field')
        self.setMinimumWidth(560)
        # `roots` lets the window hand over everywhere it has looked, so this dialog and
        # the "Map shows" combo never disagree about which files exist.
        self.available = ingest.scan_for_fields(roots if roots is not None
                                                else ingest.search_roots(near))
        self.run = run or self._default_run()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._build_run_row())
        self.kind_group = QtWidgets.QButtonGroup(self)
        layout.addWidget(self._build_dewpoint_box())
        layout.addWidget(self._build_wind_box())
        layout.addWidget(self._build_difference_box())

        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('color:#a05000;')
        layout.addWidget(self.status)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.kind_group.idToggled.connect(lambda *_: self._refresh())
        # Show the run we are actually using. The combo lists newest-first, so without
        # this an open file from an older run silently disagrees with the label above it.
        self.run_combo.blockSignals(True)
        self.run_combo.setCurrentIndex(max(0, self.run_combo.findData(self.run)))
        self.run_combo.blockSignals(False)
        self._populate_run(self.run_combo.currentData())

    # ---- construction ------------------------------------------------------------
    def _default_run(self):
        runs = sorted({run for run, _field in self.available}, reverse=True)
        return runs[0] if runs else None

    def _build_run_row(self):
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel('Run:'))
        self.run_combo = QtWidgets.QComboBox()
        for run in sorted({run for run, _field in self.available}, reverse=True):
            self.run_combo.addItem(f'{run[:4]}-{run[4:6]}-{run[6:8]} {run[8:]}Z', run)
        self.run_combo.currentIndexChanged.connect(
            lambda _: self._populate_run(self.run_combo.currentData()))
        row.addWidget(self.run_combo, 1)
        hint = QtWidgets.QLabel('All fields must come from one run: the grid, the time '
                                'axis and the member order have to match.')
        hint.setStyleSheet('color:#666;')
        row.addWidget(hint)
        return box

    def _build_dewpoint_box(self):
        box = QtWidgets.QGroupBox('From temperature and humidity')
        form = QtWidgets.QVBoxLayout(box)
        self.dewpoint_radio = QtWidgets.QRadioButton(
            'Dew point (TD_2M) - the temperature the air must cool to for saturation')
        self.depression_radio = QtWidgets.QRadioButton(
            'Dew point depression (T_2M - TD_2M) - how far the air is from saturation')
        self.kind_group.addButton(self.dewpoint_radio, 0)
        self.kind_group.addButton(self.depression_radio, 1)
        self.dewpoint_radio.setChecked(True)
        form.addWidget(self.dewpoint_radio)
        form.addWidget(self.depression_radio)
        self.inputs_label = QtWidgets.QLabel('')
        self.inputs_label.setStyleSheet('color:#666;')
        self.inputs_label.setWordWrap(True)
        form.addWidget(self.inputs_label)
        return box

    def _build_wind_box(self):
        box = QtWidgets.QGroupBox('From the wind components')
        form = QtWidgets.QVBoxLayout(box)
        self.wind_radio = QtWidgets.QRadioButton(
            'Wind map (U_10M + V_10M) - wind speed, with wind barbs for the direction')
        self.kind_group.addButton(self.wind_radio, 3)
        form.addWidget(self.wind_radio)
        note = QtWidgets.QLabel(
            'The colours are the speed; the barbs are the direction, in knots (half '
            'feather 5, full 10, pennant 50). They thin out or fill in as you zoom.')
        note.setStyleSheet('color:#666;')
        note.setWordWrap(True)
        form.addWidget(note)
        self.wind_inputs_label = QtWidgets.QLabel('')
        self.wind_inputs_label.setStyleSheet('color:#666;')
        self.wind_inputs_label.setWordWrap(True)
        form.addWidget(self.wind_inputs_label)
        return box

    def _build_difference_box(self):
        box = QtWidgets.QGroupBox('Difference between two fields')
        grid = QtWidgets.QGridLayout(box)
        self.difference_radio = QtWidgets.QRadioButton('Difference map: A - B')
        self.kind_group.addButton(self.difference_radio, 2)
        grid.addWidget(self.difference_radio, 0, 0, 1, 4)
        grid.addWidget(QtWidgets.QLabel('A:'), 1, 0)
        self.a_combo = QtWidgets.QComboBox()
        grid.addWidget(self.a_combo, 1, 1)
        grid.addWidget(QtWidgets.QLabel('   minus   B:'), 1, 2)
        self.b_combo = QtWidgets.QComboBox()
        grid.addWidget(self.b_combo, 1, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        note = QtWidgets.QLabel(
            'Only fields in the same units can be subtracted, and the difference is taken '
            'member by member - so the map shows the spread of the difference, not the '
            'difference of the spreads.')
        note.setStyleSheet('color:#666;')
        note.setWordWrap(True)
        grid.addWidget(note, 2, 0, 1, 4)
        for combo in (self.a_combo, self.b_combo):
            combo.currentIndexChanged.connect(lambda _: self._refresh())
        return box

    # ---- state -------------------------------------------------------------------
    def _fields_for(self, run):
        return sorted((field for (r, field) in self.available if r == run),
                      key=field_sort_key)

    def _populate_run(self, run):
        self.run = run
        fields = self._fields_for(run)
        for combo in (self.a_combo, self.b_combo):
            combo.blockSignals(True)
            combo.clear()
            for field in fields:
                combo.addItem(f'{field}  ({self.available[(run, field)].name})', field)
            combo.blockSignals(False)
        self._default_pair(fields)
        self._refresh()

    def _default_pair(self, fields):
        """Open on a pair that can actually be subtracted, not just the first two.

        Both sides move: if the first field has no partner in the same units, offering it
        as A with a permanently invalid B just shows the user an error they did not cause.
        """
        if len(fields) < 2:
            return
        for i, a in enumerate(fields):
            for j, b in enumerate(fields):
                if i != j and derived.units_look_compatible(a, b):
                    self.a_combo.setCurrentIndex(i)
                    self.b_combo.setCurrentIndex(j)
                    return
        self.b_combo.setCurrentIndex(1)

    def _dew_point_inputs(self):
        """The two paths the dew point needs, or None with a reason."""
        missing = [field for field in ('T_2M', 'RELHUM_2M')
                   if (self.run, field) not in self.available]
        if missing:
            return None, (f'{" and ".join(missing)} for run {self.run} is not on disk. '
                          'Download it first (Download from IMS...), then come back.')
        return ([self.available[(self.run, 'T_2M')],
                 self.available[(self.run, 'RELHUM_2M')]], None)

    def _wind_inputs(self):
        """The two paths the wind map needs, or None with a reason."""
        fields = [WIND_FIELDS[role] for role in ('zonal', 'meridional')]
        missing = [field for field in fields if (self.run, field) not in self.available]
        if missing:
            return None, (f'{" and ".join(missing)} for run {self.run} is not on disk. '
                          'Download it first (Download from IMS...), then come back.')
        return [self.available[(self.run, field)] for field in fields], None

    def _refresh(self):
        """Keep OK honest: it is only enabled when the choice can actually be built."""
        kind = self.kind()
        problem = None
        if kind in (DEW_POINT, DEPRESSION):
            paths, problem = self._dew_point_inputs()
            self.inputs_label.setText(
                'Needs T_2M and RELHUM_2M from this run' if problem else
                'Using ' + ' + '.join(p.name for p in paths))
        elif kind == WIND:
            paths, problem = self._wind_inputs()
            self.wind_inputs_label.setText(
                'Needs U_10M and V_10M from this run' if problem else
                'Using ' + ' + '.join(p.name for p in paths))
        else:
            a, b = self.a_combo.currentData(), self.b_combo.currentData()
            if not a or not b:
                problem = f'No ICON ensemble files found for run {self.run}.'
            elif a == b:
                problem = 'A and B are the same field, so the difference is zero everywhere.'
            elif not derived.units_look_compatible(a, b):
                problem = (f'{a} and {b} are not measured in the same units, so their '
                           'difference would be a number with no meaning.')
        self.status.setText(problem or '')
        ok = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok.setEnabled(problem is None)

    def kind(self):
        return {0: DEW_POINT, 1: DEPRESSION, 2: DIFFERENCE,
                3: WIND}[self.kind_group.checkedId()]

    def request(self):
        """-> DerivedRequest, or None if the dialog was cancelled or cannot be satisfied."""
        kind = self.kind()
        if kind in (DEW_POINT, DEPRESSION):
            paths, problem = self._dew_point_inputs()
            if problem:
                return None
            title = ('Dew point TD_2M' if kind == DEW_POINT
                     else 'Dew point depression T_2M - TD_2M')
            return DerivedRequest(kind, paths, title)
        if kind == WIND:
            paths, problem = self._wind_inputs()
            return None if problem else DerivedRequest(
                kind, paths, 'Wind U_10M + V_10M')
        a, b = self.a_combo.currentData(), self.b_combo.currentData()
        if not a or not b or a == b or not derived.units_look_compatible(a, b):
            return None
        return DerivedRequest(
            DIFFERENCE, [self.available[(self.run, a)], self.available[(self.run, b)]],
            f'Difference {a} - {b}')


def build(request, opened=None):
    """Turn a `DerivedRequest` into a view. Runs on a worker thread: no Qt in here.

    `opened` is a {path: FieldView} of datasets the window already holds, so choosing a
    derived field built on the open file does not re-map 407 MB that is already mapped.
    """
    opened = opened or {}

    def field_view(path):
        existing = opened.get(Path(path))
        return existing if existing is not None else derived.open_field(path)

    views = [field_view(path) for path in request.paths]
    if request.kind == DEW_POINT:
        view = derived.dew_point(*views)
    elif request.kind == DEPRESSION:
        view = derived.dew_point_depression(*views)
    elif request.kind == WIND:
        view = derived.wind(*views)
    else:
        view = derived.difference(*views)
    # Stamped here rather than at each call site, so the "Map shows" combo names what is
    # on screen whether it was built from the dialog, the combo itself, or --derive.
    view.derived_kind = request.kind
    view.derived_request = request
    return view


class BuildWorker(QtCore.QThread):
    """Opening two 407 MB files (and maybe decompressing them) off the UI thread."""

    finished_view = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str)

    def __init__(self, request, opened=None, parent=None):
        super().__init__(parent)
        self.request = request
        self.opened = opened or {}
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            resolved = []
            for path in self.request.paths:
                self.progressed.emit(f'Opening {Path(path).name}...')
                resolved.append(ingest.resolve(path, cancel=lambda: self._cancel))
            if self._cancel:
                self.finished_view.emit(None)
                return
            request = DerivedRequest(self.request.kind, resolved, self.request.title)
            self.finished_view.emit(build(request, self.opened))
        except KeyboardInterrupt:
            self.finished_view.emit(None)
        except derived.PairError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
