"""Settings... -- a simple form over the configuration file (R10, `config.py`).

Three tabs, one per thing the file holds: the IMS account, the points drawn on the map,
and the defaults each map opens with. Save writes the file and reads it straight back, so
what the app uses is what the next launch will read -- the dialog is an editor for the
file, never a second store beside it. The file stays hand-editable: *Open in text editor*
opens it, and *Reload from file* picks up what was typed there.

Validation errors are shown in the dialog rather than in a message box, so a mistyped
latitude costs a sentence in red, not a modal (and the offscreen test run cannot wedge on
one -- G30, G38).
"""
from PySide6 import QtCore, QtGui, QtWidgets

from .. import config, download, isolines, products, timefmt, transform
from . import colors

DEFAULT = '(default)'
ON_OFF = [DEFAULT, 'On', 'Off']
SCALE_CHOICES = [DEFAULT, 'Dataset range', 'This frame', 'Fixed range']
FIELD_COLUMNS = ['Map', 'Units', 'Colours', 'Scale', 'Min', 'Max', 'Isolines',
                 'Spacing', 'Profile']
COL = {name: i for i, name in enumerate(FIELD_COLUMNS)}
POINT_COLUMNS = ['Latitude', 'Longitude', 'Colour', 'Name']
# The derived maps, by the names they carry on screen, after both families' catalogues.
DERIVED_NAMES = ['TD_2M', 'T-Td', 'WSPD_10M', 'WSPD']


def known_maps():
    names = [p.field for family in products.FAMILIES for p in family.products]
    return list(dict.fromkeys(names + DERIVED_NAMES))


def unit_options(name):
    """The labels the Units combo would offer for a map, or [] when it offers no choice."""
    key = products.field_key(name)
    if key == products.field_key('T-Td'):
        key = 'T_2M'
    entry = transform.UNITS.get(key)
    return [choice.label for choice in entry.choices] if entry and entry.choices else []


def spacing_hint(name):
    """What a number in the Spacing column means for this map."""
    ladder = isolines.ladder_for(name)
    if isolines.interval_for(name) is not None or products.field_key(name) == 'T-TD':
        return ', '.join(f'{s:g}' for s in ladder.steps) + f' {ladder.unit}'
    entry = transform.UNITS.get(products.field_key(name))
    units = entry.expected[0] if entry and entry.expected else "the file's units"
    return f'any spacing, in {units}'


def _tristate_text(value):
    return DEFAULT if value is None else ('On' if value else 'Off')


def _tristate_value(text):
    return None if text == DEFAULT else text == 'On'


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, cfg=None, current=None, parent=None, path=None):
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self.setMinimumSize(900, 520)
        self.path = path or (cfg.path if cfg is not None and cfg.path else
                             config.config_path())
        self.cfg = cfg if cfg is not None else config.load(self.path)
        self.current = current          # the view on screen, for "Add the map on screen"
        self.saved = None

        layout = QtWidgets.QVBoxLayout(self)
        self.problems = QtWidgets.QLabel('')
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet('color:#a05000;')
        layout.addWidget(self.problems)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_credentials(), 'IMS account')
        self.tabs.addTab(self._build_points(), 'Points on the map')
        self.tabs.addTab(self._build_fields(), 'Map defaults')
        self.tabs.addTab(self._build_display(), 'Display')
        layout.addWidget(self.tabs, 1)

        self.error = QtWidgets.QLabel('')
        self.error.setWordWrap(True)
        self.error.setStyleSheet('color:#b00020;')
        self.error.hide()
        layout.addWidget(self.error)

        footer = QtWidgets.QHBoxLayout()
        where = QtWidgets.QLabel(f'Saved to {self.path}')
        where.setToolTip(str(self.path))
        where.setStyleSheet('color:#666;')
        where.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(where, 1)
        edit = QtWidgets.QPushButton('Open in text editor')
        edit.setToolTip('The file is plain TOML with its own instructions in it. Save '
                        'here first if you have unsaved changes, then edit, then press '
                        'Reload from file.')
        edit.clicked.connect(self.open_in_editor)
        footer.addWidget(edit)
        reload_button = QtWidgets.QPushButton('Reload from file')
        reload_button.clicked.connect(self.reload)
        footer.addWidget(reload_button)
        layout.addLayout(footer)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.populate(self.cfg)

    # ---- construction ------------------------------------------------------------------
    def _build_credentials(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.user_edit = QtWidgets.QLineEdit()
        form.addRow('User:', self.user_edit)
        row = QtWidgets.QHBoxLayout()
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        row.addWidget(self.password_edit, 1)
        self.show_password = QtWidgets.QCheckBox('Show')
        self.show_password.toggled.connect(lambda on: self.password_edit.setEchoMode(
            QtWidgets.QLineEdit.EchoMode.Normal if on
            else QtWidgets.QLineEdit.EchoMode.Password))
        row.addWidget(self.show_password)
        form.addRow('Password:', row)
        keychain = (' or the system keychain' if download.keyring_module() is not None
                    else '')
        note = QtWidgets.QLabel(
            'The IMS server account from the IMS product PDF, used by the Download... '
            'button. It is kept in plain text in the settings file below, which only '
            'your user account can read. Leave both empty to use IMS_USER / IMS_PASS '
            f'from the environment{keychain} instead; those take precedence when set.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666;')
        form.addRow(note)
        return page

    def _build_display(self):
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        self.time_combo = QtWidgets.QComboBox()
        for zone in timefmt.ZONES:
            self.time_combo.addItem(timefmt.LABELS[zone], zone)
        form.addRow('Times shown in:', self.time_combo)
        note = QtWidgets.QLabel(
            'The model files are in UTC (Zulu). Israel time is IDT (UTC+3) in summer and '
            'IST (UTC+2) in winter, decided for each time step, so a forecast that crosses '
            'the change shows it. The "Time" combo on the toolbar changes this for one '
            'session; this is what the app opens with.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#666;')
        form.addRow(note)
        return page

    def _build_points(self):
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        self.show_points = QtWidgets.QCheckBox('Show the points when the app opens '
                                               '("Points" on the toolbar turns them '
                                               'on and off)')
        box.addWidget(self.show_points)
        self.points_table = QtWidgets.QTableWidget(0, len(POINT_COLUMNS))
        self.points_table.setHorizontalHeaderLabels(POINT_COLUMNS)
        self.points_table.horizontalHeader().setStretchLastSection(True)
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.itemChanged.connect(self._on_point_edited)
        box.addWidget(self.points_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        for text, slot in (('Add point', self.add_point), ('Remove', self.remove_points),
                           ('Pick colour...', self.pick_colour)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        hint = QtWidgets.QLabel('Degrees north and east, e.g. 31.7683, 35.2137. An empty '
                                'colour is blue; any common colour name or #rrggbb works.')
        hint.setStyleSheet('color:#666;')
        buttons.addWidget(hint)
        box.addLayout(buttons)
        return page

    def _build_fields(self):
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        intro = QtWidgets.QLabel(
            'What each map opens with. Anything left at (default) keeps the app\'s own '
            'choice, and the toolbar still changes all of it while you read. Names ignore '
            'case, so T_2M also covers the deterministic run\'s t_2m. Min and Max are in '
            'the map\'s Units (or its default units). Spacing is in degrees C on a '
            'temperature and ft on a geopotential height; on any other map it is in the '
            'file\'s units, and setting one makes that map contourable.')
        intro.setWordWrap(True)
        intro.setStyleSheet('color:#666;')
        box.addWidget(intro)
        self.fields_table = QtWidgets.QTableWidget(0, len(FIELD_COLUMNS))
        self.fields_table.setHorizontalHeaderLabels(FIELD_COLUMNS)
        self.fields_table.verticalHeader().setVisible(False)
        header = self.fields_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        box.addWidget(self.fields_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton('Add map')
        add.clicked.connect(lambda: self.add_field_row())
        buttons.addWidget(add)
        self.add_current = QtWidgets.QPushButton('Add the map on screen')
        name = getattr(self.current, 'display_name', None)
        self.add_current.setEnabled(bool(name))
        if name:
            self.add_current.setText(f'Add the map on screen ({name})')
        self.add_current.clicked.connect(lambda: self.add_field_row(
            config.FieldDefaults(name=self._current_name())))
        buttons.addWidget(self.add_current)
        remove = QtWidgets.QPushButton('Remove')
        remove.clicked.connect(self.remove_fields)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        box.addLayout(buttons)
        return page

    def _current_name(self):
        """The view on screen by the name the file would use: `T-Td`, not `T_2M-TD_2M`."""
        ds = self.current
        if ds is None:
            return ''
        if getattr(ds, 'derived', False) and getattr(ds, 'display_name', None):
            return ds.display_name
        return ds.field

    # ---- filling -----------------------------------------------------------------------
    def populate(self, cfg):
        self.cfg = cfg
        if cfg.warnings:
            self.problems.setText('⚠ The settings file has problems (those lines '
                                  'were skipped):\n  ' + '\n  '.join(cfg.warnings))
            self.problems.show()
        else:
            self.problems.hide()
        self.user_edit.setText(cfg.user)
        self.password_edit.setText(cfg.password)
        self.show_points.setChecked(cfg.show_points)
        self.time_combo.setCurrentIndex(max(0, self.time_combo.findData(cfg.time_zone)))
        self.points_table.setRowCount(0)
        for point in cfg.points:
            self.add_point(point)
        self.fields_table.setRowCount(0)
        for defaults in cfg.fields.values():
            self.add_field_row(defaults)

    def add_point(self, point=None):
        point = point if isinstance(point, config.Point) else None
        row = self.points_table.rowCount()
        self.points_table.insertRow(row)
        values = (('', '', '', '') if point is None else
                  (f'{point.lat:g}', f'{point.lon:g}', point.colour, point.name))
        self.points_table.blockSignals(True)
        for col, text in enumerate(values):
            self.points_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        self.points_table.blockSignals(False)
        self._paint_colour(row)
        return row

    def remove_points(self):
        for row in sorted({i.row() for i in self.points_table.selectedIndexes()},
                          reverse=True):
            self.points_table.removeRow(row)

    def pick_colour(self):
        row = self.points_table.currentRow()
        if row < 0:
            return
        item = self.points_table.item(row, 2)
        start = QtGui.QColor(item.text() if item and item.text() else
                             config.DEFAULT_POINT_COLOUR)
        chosen = QtWidgets.QColorDialog.getColor(start, self, 'Point colour')
        if chosen.isValid():
            self.points_table.item(row, 2).setText(chosen.name())

    def _on_point_edited(self, item):
        if item.column() == 2:
            self._paint_colour(item.row())

    def _paint_colour(self, row):
        """The colour cell is tinted with its own colour, so a typo shows at once."""
        item = self.points_table.item(row, 2)
        if item is None:
            return
        colour = QtGui.QColor(item.text().strip() or config.DEFAULT_POINT_COLOUR)
        self.points_table.blockSignals(True)
        if colour.isValid():
            item.setBackground(colour)
            item.setForeground(QtGui.QColor('white' if colour.lightness() < 140
                                            else 'black'))
            item.setToolTip('')
        else:
            item.setBackground(QtGui.QBrush())
            item.setForeground(QtGui.QColor('#b00020'))
            item.setToolTip('Not a colour Qt knows; it would be drawn in blue')
        self.points_table.blockSignals(False)

    def add_field_row(self, defaults=None):
        defaults = defaults or config.FieldDefaults()
        table = self.fields_table
        row = table.rowCount()
        table.insertRow(row)

        name = QtWidgets.QComboBox()
        name.setEditable(True)
        name.addItems(known_maps())
        name.setCurrentText(defaults.name)
        name.setMinimumWidth(130)
        table.setCellWidget(row, COL['Map'], name)

        units = QtWidgets.QComboBox()
        units.setEditable(True)
        units.setMinimumWidth(80)
        table.setCellWidget(row, COL['Units'], units)

        ramps = QtWidgets.QComboBox()
        ramps.addItems([DEFAULT] + colors.COLORMAPS)
        ramps.setCurrentText(colors.ramp_named(defaults.colours) or DEFAULT)
        table.setCellWidget(row, COL['Colours'], ramps)

        scale = QtWidgets.QComboBox()
        scale.addItems(SCALE_CHOICES)
        if defaults.fixed_range is not None:
            scale.setCurrentText('Fixed range')
        elif defaults.scale == 'dataset':
            scale.setCurrentText('Dataset range')
        elif defaults.scale == 'frame':
            scale.setCurrentText('This frame')
        table.setCellWidget(row, COL['Scale'], scale)

        # Spelled out rather than `range and range[0]`: a range that starts at 0 -- the
        # usual CAPE or precipitation scale -- would read as "no minimum" that way.
        lo, hi = defaults.fixed_range or (None, None)
        for col, value in ((COL['Min'], lo), (COL['Max'], hi)):
            edit = QtWidgets.QLineEdit('' if value is None else f'{value:g}')
            edit.setValidator(QtGui.QDoubleValidator())
            edit.setMaximumWidth(80)
            table.setCellWidget(row, col, edit)

        on_off = QtWidgets.QComboBox()
        on_off.addItems(ON_OFF)
        on_off.setCurrentText(_tristate_text(defaults.isolines))
        table.setCellWidget(row, COL['Isolines'], on_off)

        spacing = QtWidgets.QLineEdit('' if defaults.isoline_step is None
                                      else f'{defaults.isoline_step:g}')
        spacing.setValidator(QtGui.QDoubleValidator(0.0, 1e12, 6))
        spacing.setMaximumWidth(80)
        table.setCellWidget(row, COL['Spacing'], spacing)

        profile = QtWidgets.QComboBox()
        profile.addItems(ON_OFF)
        profile.setCurrentText(_tristate_text(defaults.profile))
        profile.setToolTip('Pressure-level maps only: the vertical profile instead of the '
                           'time graph')
        table.setCellWidget(row, COL['Profile'], profile)

        # Carried through unedited: the dialog has no column for it, and dropping a line
        # the user wrote by hand is not something a Save should ever do.
        name.setProperty('scale_units', defaults.scale_units or '')

        def follow_name(text, units=units, spacing=spacing, keep=defaults.units):
            current = units.currentText() or keep or ''
            units.blockSignals(True)
            units.clear()
            units.addItems([''] + unit_options(text))
            units.setCurrentText(current)
            units.blockSignals(False)
            units.setToolTip('Offered: ' + (', '.join(unit_options(text)) or
                                            "the file's own units only"))
            spacing.setPlaceholderText(spacing_hint(text) if text else '')
            spacing.setToolTip(f'Isoline spacing on {text or "this map"}: '
                               f'{spacing_hint(text)}')

        def follow_scale(text, row=row):
            fixed = text == 'Fixed range'
            for col in (COL['Min'], COL['Max']):
                self.fields_table.cellWidget(row, col).setEnabled(fixed)

        name.currentTextChanged.connect(follow_name)
        scale.currentTextChanged.connect(follow_scale)
        follow_name(defaults.name)
        units.setCurrentText(defaults.units or '')
        follow_scale(scale.currentText())
        return row

    def remove_fields(self):
        rows = {i.row() for i in self.fields_table.selectedIndexes()}
        if not rows and self.fields_table.currentRow() >= 0:
            rows = {self.fields_table.currentRow()}
        for row in sorted(rows, reverse=True):
            self.fields_table.removeRow(row)

    # ---- reading the form back ---------------------------------------------------------
    def collect(self):
        """The form -> (Config, [errors]). Nothing is written here."""
        errors = []
        points = []
        for row in range(self.points_table.rowCount()):
            cells = [(self.points_table.item(row, c).text().strip()
                      if self.points_table.item(row, c) else '') for c in range(4)]
            if not any(cells):
                continue
            lat, lon, colour, name = cells
            try:
                point = config.parse_point({'lat': lat, 'lon': lon, 'colour': colour,
                                            'name': name})
            except ValueError as exc:
                errors.append(f'Point {row + 1}: {exc}')
                continue
            if not QtGui.QColor(point.colour).isValid():
                errors.append(f'Point {row + 1}: {point.colour!r} is not a colour '
                              '(try a name such as red, or #rrggbb)')
                continue
            points.append(point)

        entries, seen = [], {}
        for row in range(self.fields_table.rowCount()):
            cell = self.fields_table.cellWidget
            name = cell(row, COL['Map']).currentText().strip()
            if not name:
                continue
            key = products.field_key(name)
            if key in seen:
                errors.append(f'{name} is listed twice (rows {seen[key] + 1} and '
                              f'{row + 1}); names ignore case')
                continue
            seen[key] = row
            table = {}
            units = config.normalise_units_label(cell(row, COL['Units']).currentText())
            if units:
                table['units'] = units
            ramp = cell(row, COL['Colours']).currentText()
            if ramp != DEFAULT:
                table['colours'] = ramp
            scale = cell(row, COL['Scale']).currentText()
            if scale == 'Dataset range':
                table['scale'] = 'dataset'
            elif scale == 'This frame':
                table['scale'] = 'frame'
            elif scale == 'Fixed range':
                lo = cell(row, COL['Min']).text().strip()
                hi = cell(row, COL['Max']).text().strip()
                if not lo or not hi:
                    errors.append(f'{name}: a fixed range needs both Min and Max')
                    continue
                table['scale'] = [lo.replace(',', '.'), hi.replace(',', '.')]
            scale_units = cell(row, COL['Map']).property('scale_units')
            if scale_units:
                table['scale_units'] = scale_units
            isolines_on = _tristate_value(cell(row, COL['Isolines']).currentText())
            if isolines_on is not None:
                table['isolines'] = isolines_on
            spacing = cell(row, COL['Spacing']).text().strip()
            if spacing:
                table['isoline_step'] = spacing.replace(',', '.')
            profile = _tristate_value(cell(row, COL['Profile']).currentText())
            if profile is not None:
                table['profile'] = profile
            defaults, problems = config.parse_field(name, table)
            errors.extend(p.replace('; ignored', '') for p in problems)
            entries.append(defaults)

        cfg = config.with_fields(self.cfg, entries)
        cfg.user = self.user_edit.text().strip()
        cfg.password = self.password_edit.text()
        cfg.show_points = self.show_points.isChecked()
        cfg.time_zone = self.time_combo.currentData()
        cfg.points = points
        cfg.path = self.path
        return cfg, errors

    # ---- buttons -----------------------------------------------------------------------
    def save(self):
        """Validate, write, read back. -> the Config as the app will now read it, or None."""
        cfg, errors = self.collect()
        if errors:
            self.error.setText('Not saved:\n  ' + '\n  '.join(errors))
            self.error.show()
            return None
        try:
            config.save(cfg, self.path)
        except OSError as exc:
            self.error.setText(f'Could not write {self.path}: {exc}')
            self.error.show()
            return None
        self.error.hide()
        self.saved = config.load(self.path)
        return self.saved

    def accept(self):
        if self.save() is not None:
            super().accept()

    def reload(self):
        self.error.hide()
        self.populate(config.load(self.path))

    def open_in_editor(self):
        if not self.path.exists() and self.save() is None:
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.path)))
