"""ReadoutPanel - the F4 statistics, shown to the right of the map.

Two row sets, chosen by what the file's second axis is (`levels.py`):

* an **ensemble** gets the six F4 numbers -- the mean, the extremes and the two
  percentiles across the 20 members, which is what an ensemble is *for*;
* a **column of pressure levels** gets the value at the selected level, and the top and
  bottom of the column *named by the level they occur at*. It deliberately does NOT get a
  mean or a P90: those would read as ensemble statistics on a file that has no ensemble,
  and the mean of 1000 hPa and 150 hPa is not a temperature anyone forecasts.

Both sets are built once and one is hidden, so switching between an ensemble file and a
pressure file does not rebuild the layout under the cursor.
"""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

ROWS = (('time', 'Time (Zulu)'), ('mean', 'Ensemble mean'), ('max', 'Maximum'),
        ('min', 'Minimum'), ('p90', 'Top 90 % (P90)'), ('p10', 'Bottom 90 % (P10)'))

# The pressure-column set. `value` is the level the map is showing; the two extremes say
# *where* in the column they are, which is the question a profile actually answers.
# `height` (R9) is the geopotential height of that level at that point and time, read
# from the run's own `geopot` file -- "850 hPa" names a level, the height says where it is.
LEVEL_ROWS = (('time', 'Time (Zulu)'), ('level', 'Level shown'), ('value', 'Value here'),
              ('height', 'Height (geopot)'),
              ('max', 'Highest in column'), ('min', 'Lowest in column'))
# R9: on the deterministic run the date and the value are the two numbers a reader comes
# to this panel for, so they are set larger than the rows around them -- not bold alone,
# which the eye reads as a heading, but a size up, which it reads as the answer.
PROMINENT = (('level', 'time'), ('level', 'value'))
PROMINENT_POINTS = 4            # points added to the panel's font for those rows


def format_value(value, span=None, units=''):
    """Digits chosen from the spread of the data, so small fields stay readable."""
    if value is None or not np.isfinite(value):
        return '--'
    magnitude = span if (span and np.isfinite(span) and span > 0) else abs(value)
    if magnitude >= 100:
        text = f'{value:,.1f}'
    elif magnitude >= 10:
        text = f'{value:,.2f}'
    elif magnitude >= 1:
        text = f'{value:,.3f}'
    else:
        text = f'{value:,.4f}'
    return f'{text} {units}'.strip()


def format_height(height, units=''):
    """`4.35 kft` / `1,325 gpm`: two decimals below 100, none above -- a kilofoot
    rounded to a whole number would put every low level at the same height."""
    if height is None or not np.isfinite(height):
        return '--'
    text = f'{height:,.2f}' if abs(height) < 100 else f'{height:,.0f}'
    return f'{text} {units}'.strip()


class ReadoutPanel(QtWidgets.QFrame):
    """Values for the hovered time step at the selected grid point."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.units = ''
        self.span = None
        self.mode = 'member'

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(12, 8, 12, 8)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(3)

        self.title = QtWidgets.QLabel('No point selected')
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        grid.addWidget(self.title, 0, 0, 1, 2)

        self.subtitle = QtWidgets.QLabel('Click the map to choose a grid point')
        self.subtitle.setStyleSheet('color: #666;')
        grid.addWidget(self.subtitle, 1, 0, 1, 2)

        mono = QtGui.QFont('Menlo' if QtGui.QFontDatabase.hasFamily('Menlo') else 'Monospace')
        mono.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        # `values` is always the ACTIVE set, keyed by the plain row name, so a caller
        # (and R1's tests) can keep asking for `values['mean']` without knowing that a
        # second set exists behind it.
        self.values = {}
        self._sets = {'member': {}, 'level': {}}
        self._rows = {}
        self._hidden = set()
        self.height_units = ''
        row = 2
        for mode, spec in (('member', ROWS), ('level', LEVEL_ROWS)):
            widgets = []
            for key, label in spec:
                name = QtWidgets.QLabel(label)
                name.setStyleSheet('color: #444;')
                value = QtWidgets.QLabel('--')
                value.setFont(mono)                  # fixed width: no jitter while hovering
                if (mode, key) in PROMINENT:
                    for widget in (name, value):
                        big = widget.font()
                        big.setPointSize(big.pointSize() + PROMINENT_POINTS)
                        big.setBold(True)
                        widget.setFont(big)
                    name.setStyleSheet('color: #222;')
                value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                   | QtCore.Qt.AlignmentFlag.AlignVCenter)
                value.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
                grid.addWidget(name, row, 0)
                grid.addWidget(value, row, 1)
                self._sets[mode][key] = value
                self._rows[(mode, key)] = (name, value)
                widgets += [name, value]
                row += 1
        grid.setColumnStretch(1, 1)
        self._show_mode('member')

    # ---- which set of rows ---------------------------------------------------------
    def _show_mode(self, mode, hide=()):
        self.mode = mode
        self.values = self._sets[mode]
        self._hidden = set(hide)
        for (row_mode, key), widgets in self._rows.items():
            visible = row_mode == mode and key not in self._hidden
            for widget in widgets:
                widget.setVisible(visible)

    def row_visible(self, key):
        return self.mode in ('member', 'level') and (self.mode, key) in self._rows \
            and key not in self._hidden and self._rows[(self.mode, key)][0].isVisible()

    def font_points(self, key):
        """Point size of a row's value label -- what "more visible" means, in a test."""
        return self._rows[(self.mode, key)][1].font().pointSize()

    def configure(self, ds, span=None):
        self.units = ds.units
        self.span = span
        if ds.axis.aggregatable:
            self._show_mode('member')
        else:
            # A surface field has one level, so "highest in column" and "lowest in column"
            # would both be the value already printed above them. Two rows that can only
            # ever repeat a third are worse than no rows. The height row is for a column
            # of pressure levels only -- a 2 m field has no level to be the height of --
            # and not for the geopotential itself, whose value IS the height.
            hide = () if ds.n_members > 1 else ('max', 'min')
            if not getattr(ds.axis, 'is_pressure', False) or self._is_height_field(ds):
                hide = tuple(hide) + ('height',)
            self._show_mode('level', hide=hide)
        self.subtitle.setText(
            f'{ds.display_name} - {ds.long_name} [{ds.units}] | {ds.axis.describe()} | '
            f'run {ds.run_init:%Y-%m-%d %H:%M}Z')

    @staticmethod
    def _is_height_field(ds):
        from .. import products
        return products.field_key(getattr(ds, 'field', '')) == 'GEOPOT'

    def set_point(self, lat, lon):
        self.title.setText(f'{lat:.3f}°N  {lon:.3f}°E')

    def set_height_source(self, units='', note=''):
        """What the height row reads in, and -- while there is nothing to read -- why.

        The note goes on the row's tooltip, so a reader who sees `--` can find out that
        the run's `geopot` is not on disk, or is still being opened, without the panel
        having to spell it out in the row itself.
        """
        self.height_units = units or ''
        name, value = self._rows[('level', 'height')]
        for widget in (name, value):
            widget.setToolTip(note or 'Geopotential height of the level shown, at this '
                                      'point and time, from the run\'s geopot file')

    # ---- the numbers ---------------------------------------------------------------
    def show_stats(self, time_label, stats, hovering=False):
        """Fill whichever row set is showing, from `member_stats` or `level_stats`."""
        self.values['time'].setText(time_label)
        if self.mode == 'member':
            for key in ('mean', 'max', 'min', 'p90', 'p10'):
                self._set(key, stats.get(key))
        else:
            self.values['level'].setText(str(stats.get('level', '') or '--'))
            self._set('value', stats.get('value'))
            height = stats.get('height')
            self.values['height'].setText(format_height(height, self.height_units))
            # The extremes carry the level they came from: "12.3 °C at 700 hPa" is a
            # reading, "12.3 °C" from somewhere in a 10 km column is not.
            self._set('max', stats.get('max'), stats.get('max_level'))
            self._set('min', stats.get('min'), stats.get('min_level'))
        self.setStyleSheet('QFrame { background: %s; }'
                           % ('#f2f7ff' if hovering else 'transparent'))

    def _set(self, key, value, where=None):
        text = format_value(value, self.span, self.units)
        if where and text != '--':
            text = f'{text}  @ {where}'
        self.values[key].setText(text)

    def clear(self):
        self.title.setText('No point selected')
        for widgets in self._sets.values():
            for value in widgets.values():
                value.setText('--')
