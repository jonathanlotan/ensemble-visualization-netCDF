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
LEVEL_ROWS = (('time', 'Time (Zulu)'), ('level', 'Level shown'), ('value', 'Value here'),
              ('max', 'Highest in column'), ('min', 'Lowest in column'))


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
        row = 2
        for mode, spec in (('member', ROWS), ('level', LEVEL_ROWS)):
            widgets = []
            for key, label in spec:
                name = QtWidgets.QLabel(label)
                name.setStyleSheet('color: #444;')
                value = QtWidgets.QLabel('--')
                value.setFont(mono)                  # fixed width: no jitter while hovering
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
        for (row_mode, key), widgets in self._rows.items():
            visible = row_mode == mode and key not in hide
            for widget in widgets:
                widget.setVisible(visible)

    def configure(self, ds, span=None):
        self.units = ds.units
        self.span = span
        if ds.axis.aggregatable:
            self._show_mode('member')
        else:
            # A surface field has one level, so "highest in column" and "lowest in column"
            # would both be the value already printed above them. Two rows that can only
            # ever repeat a third are worse than no rows.
            self._show_mode('level', hide=() if ds.n_members > 1 else ('max', 'min'))
        self.subtitle.setText(
            f'{ds.display_name} - {ds.long_name} [{ds.units}] | {ds.axis.describe()} | '
            f'run {ds.run_init:%Y-%m-%d %H:%M}Z')

    def set_point(self, lat, lon):
        self.title.setText(f'{lat:.3f}°N  {lon:.3f}°E')

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
