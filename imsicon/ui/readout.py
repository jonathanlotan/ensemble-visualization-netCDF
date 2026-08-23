"""ReadoutPanel - the F4 statistics, shown to the right of the map."""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

ROWS = (('time', 'Time (Zulu)'), ('mean', 'Ensemble mean'), ('max', 'Maximum'),
        ('min', 'Minimum'), ('p90', 'Top 90 % (P90)'), ('p10', 'Bottom 90 % (P10)'))


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
    """Six values for the hovered time step at the selected grid point."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.units = ''
        self.span = None

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
        self.values = {}
        for row, (key, label) in enumerate(ROWS, start=2):
            name = QtWidgets.QLabel(label)
            name.setStyleSheet('color: #444;')
            value = QtWidgets.QLabel('--')
            value.setFont(mono)                      # fixed width: no jitter while hovering
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                               | QtCore.Qt.AlignmentFlag.AlignVCenter)
            value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self.values[key] = value
        grid.setColumnStretch(1, 1)

    def configure(self, ds, span=None):
        self.units = ds.units
        self.span = span
        self.subtitle.setText(
            f'{ds.field} - {ds.long_name} [{ds.units}] | {ds.n_members} members | '
            f'run {ds.run_init:%Y-%m-%d %H:%M}Z')

    def set_point(self, lat, lon):
        self.title.setText(f'{lat:.3f}°N  {lon:.3f}°E')

    def show_stats(self, time_label, stats, hovering=False):
        self.values['time'].setText(time_label)
        for key in ('mean', 'max', 'min', 'p90', 'p10'):
            self.values[key].setText(format_value(stats.get(key), self.span, self.units))
        self.setStyleSheet('QFrame { background: %s; }' % ('#f2f7ff' if hovering else 'transparent'))

    def clear(self):
        self.title.setText('No point selected')
        for value in self.values.values():
            value.setText('--')
