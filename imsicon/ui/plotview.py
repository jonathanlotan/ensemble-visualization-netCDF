"""PlotView - the right panel: every member (or level) through time, hover-reactive (F4).

On an ensemble file the 20 curves are the 20 members of one forecast, and the thick mean
and the min-max envelope over them are the point of the panel. On a pressure-level file
the same 20 curves are the 20 levels of the column -- a time-height section of one point,
which is worth reading -- but the mean and the envelope are NOT drawn, because a mean of
1000 hPa and 150 hPa is not a temperature anyone forecasts (see `levels.py`). The selected
level's curve is drawn heavier instead, so the up/down keys show as movement in the graph.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from .. import timefmt


class PlotView(pg.PlotWidget):
    """20 member curves, ensemble mean, min-max envelope, hover and time cursors."""

    hovered = QtCore.Signal(int)      # time index under the cursor, -1 when it leaves
    timePicked = QtCore.Signal(int)   # click or drag of the time cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds = None
        self.series = None
        self._curves = []
        self._pens = []
        self._n_times = 0
        self._highlight = None          # index of the curve drawn heavier, or None

        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel('bottom', 'forecast hour')
        self.setMenuEnabled(False)

        self._lo = pg.PlotDataItem(pen=None, connect='finite')
        self._hi = pg.PlotDataItem(pen=None, connect='finite')
        self.envelope = pg.FillBetweenItem(self._lo, self._hi,
                                           brush=pg.mkBrush(60, 60, 60, 38))
        self.addItem(self.envelope)
        self.mean_curve = pg.PlotDataItem(pen=pg.mkPen('#000000', width=2.6),
                                          connect='finite')
        self.mean_curve.setZValue(10)
        self.addItem(self.mean_curve)

        self.time_line = pg.InfiniteLine(angle=90, movable=True,
                                         pen=pg.mkPen('#d62728', width=1.8))
        self.time_line.setZValue(20)
        self.addItem(self.time_line)
        self.hover_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('#3355aa', width=1.0, style=QtCore.Qt.PenStyle.DashLine))
        self.hover_line.setZValue(19)
        self.hover_line.hide()
        self.addItem(self.hover_line)

        self.time_line.sigDragged.connect(self._on_drag)
        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)

    # ---- setup -----------------------------------------------------------------
    def set_dataset(self, ds):
        self.ds = ds
        self._n_times = ds.n_times
        for curve in self._curves:
            self.removeItem(curve)
        self._curves, self._pens = [], []
        self._highlight = None
        for m in range(ds.n_members):
            pen = pg.mkPen(pg.intColor(m, hues=max(2, ds.n_members), alpha=190), width=1.1)
            curve = pg.PlotDataItem(pen=pen, name=ds.member_labels[m],
                                    connect='finite')
            self.addItem(curve)
            self._curves.append(curve)
            self._pens.append(pen)
        # The mean and the envelope are statistics ACROSS the second axis, so they are
        # drawn only where that axis is an ensemble. Hidden rather than emptied: the items
        # keep their z-order and their brushes for the next file that does have members.
        across = ds.axis.aggregatable
        self.mean_curve.setVisible(across)
        self.envelope.setVisible(across)
        self.getAxis('left').enableAutoSIPrefix(False)   # J kg-1, never 'kJ kg-1'
        self.setLabel('left', ds.display_name, units=ds.units or None)
        self.label_time_axis()
        self.setXRange(float(ds.forecast_hours[0]), float(ds.forecast_hours[-1]), padding=0.01)

    def label_time_axis(self):
        """The x axis is forecast hours, which no clock changes; the run it counts from is
        written in the chosen one (R11)."""
        if self.ds is not None:
            self.setLabel('bottom', f'forecast hour from {timefmt.stamp(self.ds.run_init)} run')

    def set_level(self, index):
        """Draw one curve heavier -- the level (or member) the map is showing.

        Only for an axis with no mean of its own: on an ensemble the thick black line is
        already the mean, and a second thick line would compete with it.
        """
        if self.ds is None or self.ds.axis.aggregatable:
            return
        index = int(index) if self.ds.n_members > 1 else 0
        if index == self._highlight:
            return
        self._highlight = index
        for m, (curve, pen) in enumerate(zip(self._curves, self._pens)):
            if m == index:
                colour = pg.mkColor(pen.color())
                colour.setAlpha(255)
                curve.setPen(pg.mkPen(colour, width=2.8))
                curve.setZValue(9)
            else:
                curve.setPen(pen)
                curve.setZValue(0)

    def set_series(self, series):
        """series: (n_times, n_members) for the selected grid point."""
        self.series = series
        hours = self.ds.forecast_hours
        for m, curve in enumerate(self._curves):
            curve.setData(hours, series[:, m])
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                # An all-NaN time step is the window edge, not a problem.
                warnings.simplefilter('ignore', RuntimeWarning)
                mean = np.nanmean(series, axis=1)
                lo = np.nanmin(series, axis=1)
                hi = np.nanmax(series, axis=1)
        self.mean_curve.setData(hours, mean)
        self._lo.setData(hours, lo)
        self._hi.setData(hours, hi)

    def clear_series(self):
        self.series = None
        for curve in self._curves:
            curve.clear()
        self.mean_curve.clear()
        self._lo.clear()
        self._hi.clear()

    def set_yrange(self, lo, hi):
        if hi <= lo:
            hi = lo + 1.0
        self.setYRange(lo, hi, padding=0.02)

    def set_time(self, t):
        if self.ds is not None:
            self.time_line.setValue(float(self.ds.forecast_hours[t]))

    # ---- interaction -----------------------------------------------------------
    def _index_at(self, x):
        if self.ds is None or self._n_times == 0:
            return None
        return int(np.clip(np.abs(self.ds.forecast_hours - x).argmin(), 0, self._n_times - 1))

    def _on_drag(self):
        idx = self._index_at(self.time_line.value())
        if idx is not None:
            self.timePicked.emit(idx)

    def _on_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or self.ds is None:
            return
        if not self.sceneBoundingRect().contains(ev.scenePos()):
            return
        idx = self._index_at(self.getPlotItem().vb.mapSceneToView(ev.scenePos()).x())
        if idx is not None:
            self.timePicked.emit(idx)

    def _on_move(self, scene_pos):
        if self.ds is None or not self.sceneBoundingRect().contains(scene_pos):
            self._leave()
            return
        idx = self._index_at(self.getPlotItem().vb.mapSceneToView(scene_pos).x())
        if idx is None:
            self._leave()
            return
        self.hover_line.setValue(float(self.ds.forecast_hours[idx]))
        self.hover_line.show()
        self.hovered.emit(idx)

    def _leave(self):
        if self.hover_line.isVisible():
            self.hover_line.hide()
            self.hovered.emit(-1)

    def leaveEvent(self, ev):
        self._leave()
        super().leaveEvent(ev)
