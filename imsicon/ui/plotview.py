"""PlotView - the right panel: every ensemble member through time, hover-reactive (F4)."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore


class PlotView(pg.PlotWidget):
    """20 member curves, ensemble mean, min-max envelope, hover and time cursors."""

    hovered = QtCore.Signal(int)      # time index under the cursor, -1 when it leaves
    timePicked = QtCore.Signal(int)   # click or drag of the time cursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds = None
        self.series = None
        self._curves = []
        self._n_times = 0

        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel('bottom', 'forecast hour')
        self.setMenuEnabled(False)

        self._lo = pg.PlotDataItem(pen=None)
        self._hi = pg.PlotDataItem(pen=None)
        self.envelope = pg.FillBetweenItem(self._lo, self._hi,
                                           brush=pg.mkBrush(60, 60, 60, 38))
        self.addItem(self.envelope)
        self.mean_curve = pg.PlotDataItem(pen=pg.mkPen('#000000', width=2.6))
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
        self._curves = []
        for m in range(ds.n_members):
            pen = pg.mkPen(pg.intColor(m, hues=ds.n_members, alpha=190), width=1.1)
            curve = pg.PlotDataItem(pen=pen, name=ds.member_labels[m])
            self.addItem(curve)
            self._curves.append(curve)
        self.getAxis('left').enableAutoSIPrefix(False)   # J kg-1, never 'kJ kg-1'
        self.setLabel('left', ds.field, units=ds.units or None)
        self.setLabel('bottom', f'forecast hour from {ds.run_init:%Y-%m-%d %H:%M}Z run')
        self.setXRange(float(ds.forecast_hours[0]), float(ds.forecast_hours[-1]), padding=0.01)

    def set_series(self, series):
        """series: (n_times, n_members) for the selected grid point."""
        self.series = series
        hours = self.ds.forecast_hours
        for m, curve in enumerate(self._curves):
            curve.setData(hours, series[:, m])
        self.mean_curve.setData(hours, np.nanmean(series, axis=1))
        self._lo.setData(hours, np.nanmin(series, axis=1))
        self._hi.setData(hours, np.nanmax(series, axis=1))

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
