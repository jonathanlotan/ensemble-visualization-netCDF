"""ProfileView - the right panel as a VERTICAL PROFILE: value across, height up (R9).

The time graph (`PlotView`) answers "how does this point change through the forecast";
on a pressure-level file its 22 curves are a time-height section. For relative humidity
the question a forecaster asks of one point and one time is a different one -- "where in
the column is it moist?" -- and that is a profile: the value on the x axis, the height
on the y axis, one point per level.

The height comes from the run's own `geopot` file (the geopotential height of each
pressure level at that point and time, `MainWindow.height_companion`), so the y axis is
in geopotential metres and the levels sit where the model says they are today, not where
a standard atmosphere would put them. Without that file the levels are drawn against
pressure instead, inverted so up is still up, and the axis says so: a profile against a
guessed height would look just as convincing and be wrong by a few hundred metres.

Hovering a level reports it to the readout (which then shows that level's value and
height rather than the map's); clicking one moves the map to it.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

HEIGHT_LABEL = 'geopotential height'
PRESSURE_LABEL = 'pressure level'
# The lowest levels are 25 hPa apart, which at 850..1000 hPa is ~220 m -- a few pixels on
# a panel that spans 15 km. Labels closer than this on screen are thinned, top down, so
# the right-hand axis names the levels it can rather than printing them over each other.
MIN_TICK_GAP_PX = 14


class ProfileView(pg.PlotWidget):
    levelHovered = QtCore.Signal(int)     # index on the second axis, -1 when the cursor leaves
    levelPicked = QtCore.Signal(int)      # a click on (or near) a level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds = None
        self.values = None                # (n_levels,) display values at the point and time
        self.ys = None                    # (n_levels,) the y each level is drawn at
        self.labels = ()
        self.by_height = False            # True when the y axis is height, not pressure
        self._level = 0
        self._hover = None
        self._xrange = None
        self.level_ticks = []             # what the right axis labels, for tests

        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMenuEnabled(False)
        self.getAxis('left').enableAutoSIPrefix(False)
        self.getAxis('bottom').enableAutoSIPrefix(False)
        self.getAxis('right').enableAutoSIPrefix(False)
        self.showAxis('right')
        self.getAxis('right').setStyle(showValues=True)

        self.curve = pg.PlotDataItem(pen=pg.mkPen('#1f4e9c', width=2.0), symbol='o',
                                     symbolSize=7, symbolBrush='#1f4e9c',
                                     symbolPen=pg.mkPen('#ffffff', width=1.0),
                                     connect='finite')
        self.addItem(self.curve)
        # The level the map is showing: a heavier marker, and a horizontal line across
        # the panel so it can be found when the curve is far from the eye.
        self.level_line = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen('#d62728', width=1.8))
        self.level_line.setZValue(5)
        self.addItem(self.level_line)
        self.level_marker = pg.ScatterPlotItem(size=14, symbol='o',
                                               pen=pg.mkPen('#d62728', width=2.2),
                                               brush=pg.mkBrush(0, 0, 0, 0))
        self.level_marker.setZValue(6)
        self.addItem(self.level_marker)
        self.hover_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen('#3355aa', width=1.0, style=QtCore.Qt.PenStyle.DashLine))
        self.hover_line.setZValue(4)
        self.hover_line.hide()
        self.addItem(self.hover_line)

        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)

    # ---- setup -----------------------------------------------------------------------
    def set_dataset(self, ds):
        self.ds = ds
        self.values, self.ys = None, None
        self.labels = tuple(ds.member_labels)
        self.curve.clear()
        self.level_marker.clear()
        self.setLabel('bottom', ds.display_name, units=ds.units or None)
        self._xrange = None

    def set_xrange(self, lo, hi):
        """The value axis, pinned to the dataset range like the time graph's y axis (A1):
        a profile at 03Z and one at 15Z must be read on one scale."""
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1.0
        self._xrange = (lo, hi)
        self.setXRange(lo, hi, padding=0.03)

    # ---- the profile -----------------------------------------------------------------
    def set_profile(self, values, heights=None, pressures=None, level=0, height_units='gpm'):
        """Draw one column: `values` per level, at `heights` (gpm) when given, else at
        `pressures` (hPa, inverted). `level` is the map's current position on the axis."""
        values = np.asarray(values, dtype=float)
        heights = None if heights is None else np.asarray(heights, dtype=float)
        pressures = None if pressures is None else np.asarray(pressures, dtype=float)
        self.by_height = (heights is not None and heights.size == values.size
                          and np.isfinite(heights).any())
        if self.by_height:
            ys = heights
            self.setLabel('left', HEIGHT_LABEL, units=height_units)
            self.invertY(False)
        elif pressures is not None and pressures.size == values.size:
            ys = pressures
            self.setLabel('left', PRESSURE_LABEL, units='hPa')
            self.invertY(True)
        else:
            ys = np.arange(values.size, dtype=float)
            self.setLabel('left', 'level index', units=None)
            self.invertY(False)
        self.values, self.ys = values, ys
        order = np.argsort(ys)                     # drawn bottom-up whatever the file order
        self.curve.setData(values[order], ys[order])
        good = np.isfinite(ys)
        if good.any():
            lo, hi = float(ys[good].min()), float(ys[good].max())
            pad = 0.04 * (hi - lo or 1.0)
            self.setYRange(lo - pad, hi + pad, padding=0)
        self._label_levels()
        if self._xrange is None and np.isfinite(values).any():
            self.setXRange(float(np.nanmin(values)), float(np.nanmax(values)), padding=0.05)
        self.set_level(level)

    def _label_levels(self):
        """The pressure of each level on the right-hand axis, whichever quantity the left
        one is in -- a profile is read by level name, and 850 hPa is the name.

        Thinned to what fits: the levels are labelled from the top of the column down,
        and one that would land within `MIN_TICK_GAP_PX` of the last label is skipped.
        The map's current level is never skipped, because it is the one being read.
        """
        axis = self.getAxis('right')
        if self.ys is None:
            axis.setTicks([])
            return
        vb = self.getPlotItem().vb
        (y0, y1) = vb.viewRange()[1]
        height_px = max(1.0, float(vb.height()))
        px_per_unit = height_px / (abs(y1 - y0) or 1.0)
        order = [int(i) for i in np.argsort(-self.ys) if np.isfinite(self.ys[i])]
        ticks, last = [], None
        for i in order:
            y_px = float(self.ys[i]) * px_per_unit
            keep = last is None or abs(y_px - last) >= MIN_TICK_GAP_PX or i == self._level
            if keep:
                ticks.append((float(self.ys[i]), self.labels[i] if i < len(self.labels)
                              else ''))
                last = y_px
        self.level_ticks = ticks
        axis.setTicks([ticks])

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, 'ys', None) is not None:
            self._label_levels()

    def set_level(self, index):
        """Mark the level the map is showing."""
        self._level = int(index)
        if self.ys is None or not (0 <= self._level < self.ys.size):
            self.level_marker.clear()
            self.level_line.hide()
            return
        y, x = float(self.ys[self._level]), float(self.values[self._level])
        if np.isfinite(y):
            self.level_line.setValue(y)
            self.level_line.show()
        else:
            self.level_line.hide()
        self._label_levels()
        if np.isfinite(x) and np.isfinite(y):
            self.level_marker.setData([x], [y])
        else:
            self.level_marker.clear()

    def clear_profile(self):
        self.values, self.ys = None, None
        self.curve.clear()
        self.level_marker.clear()
        self.level_line.hide()
        self.level_ticks = []
        self.getAxis('right').setTicks([])

    # ---- interaction -----------------------------------------------------------------
    def _index_at(self, scene_pos):
        """The level nearest the cursor, by screen distance along y; None off the panel."""
        if self.ys is None or not self.sceneBoundingRect().contains(scene_pos):
            return None
        vb = self.getPlotItem().vb
        point = vb.mapSceneToView(scene_pos)
        good = np.isfinite(self.ys)
        if not good.any():
            return None
        distance = np.where(good, np.abs(self.ys - point.y()), np.inf)
        return int(distance.argmin())

    def _on_move(self, scene_pos):
        idx = self._index_at(scene_pos)
        if idx is None:
            self._leave()
            return
        self.hover_line.setValue(float(self.ys[idx]))
        self.hover_line.show()
        if idx != self._hover:
            self._hover = idx
            self.levelHovered.emit(idx)

    def _on_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        idx = self._index_at(ev.scenePos())
        if idx is not None:
            self.levelPicked.emit(idx)

    def _leave(self):
        if self._hover is not None or self.hover_line.isVisible():
            self._hover = None
            self.hover_line.hide()
            self.levelHovered.emit(-1)

    def leaveEvent(self, ev):
        self._leave()
        super().leaveEvent(ev)
