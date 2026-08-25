"""MapView - the left panel: the whole model domain, zoomable and clickable (F3)."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

# row-major must be set before any ImageItem exists: image[row=lat, col=lon]
pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Explicit stacking, because "drawn over the model map" must not depend on the order the
# items happened to be added in. The field is the bottom layer; every outline sits above
# it, and the picked-point marker above those.
Z_FIELD = 0
Z_COAST = 10
Z_BORDER = 12
Z_MARKER = 20


class _Outline:
    """A halo line and an ink line kept together, so a layer is set or cleared as one."""

    __slots__ = ('under', 'over')

    def __init__(self, under, over):
        self.under, self.over = under, over

    def setData(self, xs, ys):
        self.under.setData(xs, ys)
        self.over.setData(xs, ys)

    def clear(self):
        self.under.clear()
        self.over.clear()

    def isVisible(self):
        return self.over.isVisible()


class MapView(pg.GraphicsLayoutWidget):
    """Ensemble field as an image in true degrees, with coastline and a picked-point marker."""

    pointPicked = QtCore.Signal(int, int)              # grid indices (iy, ix)
    cursorMoved = QtCore.Signal(float, float, float)   # lat, lon, value (nan if off-grid)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds = None
        self._frame = None
        self._user_zoomed = False      # once True, resizes stop re-fitting the domain

        self.plot = self.addPlot(row=0, col=0)
        self.plot.setLabel('bottom', 'longitude', units='°E')
        self.plot.setLabel('left', 'latitude', units='°N')
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMenuEnabled(False)

        self.img = pg.ImageItem(axisOrder='row-major')
        self.img.setZValue(Z_FIELD)
        self.plot.addItem(self.img)
        # Coastline and borders, each drawn twice: a pale halo underneath and the line on
        # top. Over turbo or a diverging ramp there is no single ink colour that stays
        # legible against both ends of the scale, and an outline that disappears over the
        # values you are looking at is worse than none.
        self.coast = self._outline(Z_COAST, '#101010', 1.4, halo=3.2)
        self.borders = self._outline(Z_BORDER, '#7d1f1f', 1.5, halo=3.2)
        # Natural Earth marks these as disputed / indefinite / line of control. Dashed,
        # because drawing them identically to a settled boundary would overstate them.
        self.borders_uncertain = self._outline(Z_BORDER, '#7d1f1f', 1.4, halo=2.8,
                                               style=QtCore.Qt.PenStyle.DashLine)
        self.marker = pg.ScatterPlotItem(size=17, symbol='+', pen=pg.mkPen('#ffffff', width=2.5),
                                         brush=None)
        self.marker.setZValue(Z_MARKER + 1)
        self.plot.addItem(self.marker)
        self.marker_halo = pg.ScatterPlotItem(size=17, symbol='+', pen=pg.mkPen('#000000', width=4.5),
                                              brush=None)
        self.marker_halo.setZValue(Z_MARKER)
        self.plot.addItem(self.marker_halo)

        self.cmap = pg.colormap.get('turbo')
        self.cbar = pg.ColorBarItem(colorMap=self.cmap, interactive=True,
                                    values=(0.0, 1.0))
        self.cbar.setImageItem(self.img)
        self.addItem(self.cbar, row=0, col=1)

        # F3.2: wheel zoom is the ViewBox default; keep drag-pan and add a home view.
        self.plot.vb.setMouseEnabled(x=True, y=True)
        self.plot.vb.sigRangeChangedManually.connect(self._on_user_zoom)
        self.plot.vb.sigResized.connect(self._on_vb_resized)
        self.scene().sigMouseClicked.connect(self._on_click)
        self.scene().sigMouseMoved.connect(self._on_move)

    # ---- setup -----------------------------------------------------------------
    def _outline(self, z, colour, width, halo, style=QtCore.Qt.PenStyle.SolidLine):
        """A halo line and an ink line as one object, so callers set data once."""
        under = pg.PlotDataItem(connect='finite', pen=pg.mkPen(
            '#ffffff', width=halo, style=style))
        over = pg.PlotDataItem(connect='finite', pen=pg.mkPen(
            colour, width=width, style=style))
        under.setZValue(z)
        over.setZValue(z + 1)
        self.plot.addItem(under)
        self.plot.addItem(over)
        return _Outline(under, over)

    def set_dataset(self, ds, overlay=None):
        self.ds = ds
        lon0, lat0 = ds.lon[0], ds.lat[0]
        tr = QtGui.QTransform()
        tr.translate(lon0 - ds.dlon / 2.0, lat0 - ds.dlat / 2.0)
        tr.scale(ds.dlon, ds.dlat)
        self.img.setTransform(tr)

        x0, x1, y0, y1 = ds.extent()
        # Generous pan limits: an aspect-locked view needs slack on the short axis to fit
        # the whole domain, and tight limits would make it crop latitude instead.
        pad_x, pad_y = (x1 - x0) * 0.5, (y1 - y0) * 0.5
        self.plot.vb.setLimits(xMin=x0 - pad_x, xMax=x1 + pad_x,
                               yMin=y0 - pad_y, yMax=y1 + pad_y)
        # G8: 1 deg of longitude covers cos(lat) of the pixels that 1 deg of latitude does
        self.plot.setAspectLocked(True, ratio=float(np.cos(np.deg2rad(ds.lat.mean()))))
        self._user_zoomed = False
        self.set_overlay(overlay)
        self.reset_view()

    def set_overlay(self, overlay):
        """Draw the coastline and border layers (`geo.overlay_for`) over the field."""
        overlay = overlay or {}
        for name, item in (('coastline', self.coast), ('borders', self.borders),
                           ('borders_uncertain', self.borders_uncertain)):
            xs, ys = overlay.get(name) or (None, None)
            if xs is not None and len(xs):
                item.setData(xs, ys)
            else:
                item.clear()
        self.overlay_source = overlay.get('source', '')

    def reset_view(self):
        """Fit the whole domain. Aspect lock means one axis gets slack, not a crop."""
        if self.ds is None:
            return
        x0, x1, y0, y1 = self.ds.extent()
        self._user_zoomed = False
        self.plot.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.01)

    def _on_user_zoom(self):
        self._user_zoomed = True

    def _on_vb_resized(self):
        """The ViewBox knows its geometry before the widget does; refit until the user zooms."""
        if getattr(self, 'ds', None) is not None and not self._user_zoomed:
            self.reset_view()

    def resizeEvent(self, ev):
        # The aspect lock is resolved against pixel geometry, which is not final until the
        # layout settles -- so keep re-fitting until the user takes over the view.
        super().resizeEvent(ev)
        # pyqtgraph calls resizeEvent(None) from GraphicsView.__init__, before our attrs exist
        if getattr(self, 'ds', None) is not None and not self._user_zoomed:
            QtCore.QTimer.singleShot(0, self.reset_view)

    def set_colormap(self, name):
        try:
            self.cmap = pg.colormap.get(name)
        except Exception:
            self.cmap = pg.colormap.get('viridis')
        self.cbar.setColorMap(self.cmap)

    # ---- drawing ---------------------------------------------------------------
    def set_frame(self, frame, levels=None, label=None):
        self._frame = frame
        self.img.setImage(frame, autoLevels=False)
        if levels is not None:
            lo, hi = float(levels[0]), float(levels[1])
            if hi <= lo:
                hi = lo + 1.0
            self.cbar.setLevels(low=lo, high=hi)
        if label is not None:
            self.plot.setTitle(label)

    def set_marker(self, lat, lon):
        for item in (self.marker, self.marker_halo):
            item.setData([lon], [lat])

    # ---- interaction -----------------------------------------------------------
    def _view_point(self, scene_pos):
        if self.ds is None or not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        return self.plot.vb.mapSceneToView(scene_pos)

    def _on_click(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pt = self._view_point(ev.scenePos())
        if pt is None:
            return
        # F3.3: nearest_index clamps, so clicks just off the grid snap to the edge cell
        iy, ix = self.ds.nearest_index(pt.y(), pt.x())
        ev.accept()
        self.pointPicked.emit(iy, ix)

    def _on_move(self, scene_pos):
        pt = self._view_point(scene_pos)
        if pt is None:
            self.cursorMoved.emit(np.nan, np.nan, np.nan)
            return
        iy, ix = self.ds.nearest_index(pt.y(), pt.x())
        value = np.nan
        if self._frame is not None and 0 <= iy < self._frame.shape[0] and 0 <= ix < self._frame.shape[1]:
            inside = (self.ds.lon[0] - self.ds.dlon <= pt.x() <= self.ds.lon[-1] + self.ds.dlon
                      and self.ds.lat[0] - self.ds.dlat <= pt.y() <= self.ds.lat[-1] + self.ds.dlat)
            if inside:
                value = float(self._frame[iy, ix])
        self.cursorMoved.emit(pt.y(), pt.x(), value)

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key.Key_Home:
            self.reset_view()
            ev.accept()
            return
        super().keyPressEvent(ev)
