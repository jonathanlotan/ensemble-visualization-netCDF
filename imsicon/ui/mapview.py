"""MapView - the left panel: the whole model domain, zoomable and clickable (F3)."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import barbs

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
# Wind barbs sit above the outlines: they are the reading, and an outline crossing a barb
# is easier to follow than a barb hidden under a border.
Z_BARB = 15
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
        # Wind barbs (R4). Strokes -- staffs, feathers and calm circles -- are one
        # NaN-separated polyline pair, the same trick the coastline uses, so a screenful
        # of barbs is two scene items rather than a thousand. Pennants are the exception:
        # a 50 kt flag is filled, which a polyline cannot do, so they get their own path.
        self._wind = None                  # source(rows, cols) -> (u, v) in knots
        self._wind_stamp = 0
        self._barb_key = None              # what the drawn barbs were computed from
        self.barb_count = 0                # how many are on screen (status bar, tests)
        self.barb_stride = (0, 0)          # grid cells skipped between them (y, x)
        self.barb = self._outline(Z_BARB, '#101010', 1.3, halo=3.0, ignore_bounds=True)
        self.barb_flags = self._flag_layer()

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
        # The barb resolution follows the zoom, so the view range is what drives a redraw
        # -- not the time step. Barbs are added with ignoreBounds, so they can never feed
        # their own extent back into the range that produced them.
        self.plot.vb.sigRangeChanged.connect(lambda *_: self._redraw_barbs())
        self.scene().sigMouseClicked.connect(self._on_click)
        self.scene().sigMouseMoved.connect(self._on_move)

    # ---- setup -----------------------------------------------------------------
    def _outline(self, z, colour, width, halo, style=QtCore.Qt.PenStyle.SolidLine,
                 ignore_bounds=False):
        """A halo line and an ink line as one object, so callers set data once."""
        under = pg.PlotDataItem(connect='finite', pen=pg.mkPen(
            '#ffffff', width=halo, style=style))
        over = pg.PlotDataItem(connect='finite', pen=pg.mkPen(
            colour, width=width, style=style))
        under.setZValue(z)
        over.setZValue(z + 1)
        self.plot.addItem(under, ignoreBounds=ignore_bounds)
        self.plot.addItem(over, ignoreBounds=ignore_bounds)
        return _Outline(under, over)

    def _flag_layer(self):
        """Halo and ink path items for the pennant triangles, which are filled, not drawn.

        Both pens are cosmetic (pyqtgraph's default): a path item lives in data
        coordinates, so a 1.2 *degree* pen would be five times the width of the domain.
        """
        items = []
        for colour, width, brush in (('#ffffff', 3.0, None), ('#101010', 1.2, '#101010')):
            item = QtWidgets.QGraphicsPathItem()
            item.setPen(pg.mkPen(colour, width=width))
            item.setBrush(pg.mkBrush(brush) if brush else QtGui.QBrush())
            item.setZValue(Z_BARB + (0 if brush is None else 1))
            self.plot.addItem(item, ignoreBounds=True)
            items.append(item)
        return items

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
        self._cos_lat = float(np.cos(np.deg2rad(ds.lat.mean())))
        self.plot.setAspectLocked(True, ratio=self._cos_lat)
        self._user_zoomed = False
        self.set_wind(None)             # a new field's wind has not been pushed yet
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

    # ---- wind barbs ------------------------------------------------------------
    def set_wind(self, source):
        """`source(rows, cols) -> (u, v)` in KNOTS at those grid indices; None to clear.

        A source rather than two arrays, because how many barbs to draw is decided here,
        at paint time, from the view range -- that is what the zoom changes. Handing over
        a full-grid pair instead would mean computing 42,000 vectors to draw 400 of them
        on every frame.
        """
        self._wind = source
        self._wind_stamp += 1
        self._redraw_barbs()

    def _clear_barbs(self):
        self.barb.clear()
        for item in self.barb_flags:
            item.setPath(QtGui.QPainterPath())
        self.barb_count, self.barb_stride, self._barb_key = 0, (0, 0), None

    def _barb_sample(self):
        """-> (rows, cols, px, py) for the barbs the current view should hold.

        The stride comes from how many pixels one grid cell spans right now, so zooming
        in walks it down to 1 -- a barb at every grid point -- and zooming out walks it
        back up. The sample is also clipped to the visible window with a cell of margin,
        which is what keeps a 261x161 grid from being turned into 42,000 glyphs the
        moment somebody zooms in.
        """
        vb = self.plot.vb
        px, py = (float(value) for value in vb.viewPixelSize())
        if not (np.isfinite(px) and np.isfinite(py)) or px <= 0 or py <= 0:
            return None
        (x0, x1), (y0, y1) = vb.viewRange()
        stride_x = barbs.choose_stride(self.ds.dlon / px)
        stride_y = barbs.choose_stride(self.ds.dlat / py)
        cols = barbs.sample_indices(self.ds.nx, stride_x,
                                    (x0 - self.ds.lon[0]) / self.ds.dlon - stride_x,
                                    (x1 - self.ds.lon[0]) / self.ds.dlon + stride_x)
        rows = barbs.sample_indices(self.ds.ny, stride_y,
                                    (y0 - self.ds.lat[0]) / self.ds.dlat - stride_y,
                                    (y1 - self.ds.lat[0]) / self.ds.dlat + stride_y)
        return rows, cols, px, py

    def _redraw_barbs(self):
        """Rebuild the barb geometry for the view as it is now."""
        if self.ds is None or self._wind is None:
            if self._barb_key is not None:
                self._clear_barbs()
            return
        sample = self._barb_sample()
        if sample is None:
            return
        rows, cols, px, py = sample
        if rows.size == 0 or cols.size == 0:
            self._clear_barbs()
            return
        # Redrawing on every range change would rebuild identical geometry all through a
        # pan that does not cross a grid line. The key is what the geometry depends on --
        # nothing else -- so this cannot go stale behind a real change.
        key = (self._wind_stamp, int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]),
               int(rows.size), int(cols.size), round(px, 9), round(py, 9))
        if key == self._barb_key:
            return
        self._barb_key = key

        u, v = self._wind(rows, cols)
        lon, lat = np.meshgrid(self.ds.lon[cols], self.ds.lat[rows])
        geometry = barbs.barb_geometry(
            lon.ravel(), lat.ravel(), np.asarray(u).ravel(), np.asarray(v).ravel(),
            px=px, py=py, cos_lat=getattr(self, '_cos_lat', 1.0))
        self.barb.setData(*geometry['lines'])
        self._set_flags(geometry['flags'])
        self.barb_count = int(rows.size * cols.size)
        self.barb_stride = (int(rows[1] - rows[0]) if rows.size > 1 else 1,
                            int(cols[1] - cols[0]) if cols.size > 1 else 1)

    def _set_flags(self, triangles):
        path = QtGui.QPainterPath()
        for triangle in (triangles if triangles is not None else ()):
            path.moveTo(float(triangle[0][0]), float(triangle[0][1]))
            for point in triangle[1:]:
                path.lineTo(float(point[0]), float(point[1]))
            path.closeSubpath()
        for item in self.barb_flags:
            item.setPath(path)

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
