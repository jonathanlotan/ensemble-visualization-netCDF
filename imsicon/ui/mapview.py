"""MapView - the left panel: the whole model domain, zoomable and clickable (F3)."""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import barbs, isolines as iso, terrain

# row-major must be set before any ImageItem exists: image[row=lat, col=lon]
pg.setConfigOption('imageAxisOrder', 'row-major')
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Explicit stacking, because "drawn over the model map" must not depend on the order the
# items happened to be added in. The land fill is under everything, the field over it,
# every outline above that, and the picked-point marker above those.
#
# The land is under the FIELD rather than over it because the field is what it is there
# for: a turbo map is transparent where the value is zero (`ui/colors.py`), so the grey
# shows through exactly where there is nothing to say, and a forecaster reading an empty
# CAPE map still sees which side of the coast they are looking at. The sea is not drawn
# at all -- the widget background is already white.
Z_LAND = -10
Z_FIELD = 0
# The shaded relief (R9) sits directly on the field and multiplies it: slopes facing away
# from the light darken whatever is under them -- the field's colour, or the grey land
# where the field is transparent -- and nothing else changes. Under the isolines, which
# are a reading, and above the field, because a multiply under an opaque image would do
# nothing at all.
Z_TERRAIN = 1
# Isolines of the field sit directly on it, under every geographic outline: the coastline
# is the frame you read the contours against, so it goes on top of them, not under.
Z_ISOLINE = 5
# The values written on the isolines (R15) sit over every outline and under the barbs: a
# number half-hidden under a border is unreadable, while a coastline running under a small
# label box is still a coastline either side of it.
Z_ISOLINE_LABEL = 14
Z_COAST = 10
Z_BORDER = 12
# Wind barbs sit above the outlines: they are the reading, and an outline crossing a barb
# is easier to follow than a barb hidden under a border.
Z_BARB = 15
# The user's own points (R10, from the configuration file) sit over the reading layers --
# they are landmarks the eye navigates by -- and under the picked-point marker, which is
# the one thing on the map that must never be hidden.
Z_POINTS = 18
Z_MARKER = 20
POINT_SIZE_PX = 9

# The colorbar column plus the layout's own margins, kept out of the title's wrapping
# width (G36). MEASURED (R9): at 110 px the layout came out 738 px wide in a 718 px
# widget -- the plot's minimum is the wrapped title plus 41 px of its own frame, and the
# colorbar is 63 px once its axis sizes itself to four-digit labels -- so the colorbar's
# last 20 px, and with them the last digit of every label over 999, were off the edge of
# the widget. 150 px leaves room for a five-digit scale.
TITLE_MARGIN_PX = 150

# Pale enough to sit under a colour ramp without competing with the lowest values it
# carries, dark enough to read as land against the white sea at a glance.
LAND_COLOUR = '#e7e7e2'


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

        # Land under the field, so an empty map is still a map. `ignoreBounds` for the
        # reason the barbs use it: the fill reaches past the model domain to the edge of
        # the bundled overlay, and it must not be able to drag the view range out with it.
        self.land = QtWidgets.QGraphicsPathItem()
        self.land.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.land.setBrush(pg.mkBrush(LAND_COLOUR))
        self.land.setZValue(Z_LAND)
        self.plot.addItem(self.land, ignoreBounds=True)
        self.land_rings = 0                # how many are filled (status, tests)

        self.img = pg.ImageItem(axisOrder='row-major')
        self.img.setZValue(Z_FIELD)
        self.plot.addItem(self.img)
        # Shaded relief (R9), built on first use from the bundled elevation grid and the
        # land polygons above, and composited with Multiply so it can only ever darken.
        # `ignoreBounds`: it covers the whole pan range, and the pan range must not be
        # allowed to grow to fit it.
        self.terrain = pg.ImageItem(axisOrder='row-major')
        self.terrain.setZValue(Z_TERRAIN)
        self.terrain.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_Multiply)
        self.terrain.hide()
        self.plot.addItem(self.terrain, ignoreBounds=True)
        self.terrain_on = False            # asked for
        self._terrain_key = None           # what the relief image was built from
        self._land_ref = None              # the rings the land path was filled from
        # Isolines (R5), in two weights: every line, and every 5th (or 2nd) drawn heavier
        # so the eye can count in fives instead of one at a time. Both `ignoreBounds`, for
        # the reason the barbs are: geometry computed FROM the frame must never feed back
        # into the range that decides which frame is on screen.
        self.isoline = self._outline(Z_ISOLINE, '#1c1c1c', 0.9, halo=2.2,
                                     ignore_bounds=True)
        self.isoline_heavy = self._outline(Z_ISOLINE, '#1c1c1c', 1.7, halo=3.0,
                                           ignore_bounds=True)
        self.isoline_levels = np.empty(0)     # what is drawn, for the status bar and tests
        self.isoline_step = 0.0               # the interval actually used (see G35)
        # R15: the values written on the lines. A pool of text items reused across frames,
        # because creating and destroying a hundred of them on every scrub step is what
        # would cost the frame, not drawing them.
        self.isoline_labels_on = False
        self._iso_segments = iso.NO_SEGMENTS
        self._iso_decimals = 0
        self._iso_label_key = None
        self._iso_label_pool = []
        self.isoline_label_texts = []         # what is written, in order (tests, status)
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

        # R10: the configured points. One scatter for the dots, one text item per name.
        self.points = pg.ScatterPlotItem(size=POINT_SIZE_PX, pen=pg.mkPen('#ffffff', width=1.5))
        self.points.setZValue(Z_POINTS)
        self.plot.addItem(self.points, ignoreBounds=True)
        self.point_labels = []
        self.points_shown = True

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
        # pyqtgraph pins the colorbar's axis to 45 px, which fits three digits and clips
        # the fourth: a CAPE scale read 0, 100, 200, 300 for 0..3000 J kg-1, and a 500 hPa
        # height chart 552 for 5,520 gpm (R9). None lets the axis size itself to its text.
        self.cbar.axis.setWidth(None)
        self.addItem(self.cbar, row=0, col=1)

        # F3.2: wheel zoom is the ViewBox default; keep drag-pan and add a home view.
        self.plot.vb.setMouseEnabled(x=True, y=True)
        self.plot.vb.sigRangeChangedManually.connect(self._on_user_zoom)
        self.plot.vb.sigResized.connect(self._on_vb_resized)
        # The barb resolution follows the zoom, so the view range is what drives a redraw
        # -- not the time step. Barbs are added with ignoreBounds, so they can never feed
        # their own extent back into the range that produced them.
        self.plot.vb.sigRangeChanged.connect(lambda *_: self._redraw_barbs())
        # The labels are placed on a screen-pixel lattice, so they follow the zoom too.
        self.plot.vb.sigRangeChanged.connect(lambda *_: self._redraw_isoline_labels())
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
        # Nothing of the previous field survives into the new one: its frame would read
        # out under the new field's title on a hover, and its contours would be drawn
        # against the new field's coordinates. `refresh_map` supplies both immediately.
        self._frame = None
        self.set_wind(None)             # a new field's wind has not been pushed yet
        self._draw_isolines(None)
        self.set_overlay(overlay)
        self.reset_view()

    def set_overlay(self, overlay):
        """Draw the land fill under the field and the outline layers over it."""
        overlay = overlay or {}
        self._set_land(overlay.get('land') or ())
        for name, item in (('coastline', self.coast), ('borders', self.borders),
                           ('borders_uncertain', self.borders_uncertain)):
            xs, ys = overlay.get(name) or (None, None)
            if xs is not None and len(xs):
                item.setData(xs, ys)
            else:
                item.clear()
        self.overlay_source = overlay.get('source', '')

    def _set_land(self, rings):
        """Fill the land rings pale grey; the sea is the widget's own white background.

        Odd-even, so a ring inside a ring is a hole and not a second island -- which is
        what `tools/build_mapdata.py` relies on by keeping the source's interior rings
        instead of dropping them.
        """
        path = QtGui.QPainterPath()
        path.setFillRule(QtCore.Qt.FillRule.OddEvenFill)
        for xs, ys in rings:
            if len(xs) < 3:
                continue
            polygon = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y))
                                       for x, y in zip(xs, ys)])
            path.addPolygon(polygon)
            path.closeSubpath()
        self.land.setPath(path)
        self.land_rings = len(rings)
        self._land_ref = rings
        if self.terrain_on:
            self._ensure_terrain()         # a new land mask means a new relief image

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
        if getattr(self, 'plot', None) is not None:
            self._wrap_title()
        if getattr(self, 'ds', None) is not None and not self._user_zoomed:
            QtCore.QTimer.singleShot(0, self.reset_view)

    def set_colormap(self, cmap):
        """A pyqtgraph colormap NAME, or a ready-made `ColorMap` (R5's sort scale)."""
        if isinstance(cmap, str):
            try:
                cmap = pg.colormap.get(cmap)
            except Exception:
                cmap = pg.colormap.get('viridis')
        self.cmap = cmap
        self.cbar.setColorMap(self.cmap)

    # ---- drawing ---------------------------------------------------------------
    def set_frame(self, frame, levels=None, label=None, interval=None):
        """Paint one frame. `interval` is an `isolines.Interval` in the frame's own units.

        The contours are drawn here rather than through a call of their own so that they
        cannot be one frame behind what the colours show: there is exactly one place the
        image changes, and the lines are rebuilt from the same array in the same call.
        """
        self._frame = frame
        self.img.setImage(frame, autoLevels=False)
        if levels is not None:
            lo, hi = float(levels[0]), float(levels[1])
            if hi <= lo:
                hi = lo + 1.0
            self.cbar.setLevels(low=lo, high=hi)
        self._draw_isolines(interval)
        if label is not None:
            self.plot.setTitle(label)

    def set_title(self, text):
        """Set the map title, wrapped so it cannot widen the layout (**G36**).

        Separate from `set_frame` because the title names the interval the isolines were
        actually drawn at, which is only known afterwards.
        """
        self.plot.setTitle(text)
        self._wrap_title()

    def _wrap_title(self):
        """G36: a `LabelItem`'s minimum width is the width of its text, and a
        GraphicsLayout widens the whole column to honour it -- swallowing the colorbar
        and, because the ViewBox is aspect-locked, CROPPING the map's latitude to match
        (the **G10** failure again, reached from the other end). Wrapping makes a long
        title take a second line instead of taking the domain.
        """
        label = self.plot.titleLabel
        label.item.setTextWidth(max(240, int(self.width()) - TITLE_MARGIN_PX))
        label.updateMin()

    def _draw_isolines(self, interval):
        if interval is None or self.ds is None or self._frame is None:
            self.isoline.clear()
            self.isoline_heavy.clear()
            self.isoline_levels, self.isoline_step = np.empty(0), 0.0
            self._iso_segments = iso.NO_SEGMENTS
            self._redraw_isoline_labels(force=True)
            return
        drawn = iso.contour_set(self.ds.lon, self.ds.lat, self._frame, interval)
        self.isoline.setData(*drawn['ordinary'])
        self.isoline_heavy.setData(*drawn['emphasised'])
        self.isoline_levels, self.isoline_step = drawn['levels'], drawn['step']
        self._iso_segments = drawn['segments']
        self._iso_decimals = iso.label_decimals(drawn['levels'])
        self._redraw_isoline_labels(force=True)

    # ---- the values on the isolines (R15) -----------------------------------------------
    def set_isoline_labels(self, on):
        """Write each isoline's value on it, or take the values down."""
        self.isoline_labels_on = bool(on)
        self._redraw_isoline_labels(force=True)

    @property
    def isoline_label_count(self):
        return len(self.isoline_label_texts)

    def _label_item(self, i):
        """The `i`th pooled label, made on first use. White-boxed, so the box itself is the
        gap in the line a contour label conventionally sits in, and the number stays
        legible over every colour of the ramp."""
        while len(self._iso_label_pool) <= i:
            item = pg.TextItem('', color='#1c1c1c', anchor=(0.5, 0.5),
                               fill=pg.mkBrush(255, 255, 255, 215))
            item.setZValue(Z_ISOLINE_LABEL)
            item.hide()
            self.plot.addItem(item, ignoreBounds=True)
            self._iso_label_pool.append(item)
        return self._iso_label_pool[i]

    def _redraw_isoline_labels(self, force=False):
        """Place the labels for the frame and the view as they are now."""
        if not force and not self.isoline_labels_on and not self.isoline_label_texts:
            return
        chosen = np.empty(0, dtype=np.intp)
        px = py = None
        if self.isoline_labels_on and self.ds is not None and self._iso_segments[0].size:
            vb = self.plot.vb
            px, py = (float(value) for value in vb.viewPixelSize())
            (x0, x1), (y0, y1) = vb.viewRange()
            key = (id(self._iso_segments), round(px, 12), round(py, 12),
                   round(x0 / px), round(x1 / px), round(y0 / py), round(y1 / py))
            if not force and key == self._iso_label_key:
                return
            self._iso_label_key = key
            chosen = iso.place_labels(self._iso_segments, (x0, x1, y0, y1), px, py)
        else:
            self._iso_label_key = None
        mid_x, mid_y, line, heavy = self._iso_segments
        level = self.isoline_levels
        texts = []
        for i, k in enumerate(chosen):
            item = self._label_item(i)
            text = iso.label_text(level[line[k]], self._iso_decimals)
            font = item.textItem.font()
            if font.bold() != bool(heavy[k]):
                font.setBold(bool(heavy[k]))
                item.setFont(font)
            if item.textItem.toPlainText() != text:
                item.setText(text)
            item.setPos(float(mid_x[k]), float(mid_y[k]))
            item.show()
            texts.append(text)
        for item in self._iso_label_pool[len(chosen):]:
            if item.isVisible():
                item.hide()
        self.isoline_label_texts = texts

    # ---- the user's points (R10) ------------------------------------------------------
    def set_points(self, points, visible=True):
        """Draw `config.Point`s as dots, named when they have a name. -> the colour
        strings Qt did not recognise (those points are drawn in the default blue)."""
        for item in self.point_labels:
            self.plot.removeItem(item)
        self.point_labels = []
        points = list(points or ())
        unknown, brushes = [], []
        for point in points:
            colour = QtGui.QColor(point.colour)
            if not colour.isValid():
                unknown.append(point.colour)
                colour = QtGui.QColor('blue')
            brushes.append(pg.mkBrush(colour))
        if points:
            self.points.setData(x=[p.lon for p in points], y=[p.lat for p in points],
                                brush=brushes)
        else:
            self.points.clear()
        for point in points:
            if not point.name:
                continue
            label = pg.TextItem(point.name, color='#101010', anchor=(-0.12, 0.5),
                                fill=pg.mkBrush(255, 255, 255, 170))
            label.setPos(point.lon, point.lat)
            label.setZValue(Z_POINTS)
            self.plot.addItem(label, ignoreBounds=True)
            self.point_labels.append(label)
        self.show_points(visible)
        return unknown

    def show_points(self, on):
        self.points_shown = bool(on)
        self.points.setVisible(self.points_shown)
        for label in self.point_labels:
            label.setVisible(self.points_shown)

    @property
    def point_count(self):
        return len(self.points.data) if self.points_shown else 0

    def set_marker(self, lat, lon):
        for item in (self.marker, self.marker_halo):
            item.setData([lon], [lat])

    # ---- shaded relief (R9) ------------------------------------------------------------
    def set_terrain(self, on):
        """Show or hide the shaded relief. -> True when it is actually drawn.

        Built lazily: a user who never ticks Topography never pays for the hillshade, and
        one who does pays once -- the image depends on the bundle and the land polygons,
        neither of which changes with the field, the time step or the zoom.
        """
        self.terrain_on = bool(on)
        drawn = self.terrain_on and self._ensure_terrain()
        self.terrain.setVisible(drawn)
        return drawn

    @property
    def terrain_drawn(self):
        return self.terrain.isVisible()

    def _ensure_terrain(self):
        data = terrain.load_terrain()
        if data is None:
            self._terrain_key = None
            return False
        key = (id(data), id(self._land_ref))
        if key == self._terrain_key:
            return True
        shade = terrain.hillshade(data['z'], data['lat'], data['lon'])
        rgba = terrain.relief_rgba(shade, self._land_mask(data))
        self.terrain.setImage(rgba, autoLevels=False)
        x0, _x1, y0, _y1 = terrain.extent(data)
        dlon = float(np.diff(data['lon']).mean())
        dlat = float(np.diff(data['lat']).mean())
        tr = QtGui.QTransform()
        tr.translate(x0, y0)
        tr.scale(dlon, dlat)
        self.terrain.setTransform(tr)
        self._terrain_key = key
        return True

    def _land_mask(self, data):
        """True on land, on the elevation grid -- rasterised from the land polygons.

        The same rings, the same odd-even rule and the same painter as the grey fill, so
        the relief ends exactly where the grey does and the sea stays white whatever the
        sea floor under it looks like. Without rings (an older bundle) it falls back to
        'above sea level', which loses the Dead Sea shore and nothing else.
        """
        path = self.land.path()
        if path.isEmpty():
            return terrain.land_from_elevation(data['z'])
        ny, nx = data['z'].shape
        x0, _x1, y0, _y1 = terrain.extent(data)
        dlon = float(np.diff(data['lon']).mean())
        dlat = float(np.diff(data['lat']).mean())
        image = QtGui.QImage(nx, ny, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor(0, 0, 0, 255))
        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
            # degrees -> pixels: translate the grid's south-west edge to the origin, then
            # scale a cell to one pixel. Row 0 is the SOUTH edge, which is also row 0 of
            # the elevation array, so the mask needs no flip.
            painter.setTransform(QtGui.QTransform.fromTranslate(-x0, -y0)
                                 * QtGui.QTransform.fromScale(1.0 / dlon, 1.0 / dlat))
            painter.fillPath(path, QtGui.QColor(255, 255, 255, 255))
        finally:
            painter.end()
        pixels = pg.functions.imageToArray(image, copy=True, transpose=False)
        return np.asarray(pixels[..., 1]) > 127

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
