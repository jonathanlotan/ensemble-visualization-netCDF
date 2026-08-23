# IMS ICON Ensemble Viewer — Implementation Plan

Desktop app (Windows-first) for exploring IMS ICON-IL **ensemble** NetCDF products.
User flow: download a `.nc.bz2` from the IMS server → open the app → pick the file →
**left = map of the whole domain**, **right = line graph of all 20 ensemble members**
(x = forecast time, y = value) for the grid point selected on the map.

---

## 0. Ground truth — verified against real data on 2026-08-23

Everything in this section was **measured**, not assumed, using
`data/ICON_ENS_2026082300_CAPE_ML.nc` (downloaded from the live server) and
`data/topo_icon_web.nc`. Do not re-derive it; do re-check it if IMS changes the product.

### 0.1 Source of files

| | |
|---|---|
| Server | `https://data.israel-meteo-service.org/ims/` (IIS directory listing) |
| Ensemble folder | `/ims/IMS_ICON_ENSEMBLE/` |
| Docs folder | `/ims/MANUALS/IMS_ICON_ENSEMBLE/IMS_ICON_ENS.pdf` |
| Topography | `/ims/MANUALS/IMS_ICON/topo_icon_web.nc` |
| **Auth** | **NTLM / Negotiate — NOT HTTP Basic.** `curl --ntlm -u user:pass` works; plain `-u` returns 401 |
| Credentials | In the IMS product PDF (`ims_products_*.pdf`). Keep them in env vars `IMS_USER` / `IMS_PASS` or Windows Credential Manager — **never commit them** |
| Retention | ~1 month rolling (30 runs were present: 2026-07-25 → 2026-08-23) |
| Cadence | One run per day, **00 UTC only**, published ~12:20 local |

File name: `ICON_ENS_<YYYYMMDDHH>_<FIELD>.nc.bz2`

### 0.2 The 15 fields (one file each, per run)

| Field | Description | Units | Note |
|---|---|---|---|
| `CAPE_ML` | CAPE of mean surface layer parcel | J kg⁻¹ | **the "cape index"** |
| `T_2M` | 2 m air temperature | K | offer °C display |
| `T_S` | surface soil temperature | K | offer °C display |
| `RELHUM_2M` | 2 m relative humidity | % | |
| `TOT_PREC` | total precipitation | kg m⁻² | **accumulated since model start** |
| `U_10M`, `V_10M` | 10 m wind components | m s⁻¹ | can derive speed/direction |
| `VMAX_10M` | max 10 m gust | m s⁻¹ | max over previous 1 h |
| `CLCT`,`CLCL`,`CLCM`,`CLCH` | total/low/mid/high cloud cover | 0–1 | |
| `ASWDIFD_S`,`ASWDIR_S` | diffuse / direct downward SW radiation | W m⁻² | **averaged since model start** |
| `H_SNOW` | snow depth | m | ~all zeros in summer |

Constant fields live in `topo_icon_web.nc`: `topography_c` (m), `fr_land` (0–1).

### 0.3 File structure (identical for every field)

```
format   NETCDF3_64BIT_OFFSET   (classic, big-endian, NOT HDF5, no compression, no chunking)
dims     time = 121 (UNLIMITED / record dim)   sfc = 20   lat = 261   lon = 161
vars     time(time)             float64  "minutes since 2026-8-23 00:00:00"
         lat(lat)               float64  28.000 .. 34.500 step 0.025  (N)
         lon(lon)               float64  33.000 .. 37.000 step 0.025  (E)
         sfc(sfc)               float64  <all zeros — see gotcha G1>
         <FIELD>_eps(time,sfc,lat,lon)  float32  big-endian
attrs    Conventions=CF-1.6, source=icon-2025.04-dwd, history=<cdo merge …>
```

* 121 hourly steps = **+0 h … +120 h**; model = ICON-IL, 2.5 km, 65 levels, 20 members,
  ECMWF IFS-ENS initial/boundary conditions, IMS radar latent-heat nudging.
* **Compressed ≈ 262 MB → uncompressed exactly 406,770,612 B (~407 MB).**
  All 15 fields of one run ≈ 6 GB uncompressed. Plan cache eviction accordingly.
* Measured: `bunzip2` CLI 16.2 s; Python `bz2` streaming 15.7 s (26 MB/s out).
* No `_FillValue` / `missing_value` on `CAPE_ML_eps`; no NaNs observed. Still handle both.

### 0.4 Byte layout (this is what makes the app fast)

Classic NetCDF stores record variables **interleaved per record**:

```
record_size = Σ vsize(all record vars) = 8 (time) + 3,361,680 (field) = 3,361,688 bytes
offset(t, m, y, x) = var.begin + t*record_size + ((m*261 + y)*161 + x)*4      # dtype '>f4'
```

Build it once with `numpy.memmap` + `numpy.lib.stride_tricks.as_strided`
(shape `(121,20,261,161)`, strides `(3361688, 168084, 644, 4)`), then **every read is a
plain numpy slice** — no decode, no copy, no dask.

Measured on this machine (warm page cache), verified element-for-element against
`netCDF4.Dataset`:

| operation | time |
|---|---|
| header parse | 0.7 ms |
| one map frame `A[t,m]` (261×161) | < 1 ms |
| all-20-member frame `A[t]` (3.4 MB) | < 1 ms |
| point time series `A[:,:,y,x]` (121×20) | < 1 ms |
| **full 407 MB scan** (global min/max) | **0.58 s** |

⇒ No preprocessing/conversion step is needed. Cold-cache first touch costs one SSD read.

### 0.5 Gotchas — each of these will silently corrupt output if missed

* **G1 — `sfc` is the ensemble member axis, not a vertical level.** CDO merged the 20
  per-member files into one "level" axis; it is tagged `long_name="surface"`, `axis="Z"`
  and **all 20 values are 0.0**. Member identity is positional only, recoverable from the
  global `history` attribute (`…_01_CAPE_ML.nc …_02_… … _20_…`). Parse `history` to label
  members; fall back to index+1.
* **G2 — two record variables.** `time` is also a record var, so the record stride is
  3,361,**688**, not 3,361,680. Using the field size alone drifts 8 bytes per time step and
  produces plausible-but-wrong values.
* **G3 — big-endian.** dtype must be `'>f4'` / `'>f8'`, never native `float32`.
* **G4 — non-padded date in time units:** `"minutes since 2026-8-23 00:00:00"` (single-digit
  month). `datetime.fromisoformat` **fails**. Use a tolerant regex or `cftime.num2date`.
* **G5 — accumulated fields.** `TOT_PREC`, `ASWDIFD_S`, `ASWDIR_S` accumulate/average from
  model start. Plot raw *and* offer `np.diff` along time for hourly rates; label clearly.
* **G6 — `topo_icon_web.nc` is a different, smaller grid**: lat 29.0–34.0 (201), lon
  34.0–36.0 (81), same 0.025° spacing, and it aligns on **exact integer offsets** —
  ensemble rows 40..240, cols 40..120 (verified). Its `fr_land` gives a free offline
  coastline, **but only inside that inner box**; the full domain (28–34.5 N, 33–37 E) needs
  a bundled coastline (see step 4.3).
* **G7 — NTLM auth** (see 0.1). `requests` needs `requests-ntlm`; `urllib` won't do it.
* **G8** — `lat` is ascending (28→34.5), so `imshow` needs `origin='lower'`. Aspect ratio
  must be `1/cos(lat)` ≈ 1.17 or the map looks squashed.

---

## 1. Architecture decision

**Stack: Python 3.12 + PySide6 (LGPL) + pyqtgraph (MIT) + NumPy.**

* Native desktop app → matches "open the program, select the file" on Windows.
* pyqtgraph redraws an `ImageItem` and 20 curves at 60 fps → the time slider can be
  *scrubbed*, which is the single biggest usability win over the current IMS PNGs.
* **No netCDF4 / HDF5 / xarray / scipy at runtime.** A ~150-line pure-NumPy NetCDF-3
  reader (already prototyped and validated) removes the #1 Windows packaging headache
  (HDF5 DLLs) and cuts the frozen exe by >100 MB. Keep `netCDF4` as a **dev-only** test
  dependency to assert the reader stays correct, and as an optional runtime fallback if a
  file turns out to be HDF5-based (`\x89HDF` magic) rather than `CDF\x01/\x02`.

```
imsicon/
  nc3.py         # NetCDF-3 header parser + strided memmap  (no deps but numpy)
  dataset.py     # EnsembleFile: coords, members, units, stats cache
  ingest.py      # .bz2 detect/decompress/cache, file picker plumbing
  stats.py       # ensemble mean/spread/percentile/probability, global min-max
  geo.py         # coastline + topography overlay, lat/lon <-> pixel
  ui/main.py     # QMainWindow: map (left) | plot (right) | time slider
  ui/mapview.py
  ui/plotview.py
  download.py    # optional NTLM fetcher
tests/
data/            # gitignored — 407 MB files live here
```

---

## 2. Step-by-step implementation

### Phase 0 — Project skeleton  *(0.5 h)*
1. `python -m venv .venv`; `pip install pyside6 pyqtgraph numpy`; dev: `pytest netCDF4`.
2. `pyproject.toml`, `.gitignore` (`data/`, `*.nc`, `*.nc.bz2`, `.venv/`, `*.cache`).
3. `git init` — the folder is not yet a repo.
**Done when:** `python -m imsicon` opens an empty window on Windows and macOS.

### Phase 1 — `nc3.py`, the reader  *(3 h — the load-bearing piece)*
1. Parse the classic header: magic+version, `numrecs`, dim list, global attrs, var list
   (`begin` is `>u4` for version 1, `>i8` for version 2 / 64-bit-offset).
2. `record_size = Σ vsize` over record vars (**G2**).
3. `view(name)` → `as_strided` over a `np.memmap`, big-endian dtype (**G3**).
4. Convenience: `field_name` (the single `*_eps` var), `coords()`, `members()` from
   `history` (**G1**), `valid_times()` via tolerant unit parsing (**G4**).
5. Guard: if magic is `\x89HDF`, raise `UnsupportedFormat` and let the UI suggest the
   netCDF4 fallback path.
**Done when:** `tests/test_nc3.py` asserts, for the CAPE file, `np.array_equal` against
`netCDF4.Dataset` for time steps 0/1/47/120 and 200 random `(t,m,y,x)` points, plus a
regression test pinning `record_size == 3_361_688`.

### Phase 2 — `dataset.py`, the domain model  *(2 h)*
1. `EnsembleFile`: `.field`, `.units`, `.long_name`, `.lat/.lon`, `.times` (datetime),
   `.run_init`, `.n_members`, `.member_labels`.
2. Accessors: `frame(t, member=None)`, `ens_frame(t)`, `series(y, x)` → `(121, 20)`,
   `nearest_index(lat, lon)`.
3. **Sidecar stats cache** `<file>.imsstats.json`: global min/max, per-timestep
   ensemble min/max/mean, computed by one 0.58 s background scan; keyed by file
   size+mtime so it invalidates itself.
**Done when:** opening a file yields a populated header panel in < 100 ms and the stats
scan finishes off the UI thread with a progress bar.

### Phase 3 — `ingest.py`, getting from download to memmap  *(2 h)*
1. File dialog filter: `*.nc *.nc.bz2`; also accept drag-and-drop and a CLI argument.
2. If `.bz2`: stream-decompress to a cache dir
   (`%LOCALAPPDATA%\IMSIconViewer\cache` on Windows) with a cancellable progress dialog
   (~16 s, 407 MB). Reuse the cached `.nc` if it already exists and sizes match.
3. **Disk guard:** refuse if free space < 1 GB; LRU-evict the cache above a configurable
   budget (default 4 GB — one full run is ~6 GB).
4. Parse run date + field from the filename; fall back to header attrs.
**Done when:** picking a freshly downloaded `.nc.bz2` renders the map, and picking it a
second time is instant.

### Phase 4 — Map panel (left)  *(4 h)*
1. `pyqtgraph.ImageItem` fed by `frame()`, `origin='lower'`, aspect `1/cos(31°)` (**G8**),
   `QTransform` mapping array indices → degrees so axes read in lat/lon.
2. **Aggregation selector** — what the map shows across the 20 members:
   `member N` | `mean` | `max` | `min` | `spread (max−min)` | `P(value > threshold)`.
   The probability view is the thing an ensemble is *for* and is what makes this "better".
3. Overlays: coastline + borders. Use `fr_land==0.5` contour from `topo_icon_web.nc`
   inside the inner box, and bundle a trimmed Natural Earth 1:10 m coastline/border
   GeoJSON (public domain, ~50 KB, clipped to 27–35 N / 32–38 E) for the full domain
   (**G6**). Optional `topography_c` hillshade underlay, off by default.
4. Colormap per field (turbo/viridis for CAPE, blues for precip, RdBu for temperature),
   colorbar, and a **fixed vs per-frame scale** toggle — fixed to the global max by
   default so scrubbing time doesn't rescale colors under the user.
5. Hover readout: lat, lon, value, member; crosshair.
6. **Time slider + play/pause** under the map, labelled with both valid UTC time and
   `+N h`; keyboard ←/→ steps hours, shift = 6 h.
**Done when:** scrubbing all 121 steps stays above ~30 fps and the coastline visually
matches the land/sea CAPE gradient.

### Phase 5 — Line graph panel (right)  *(3 h)*
1. On map click/drag: `series(y, x)` → 20 curves, x = valid time (or `+h`, toggle).
2. Overlays: ensemble **mean** (thick black), **min–max envelope** (shaded),
   optional P10/P50/P90. A red vertical line marks the map's current time step;
   dragging that line scrubs the map — the two panels are bidirectionally linked.
3. **Y-axis scaling** — see assumption A1: default **fixed to the dataset-wide max**
   (e.g. 3252 J/kg for the test file) so every point is compared on one scale;
   toggle to "fit this point" and to a manual max.
4. Distinguishable member colors (20-class qualitative ramp), legend, hover highlights
   one member in both panels, click-to-pin. Right-click → copy series as CSV.
5. Empty state before a point is chosen: show the domain-max series across all members.
**Done when:** clicking the CAPE hot spot at 34.075 N / 35.800 E reproduces
`poc_cape_ensemble.png` — 4 diurnal convective peaks, max 3252 J/kg at +110 h.

### Phase 6 — Shell & polish  *(3 h)*
`QSplitter` (map | plot), toolbar (open, field, aggregation, colormap, export),
status bar (file, run init, field, units, member count), recent-files list,
`QSettings` persistence, export PNG of either panel, export CSV of the series,
and a `Ctrl+E` "export current view" that writes both plus a metadata header.

### Phase 7 — Multi-field / multi-run *(stretch)*
Open several fields of the same run side by side (they share the grid and time axis, so
the same `(t, y, x)` selection applies), and overlay the same point from consecutive runs
to show run-to-run consistency. Derive wind speed from `U_10M`/`V_10M`, °C from K,
and hourly rates from accumulated fields (**G5**).

### Phase 8 — Optional built-in downloader *(stretch)*
`requests` + `requests-ntlm` (**G7**), credentials from env vars or Windows Credential
Manager via `keyring`, listing scraped from the IIS HTML index, resumable
`Range` downloads, checked against the size in the listing. Never hardcode the password.

### Phase 9 — Windows packaging  *(2 h)*
1. PyInstaller one-folder build (faster start than one-file for a 400 MB-mapping app);
   `--noconsole`, icon, `--collect-all pyqtgraph`.
2. Verify: `numpy.memmap` on a 407 MB file, long paths, spaces in paths, non-ASCII user
   names, and that `%LOCALAPPDATA%` cache creation works without admin rights.
3. Optional Inno Setup installer + `.nc`/`.nc.bz2` file association so double-clicking a
   download opens the app.
**Done when:** the build runs on a clean Windows 11 VM with no Python installed.

### Phase 10 — Tests & QA
`pytest` on: header parse, record stride, endianness, member extraction from `history`,
tolerant time parsing, coordinate↔index round-trip, stats-cache invalidation, and one
golden-image test of the map render. Keep a tiny synthetic 2-member/3-step NetCDF-3
fixture in `tests/` so CI never needs the 407 MB file.

---

## 3. Assumptions (flag to the user if any is wrong)

* **A1 — "x being time, z being value (max z being max value)".** Read as: y-axis = the
  field value, with the axis top pinned to the maximum value so all points share one
  scale. Implemented as a toggle (dataset max / point max / manual), defaulting to
  dataset max. If "z" instead meant a third dimension (e.g. a color axis or a vertical
  level) this needs revisiting — but these files have no vertical dimension, only the
  20-member axis.
* **A2 — "all of the runs" = the 20 ensemble members** (the PDF calls them
  "20 חברי אנסמבל" / "20 members"), not the 30 daily runs on the server. Cross-run
  comparison is Phase 7.
* **A3 — "map of the whole section" = the whole model domain** (28–34.5 N, 33–37 E),
  with zoom/pan available.

## 4. Test assets already in `data/` (gitignored)

* `ICON_ENS_2026082300_CAPE_ML.nc.bz2` + `.nc` — the reference file for all the numbers above.
* `H_SNOW.nc.bz2` + `.nc` — degenerate all-zero case (125 KB → 407 MB), good for
  testing empty-data rendering and for cheap schema checks.
* `topo_icon_web.nc` — coastline/terrain source.
* `poc_cape_ensemble.png` — the proof-of-concept render Phase 5 must reproduce.
* `prototype/nc3.py` — the validated reader from section 0.4; Phase 1 starts by moving it
  to `imsicon/nc3.py` and writing the netCDF4 equivalence test around it.

## 5. Working agreements

* Never commit `data/`, IMS credentials, or `.nc` files.
* Any change to `nc3.py` must keep `tests/test_nc3.py` green against `netCDF4` — that
  test is the contract that the byte math (**G2/G3**) stays correct.
* Prefer measuring over guessing: this file's numbers came from real reads; keep it that way.

---

# Release 1 — the interactive viewer

Scope: four items requested on 2026-08-23 — **B1** (crash fix), **F1** (choose the file at
startup), **F3** (zoomable + clickable map), **F4** (hover readout). Together they turn the
Phase 4/5 sketch above into a working two-panel app, so Release 1 implements
`imsicon/` for real. Section 0 stays the contract: all byte math comes from there.

## R1.0 Layout

```
┌──────────────────────────────┬──────────────────────────────────┐
│                              │  READOUT  (F4 — right of the map)│
│   MAP  (F3)                  │  time (Z) · mean · max · min     │
│   zoom = wheel               │  P90 · P10                       │
│   click = pick point         ├──────────────────────────────────┤
│                              │  GRAPH — 20 members, x = time    │
│  [time slider] [◀ ▶] [play]  │  hover anywhere → readout above  │
└──────────────────────────────┴──────────────────────────────────┘
```

The readout sits at the top of the right-hand column: literally to the right of the map,
directly above the graph it describes, so the eye travels hover → numbers without
crossing the window.

## R1.1 Modules to create

| file | responsibility |
|---|---|
| `imsicon/nc3.py` | promoted from `prototype/nc3.py`, plus `UnsupportedFormat`, tolerant time parsing (**G4**), member labels from `history` (**G1**) |
| `imsicon/dataset.py` | `EnsembleFile` — coords, times, `frame/agg_frame/series/nearest_index`, lazy global range |
| `imsicon/ingest.py` | `.bz2` → cache dir decompress with progress; LRU eviction; disk guard |
| `imsicon/geo.py` | topo loader + dependency-free marching-squares coastline from `fr_land` |
| `imsicon/ui/mapview.py` | `MapView` — image, colorbar, coastline, crosshair, wheel zoom, click-to-pick |
| `imsicon/ui/plotview.py` | `PlotView` — 20 curves, mean, envelope, hover line, time cursor |
| `imsicon/ui/readout.py` | `ReadoutPanel` — the six F4 statistics |
| `imsicon/ui/main.py` | `MainWindow` — splitter, toolbar, slider, threads, status bar |
| `imsicon/__main__.py` | `python -m imsicon [file]` |

## R1.2 Step-by-step

### B1 — `prototype/nc3.py` crashes with no argument  *(10 min)*
`sys.argv[1]` raises `IndexError` when the module is run bare. Replace the `__main__`
block with an `argparse` front end: optional positional `path`; when omitted, auto-discover
the newest `data/*.nc`; if nothing is found, print usage and `raise SystemExit(2)`.
Keep the same demo output (header, variables, timing) so it stays a smoke test.
**Done when:** `python prototype/nc3.py` with no arguments prints the CAPE header instead
of a traceback, and `python prototype/nc3.py /nope.nc` exits 2 with a readable message.

### F1 — Choose the file at startup  *(1.5 h)*
1. `MainWindow` builds empty, shows, then fires `QTimer.singleShot(0, self.open_dialog)`
   so the picker appears over a drawn window rather than a grey rectangle.
2. `QFileDialog.getOpenFileName` filtered to `ICON ensemble (*.nc *.nc.bz2)`, starting in
   the last-used directory (`QSettings`), falling back to `./data`.
3. Cancel is not fatal: fall back to an empty state that says *Open a file* and keeps the
   toolbar `Open…` button, `Ctrl+O`, drag-and-drop, and recent-files list live.
4. `python -m imsicon <path>` skips the dialog — a `.nc.bz2` file association can then
   open the app by double-click (Phase 9).
5. `.nc.bz2` input runs `ingest.decompress` on a `QThread` with a modal cancellable
   progress dialog (~16 s / 407 MB), writing to the cache dir and reusing an existing
   cached `.nc` whose size matches.
6. Loading is guarded: `UnsupportedFormat`, missing `*_eps` variable, and unreadable files
   produce a `QMessageBox` and return to the empty state instead of dying.
**Done when:** launching with no arguments presents the picker; choosing the CAPE `.nc`
paints map + graph; choosing the `.bz2` shows progress then the same result; Cancel leaves
a usable window.

### F3 — Zoomable, clickable map  *(2 h)*
1. `pg.setConfigOption('imageAxisOrder', 'row-major')` before any `ImageItem`, then a
   `QTransform` of `translate(lon0 - dlon/2, lat0 - dlat/2) · scale(dlon, dlat)` so the
   image sits in true degrees and pixel centres land on grid points.
2. Wheel zoom is the ViewBox default; keep it and add: aspect locked so a degree of
   longitude gets `cos(lat)` of the pixels a degree of latitude gets (**G8**; in
   pyqtgraph's convention that is `setAspectLocked(True, ratio=cos(lat))`), generous pan
   limits (**G10**), `setMouseEnabled` for drag-pan, and **Reset view** + `Home`.
3. Click anywhere: `scene().sigMouseClicked` → `vb.mapSceneToView` → `nearest_index`,
   clamped to the domain, emitting `pointPicked(iy, ix)`. Clicks outside the image
   snap to the nearest edge cell rather than being ignored.
4. A crosshair marker pins the chosen cell; hover updates the status bar with
   lat/lon/value under the cursor.
5. Aggregation combo (`member N | mean | max | min | spread`) drives what the map shows;
   the colorbar is fixed to the dataset range by default so scrubbing does not rescale.
**Done when:** wheel zoom is smooth at all 121 steps, and clicking a point redraws the
right-hand graph for that point in well under a frame.

### F4 — Hover readout  *(1.5 h)*
1. `PlotView` tracks `sigMouseMoved`, converts x → nearest time index, and emits
   `hovered(t_index)`; leaving the plot emits `hovered(-1)`, which reverts the panel to
   the map's current time step so it is never blank.
2. A dashed vertical line follows the hover; the solid red line stays on the map's time.
3. `ReadoutPanel` shows, for the selected point at that time, across the 20 members:
   **date/time in Zulu** (`2026-08-27 14:00Z`, with `+110 h` from run init),
   **ensemble mean**, **maximum**, **minimum**, **top 90 % (P90)**, **bottom 90 % (P10)**.
   Percentiles are `np.percentile(..., [10, 90])` — linear interpolation over 20 members,
   so P90 means "9 in 10 members fall below this".
4. Values are formatted per field: significant digits from the data range, units from the
   header, and a fixed-width font so the numbers do not jitter while the mouse moves.
**Done when:** dragging along the graph updates all six readouts every frame without
recomputation lag, and the times shown match `time` in the file exactly.

## R1.3 Verification

* `tests/test_nc3.py` — netCDF4 equivalence, `record_size == 3_361_688`, argv guard.
* `tests/test_dataset.py` — time parsing (**G4**), member labels (**G1**), `nearest_index`
  round-trip, percentile maths against `np.percentile` on a known slice.
* `python -m imsicon --screenshot out.png <file>` — a headless dev flag that loads a file,
  picks a point, renders, and saves a PNG. Keeps the UI verifiable without a human.

## R1.4 Status — shipped and verified 2026-08-23

All four items are implemented, and `42 passed` covers them:

| item | where | verified by |
|---|---|---|
| **B1** argv crash | `prototype/nc3.py` argparse front end | `test_prototype_cli_runs_without_arguments`, `..._rejects_a_missing_file` |
| **F1** choose file at startup | `ui/main.py` `open_dialog` / `open_path`, `ingest.py` | `test_startup_without_a_file_asks_for_one`, `..._cancelling_...`, `..._unreadable_file_...`, `tests/test_ingest.py` |
| **F3** zoom + click map | `ui/mapview.py` | `test_map_click_moves_the_graph_to_that_point` (real `QTest.mouseClick`), `..._click_outside_...`, `..._wheel_zoom_...`, `..._reset_view_...` |
| **F4** hover readout | `ui/plotview.py`, `ui/readout.py` | `test_hover_shows_the_six_statistics_for_that_time`, `..._mouse_move_...` (real `QTest.mouseMove`), `..._leaving_the_plot_...` |

End-to-end runs, both `QT_QPA_PLATFORM=offscreen` and native cocoa:
`.nc` opens and renders; `.nc.bz2` decompresses to the cache in ~16 s (18.6 s to first
paint) and reuses the cache on the next open.

## R1.5 Gotchas found while building (they cost real time — do not rediscover them)

* **G9 — macOS: a venv named `.venv` breaks Qt entirely.** Symptom:
  `Could not find the Qt platform plugin "cocoa" in ""`, even though `libqcocoa.dylib` is
  present and `ctypes.CDLL` loads it. Cause: `python -m venv` leaves the macOS `UF_HIDDEN`
  flag on the tree, Qt's plugin loader skips hidden files, so it enumerates **zero**
  plugins (`QDir(...).entryList()` returns just `.` and `..`). Fix: name the venv `venv/`
  and run `chflags -R nohidden venv` after creating it or installing into it.
  Windows and Linux are unaffected — this is a macOS-dev-only trap.
* **G10 — an aspect-locked ViewBox with tight pan limits crops instead of padding.**
  Fitting 4° of longitude by 6.5° of latitude needs longitude slack; with limits set to
  domain + 2 % pyqtgraph could not widen x, so it silently shrank y and showed ~two thirds
  of the domain. Pan limits are now domain ± 50 % of the span.
* **G11 — pyqtgraph calls `resizeEvent(None)` from `GraphicsView.__init__`**, before a
  subclass's own attributes exist. Any override must use `getattr(self, 'ds', None)`.
* **G12 — a pyqtgraph axis auto-applies SI prefixes.** With `units='J kg-1'` the y-axis
  relabelled itself `kJ kg-1` while the readout said `J kg-1`. Call
  `enableAutoSIPrefix(False)` on any axis whose units come from a file.
* **G13 — LRU touch must not move `mtime`.** The cache reuse path originally called
  `os.utime(path, None)`, which rewrote mtime and so invalidated the `.imsstats.json`
  sidecar (keyed on size+mtime) on every reopen. It now touches atime only, in `ns` form
  so no sub-microsecond drift creeps in. Caught by `test_decompress_...reuses_it`.

## R1.6 Running it

```bash
python -m venv venv && chflags -R nohidden venv     # macOS: see G9
venv/bin/python -m pip install -r requirements.txt        # or requirements-dev.txt to run the tests
venv/bin/python -m imsicon                          # asks for a file (F1)
venv/bin/python -m imsicon data/ICON_ENS_2026082300_CAPE_ML.nc.bz2
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest tests -q
```

Controls: **wheel** zooms the map, **drag** pans, **click** picks the grid point that
feeds the graph, **Home** / *Reset view* refits the domain, the slider and ←/→
(shift = 6 h) move through the 121 forecast hours, **Play** animates, and hovering the
graph fills the readout. Dropping a file on the window opens it.
