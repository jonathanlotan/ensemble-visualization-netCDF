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

**All 15 units strings MEASURED 2026-08-24** against run `2026082400`, with
`tools/sniff_headers.py --all` (60 MB of prefixes, not 3.9 GB of files). The `units`
column below is now the literal string in the file header, not the PDF's rendering of it.
Two of them are **not** what this file previously claimed — see the bold notes.

| Field | Description | `units` (verbatim) | Note |
|---|---|---|---|
| `CAPE_ML` | CAPE of mean surface layer parcel | `J kg-1` | **the "cape index"** |
| `T_2M` | 2 m air temperature | `K` | displayed as °C by default (v2) |
| `T_S` | weighted surface temperature | `K` | displayed as °C by default (v2) |
| `RELHUM_2M` | relative humidity in 2m | `%` | |
| `TOT_PREC` | total precip | `kg m-2` | **accumulated since model start**; `sum` kind (**G14**) |
| `U_10M`, `V_10M` | zonal / meridional wind in 10m | `m s-1` | can derive speed/direction |
| `VMAX_10M` | gust at 10 m *since end of previous full 01H interval* | `m s-1` | already per-interval — **never de-accumulate** |
| `CLCT`,`CLCL`,`CLCM`,`CLCH` | total/low/mid/high cloud cover | `%` | **⚠ CORRECTION: these are 0–100 %, NOT the 0–1 fraction previously documented here.** Measured range 0..100 on every one. Still decided from the data at load (**G22**), because the header alone cannot settle it |
| `ASWDIFD_S`,`ASWDIR_S` | Surface down solar diff./direct rad. *mean since model start* | `W/m**2` | **⚠ the spelling is `W/m**2`, not `W m-2`** — handled by the **G21** normaliser, not by widening the registry. `mean` kind: `np.diff` is wrong (**G14**) |
| `H_SNOW` | weighted snow depth | `m` | ~all zeros in summer (measured max 2.0 mm) |

The `long_name`s settle the two accumulation kinds beyond doubt: the radiation fields say
**"mean since model start"** and `VMAX_10M` says **"since end of previous full 01H
interval"**. Neither is a guess any more.

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
* **G26 — a file shorter than its header claims is a SEGFAULT, not an exception.**
  `numrecs` lives in the header, so an interrupted download (or a deliberately truncated
  prefix) leaves a `.nc` that parses as 121 steps while holding 9. `nc3.view` builds the
  window with `as_strided`, which does **not** bounds-check, so touching the missing tail
  crashes the interpreter with nothing to catch. `nc3.parse` now clamps `numrecs` to the
  records the file actually holds (keeping `declared_numrecs` and `truncated` for
  diagnostics), and `nc3.view` drops a partial trailing element so the dtype cast cannot
  fail. `EnsembleFile.truncation_note` tells the user the file stops early.

## R1.7 Gotchas from v2 (units and de-accumulation) — promoted here as the durable record

* **G14 — `TOT_PREC` is accumulated; `ASWDIFD_S`/`ASWDIR_S` are AVERAGED. The formula
  differs, and `np.diff` is wrong for radiation.** With hours-since-init `h[t]`:
  `sum` kind → `A[t] − A[t−k]`; `mean` kind → `(A[t]·h[t] − A[t−k]·h[t−k]) / (h[t] − h[t−k])`.
  Measured on a synthetic running mean of a known hourly signal `[0,100,400,800,300,50]`:
  `np.diff` returns `[100, 150, 183.3, −33.3, −70]` — plausible-looking W m⁻² and wrong —
  while the `mean` formula returns the true signal exactly. Detection guard: an accumulated
  field is monotonically non-decreasing; an averaged one is not.
* **G15 — an affine conversion must not apply its offset to a difference.** K→°C is
  `a=1, b=−273.15`; on a difference only `a` contributes. A 5 K rise is a 5 °C rise, not
  −268.15 °C. Note the asymmetry with G14: an averaged field's window value is a *mean*,
  which is absolute and takes the **full** affine — only the `sum` branch is a difference.
  `spread` (max−min) is a difference too, which is why `FieldView.agg_frame` converts the
  member stack *before* aggregating rather than after.
* **G19 — the `.imsstats.json` sidecar is keyed by transform signature.** A rate view
  cannot reuse the raw range. Schema v2 is
  `{"schema": 2, "key": {...}, "ranges": {"raw": {...}, "rate:sum:1": {...}}}`; v1 blobs
  (a bare `{min, max}`) are read as `{"raw": ...}` so no cache is lost. A **unit** change
  never rescans: the cached range is stored pre-units and transformed affinely on read.
* **G20 — `float(np.nanmax(frame)) or 1.0` does not do what it looks like.** `bool(nan)` is
  `True`, so `nan or 1.0` is `nan`, and a downstream `if hi <= lo` guard never fires because
  `nan <= 0.0` is `False`. Fixed with explicit `np.isfinite` checks (`_finite_max/_finite_min`
  in `ui/main.py`). Not theoretical: the first `k` steps of a rate view are legitimately
  all-NaN.
* **G21 — units strings have many spellings; compare normalised, and never raise.**
  `transform.normalise_units` strips/lowercases, deletes `**` and `^`, rewrites `a/b` to
  `a b-1`, then aliases to a canonical token. It must be **total** — an unrecognised string
  returns itself and lands in the "no conversion" branch. A units string must not be able to
  crash a file open.
* **G22 — cloud-cover encoding is undecidable from the units string alone.** A file may say
  `1` and store 0–100. Decide from the data range: any value > 1 ⇒ stored as %; max ≈ 1 with
  spread ⇒ fraction, offer ×100; all ≈ 0 (clear sky) ⇒ **undecidable, refuse to convert**.
  This is a permanent runtime guard, not a one-off check.
* **G23 — difference in float64, not float32.** Differencing two large near-equal
  accumulations is catastrophic cancellation. The upcast is unconditional and costs nothing
  at these array sizes.
* **G24 — clamp float noise, surface real negatives.** A `sum`-kind rate must be ≥ 0.
  Clamp `|x| < eps` (eps = 8 ulp at the accumulation's magnitude) to 0; anything larger is
  evidence the field is not actually accumulated, so it is **surfaced as a warning** — else
  the fix for G23 hides the G14 bug it exists to expose.
* **G25 — QSettings in tests writes the developer's real preferences.** v2 persists a units
  choice per field, so an unisolated test run silently changes what the app shows on the next
  real launch, and reads back whatever a previous run left behind. `tests/conftest.py` now
  redirects `QSettings` to a temp dir and clears it per test.

---

# Release 2 — units and de-accumulation (v2)

Shipped 2026-08-24. Plan and rationale live in `v2.md`; this section is the durable record
of what exists. **A** unit conversion (°C by default), **B** correct units for every field,
**C** de-accumulation to 1 h / 3 h windows.

## R2.1 The one transform layer

Both A and C are transformations *between the file and the screen*, and there were already
four consumers of raw values — the map, the graph, the readout and the status bar. Applying
conversions at each call site is four chances to drift, and the failure mode (map in °C,
readout in K) is worse than no conversion at all. So there is exactly one place values are
produced:

```
memmap ─► EnsembleFile ──────────────► raw physical values   (R1 code, UNMODIFIED)
               │                        tests/test_dataset.py stays green untouched
               ▼
          FieldView  ◄── Transform (rate ∘ units)
               │
    ┌──────────┼──────────┬───────────────┐
    ▼          ▼          ▼               ▼
 MapView   PlotView   ReadoutPanel   status bar
```

`main.py` wraps the dataset in one line (`FieldView(EnsembleFile(path))`) and `MapView` /
`ReadoutPanel` needed no change at all. `EnsembleFile` stays pure as the raw-truth layer,
which is what lets every transform be tested by round-tripping against it.

**Ordering rule — do not reorder, the transforms do not commute:**

```
raw ──► [1] time-differencing ──► [2] unit conversion ──► display
```

and step [2] is *told* what step [1] produced: a window **sum** is a difference and takes
`apply_delta` (scale only, **G15**); an instantaneous value or a window **mean** is absolute
and takes the full affine.

| file | responsibility |
|---|---|
| `imsicon/transform.py` | `Affine`, `normalise_units` (**G21**), the units registry, `ACCUMULATION`, the two **G14** window formulas |
| `imsicon/fieldview.py` | `FieldView` — the decorator; `__getattr__` delegates everything not transformed |
| `tests/synth.py` | a real NetCDF-3 64-bit-offset **writer**, so 13 untested fields become testable without the 407 MB file |
| `tools/sniff_headers.py` | dev-only: settle a field's units from a 4 MiB prefix instead of a 262 MB download |

## R2.2 Registry policy — field name is the key, units string is a guard

If a file's units string does not match what the registry expects for that field, the app
**warns and offers no conversion**, falling back to R1 behaviour. A mismatch means IMS
changed something, and the safe response to "my assumption may be stale" is to stop
converting, not to guess. An unknown field likewise opens normally with no conversion.

Defaults: **°C** for `T_2M`/`T_S`, **mm** for `TOT_PREC` (exact — 1 kg m⁻² of water is 1 mm,
so only the label changes), **cm** for `H_SNOW`, **%** for cloud *when the data says it is a
fraction* (**G22**). `VMAX_10M` stays in file units — guessing a forecaster wants knots is a
preference, not a fact.

## R2.3 De-accumulation

Only `TOT_PREC` (`sum`), `ASWDIFD_S` and `ASWDIR_S` (`mean`) accumulate; for everything else
the Rate control is **disabled, not hidden** (a stable layout beats a jumping toolbar).
The window is computed in *steps* — `round(hours / median(diff(forecast_hours)))` — never
assuming 1 step = 1 hour. The first `k` steps are **NaN, not a partial window**: losing 3 of
121 steps is nothing, while a silently-partial "3-hourly" total is a misread waiting to
happen, so the curves use `connect='finite'` and the gap renders as a gap. Switching into a
rate mode jumps to `t = k` so nobody lands on a blank map and concludes the app is broken.

The units label follows the kind and is **not cosmetic**: `sum` k=1 → `mm h-1`, k=3 →
`mm/3h`, while `mean` stays `W m-2` (a window mean is still a mean). `label_for` names the
window — `2026-08-27 14:00Z (+110 h) [1 h to 14:00Z]` — because a rate is **backward
looking** and a reader who takes it as instantaneous is off by one interval.

## R2.4 Measured, not assumed

| claim | measurement |
|---|---|
| 4 MiB of a `.nc.bz2` yields the header + 2 time steps; 16 MiB yields 8 | reproduced 2026-08-24: 1,637,625 / 8,177,912 / 17,475,602 / 27,221,957 B |
| a prefix frame is the real file's frame | `np.array_equal` against the full 407 MB file, t=0 |
| `np.diff` is wrong for an averaged field | see **G14** — recovers `[100,150,183,−33,−70]` instead of `[100,400,800,300,50]` |
| the rate path stays inside a 60 fps scrub | on the real CAPE file: identity 0.12 ms, affine 0.33 ms, 1 h rate frame 3.47 ms, 3 h aggregated map 4.77 ms, point series 0.04 ms — budget is 16.7 ms |
| no frame caching is needed | ⇒ confirmed; R1's "every read is a plain numpy slice" is intact |
| **all 15 units strings** | measured against run `2026082400` — §0.2 is now observation, not documentation |
| the `mean` formula on **real** `ASWDIR_S` | 23 real time steps: 0 at night, peak **1000.7 W m⁻² at 10 UTC** (local solar noon ≈ 09:40 UTC at 35°E), back to 0 by 17 UTC |
| `np.diff` on the same real data | peaks at **76 W m⁻²** and goes **negative all afternoon** — impossible for a downward flux, and *plausible-looking all morning*, which is what makes G14 dangerous |
| the `sum` round-trip on **real** `TOT_PREC` | `cumsum(hourly) − stored` max abs error **0.0**; 3 h rate exactly `A[t] − A[t−3]` |

**Data quirk worth knowing:** on real `ASWDIR_S`, about **74 cells in 18.5 million**
(0.0004 %) have a stored running mean whose *implied cumulative total decreases*, which a
non-negative flux cannot do. That is in the file, not in the arithmetic. The rate view
surfaces them as a warning rather than clamping them away (**G24**); the float32 rounding
noise around them — 4.7 M values at ≈ −1e-4 — is clamped silently.

## R2.5 Running it

```bash
venv/bin/python -m imsicon                                    # ask for a file
venv/bin/python -m imsicon --units C data/ICON_ENS_..._T_2M.nc
venv/bin/python -m imsicon --rate 1h data/ICON_ENS_..._TOT_PREC.nc
venv/bin/python -m imsicon --screenshot out.png --units C --rate 1h <file>
python tools/sniff_headers.py --local data/*.nc.bz2           # offline
IMS_USER=... IMS_PASS=... python tools/sniff_headers.py --all --deep    # measures §0.2
```

Toolbar row 2 carries **Units** and **Rate**; each is disabled when the registry says it
does not apply. A units choice is remembered per field, so Kelvin is never shown unless
asked for.

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

---

# Release 3 — downloader, dew point, difference map (v3)

Scope: three items requested on 2026-08-25 — **D1** a downloader that lets the user choose
a map (CAPE, precipitation or any of the other 13), **D2** an option that *visualizes and
writes* the dew point temperature from temperature and humidity, and **D3** a map that
shows the difference. Sections 0 (byte math), R1 (the two-panel viewer) and R2 (the one
transform layer) are unchanged and still the contract.

**A1 (v3) — "a map that shows the difference".** Read as the difference between two
fields of one run, with the **dew point depression `T_2M − TD_2M`** as the flagship case,
since it is what the dew point item makes newly possible and it is the quantity a
forecaster actually reads. The implementation is the general `A − B`, so it also covers
"the same field two ways". If "the difference" meant something else — between two *runs*,
or between a member and the ensemble mean — the pairing rule changes but `DifferenceView`
does not; see R3.6.

## R3.1 The shape of it

Every item is the same problem twice over: **one run's files belong together.** The
downloader fetches several fields of one run, and both derived fields combine two files of
one run. So there is one discovery function (`ingest.scan_for_fields`), one pairing check
(`derived.check_pairable`), and one view base class.

```
        IMS server  ──download.py──►  ICON_ENS_<run>_<FIELD>.nc.bz2
                                              │ ingest.decompress (R1, unchanged)
                                              ▼
                                      EnsembleFile ──► FieldView       (R1 / v2, unchanged)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              DewPointView              DifferenceView             ncwrite.py
           (T_2M + RELHUM_2M)               (A − B)            write it back as .nc
                    └──────────► both mirror FieldView's surface ◄──────────┘
                                              │
                              MapView · PlotView · ReadoutPanel · status bar
                                     (no special case anywhere)
```

The last line is the design claim, and `tests/test_ui_derived.py` is what holds it: the
window installs a derived view through the same `_install` the file path uses, and every
panel keeps working. `MainWindow._load` was split into `_load` (file → `FieldView`) and
`_install` (any view → screen); nothing else in the UI knows derived fields exist.

| file | responsibility |
|---|---|
| `imsicon/download.py` | field catalogue, credentials, listing parse, resumable transfer. No Qt |
| `imsicon/derived.py` | Magnus dew point, `check_pairable` (**G17**), `DewPointView`, `DifferenceView`. No Qt |
| `imsicon/ncwrite.py` | NetCDF-3 64-bit-offset **writer**, promoted out of `tests/synth.py` |
| `imsicon/ui/downloaddialog.py` | run × field picker, sizes, cached state, sequential fetch |
| `imsicon/ui/derivedialog.py` | dew point / depression / A−B picker, and the build worker |
| `imsicon/geo.py` | loads the bundled overlay; keeps the `fr_land` contour as the fallback |
| `imsicon/mapdata/levant_10m.json` | Natural Earth coastline + borders, clipped and committed (85 kB) |
| `tools/build_mapdata.py` | dev-only: rebuild that file from the Natural Earth sources |

## R3.2 D1 — the downloader

`CLAUDE.md` Phase 8 and `v2.md` 6.2, built. **Download…** (`Ctrl+D`) → credentials →
listing → tick maps → fetch → the first one opens in the viewer.

* 15 products with menu labels (`CAPE - instability`, `Precipitation - total`, …) so the
  choice is a *map*, not a variable name. Catalogue order, not alphabetical.
* **Several fields at once**, downloaded one at a time. The dew point needs two files, and
  a *Select what the dew point needs* button ticks exactly `T_2M` and `RELHUM_2M`.
* **Resumable.** `Range` into `<name>.part`, atomic rename only after a size check against
  the listing. A cancelled or dropped transfer deliberately **keeps** the `.part`.
* Downloads land in their own directory beside the decompressed cache, with their own LRU
  budget; `.part` files are spared from eviction.
* Credentials: keyring → `IMS_USER`/`IMS_PASS` → prompt. **Never `QSettings`.**

## R3.3 D2 — the dew point

Alduchov & Eskridge (1996), the modern refit of Magnus–Tetens, better than 0.1 °C over
−40…+50 °C and 1–100 % RH:

```
gamma = ln(RH/100) + a·T/(b + T)          a = 17.625,  b = 243.04 °C
Td    = b·gamma / (a − gamma)
```

Verified against published values at (20 °C, 50 %) → 9.3, (30 °C, 60 %) → 21.4,
(−10 °C, 80 %) → −12.8, (35 °C, 20 %) → 8.7, (5 °C, 90 %) → 3.5.

Three edge cases decide whether it survives real model output, and all three are enforced
rather than assumed:

| input | why it matters | what happens |
|---|---|---|
| RH > 100 % | float noise at saturation gives **Td > T**, which is impossible | clamped to 100, where the formula collapses **exactly** to Td = T |
| RH ≤ 0 % | `ln(0)` is −inf; a huge negative Td still colours a map convincingly | **NaN**, and the count is reported once |
| T ≤ −243.04 °C | the formula's pole | masked; a pole that is "unreachable" is what turns up in a file one day |

`Td ≤ T` is enforced at the end, so rounding cannot produce a supersaturated pixel.

**Canonical space is Kelvin.** `TD_2M` joins the v2 registry as a Kelvin temperature, so it
gets °C by default and °C/K/°F on the combo — and a *written* `TD_2M` file reopens with
exactly the treatment `T_2M` gets, rather than a "these units look stale" warning.

**Writing it.** `ncwrite.write_canonical` streams the field out as a genuine NetCDF-3
64-bit-offset file — `netCDF4` reads it, and so does this app. Measured: values round-trip
**bit-identically**, member labels survive (**G1**), and `T_2M − TD_2M` built from the
written file equals the live computation. The file carries an `imsicon_provenance`
attribute naming both inputs and the formula. **Save field…** (`Ctrl+S`) writes whatever is
on screen; `--write PATH` does it headlessly.

## R3.4 D3 — the difference map

`DifferenceView` subtracts the operands' **display** values, member by member. That is not
laziness: for a shared affine `y = s·x + o`, `(s·a + o) − (s·b + o) = s·(a − b)` — the
offset cancels on its own, so **G15** is satisfied by construction rather than by a special
case, and the same identity is why a unit change only ever *rescales* the cached range by
`s'/s` instead of forcing a rescan. Both operands are therefore held to one unit selection.

* **Member by member, then aggregate.** `max(a) − max(b) ≠ max(a − b)`, and only the
  latter is the question asked. `agg_frame` aggregates the stack of differences.
* **Diverging colormap, symmetric scale.** `CET-D1A`, levels pinned to ±max|v|. A
  sequential ramp cannot show which side of zero a value is on, and an asymmetric scale
  moves the colour that means "no difference" as the data changes. `spread` is excluded —
  it is non-negative already.
* **Unlike quantities are refused**, with a cheap registry pre-check
  (`derived.units_look_compatible`) so the dialog greys out `RELHUM_2M − T_2M` *before*
  spending 16 s decompressing to find out.

## R3.5 Measured, not assumed

Timings on the real spatial grid (20 members × 261 × 161 = 840,420 values per frame),
against v2's 16.7 ms 60 fps budget:

| operation | before | after | note |
|---|---|---|---|
| dew point kernel | 13.22 ms | **4.73 ms** | in-place numpy; see **G28** |
| dew point, aggregated map | 16.54 ms | **6.87 ms** | |
| dew point, spread map | 16.77 ms ⚠ | **6.12 ms** | was over budget |
| difference (T − Td), aggregated map | 22.61 ms ⚠ | **14.05 ms** | was over budget |
| plain field, aggregated map | 2.76 ms | **2.12 ms** | `Affine.apply` fix helps every view |
| point time series (either) | — | **0.03 ms** | |
| dew point range scan | — | **~1.1 s** for 121 steps | background, no sidecar (see below) |
| writing a full field | — | **~2 s** for 407 MB | streamed, one frame at a time |

Where the kernel time went, before the rewrite: `np.where` for the guards was **8.2 ms** of
13.2, and the logarithm only **0.47 ms**. The formula is not the expensive part; the
allocations are.

**Derived views cache their range in memory only, never in the `.imsstats.json` sidecar.**
The sidecar is keyed on *one* file's size and mtime (`EnsembleFile._cache_key`), which
cannot identify a value depending on two files — a stale entry would be indistinguishable
from a fresh one. ~1 s off the UI thread is a fair price for not inventing a two-file key.

## R3.6 Deliberately not done

* **Cross-run differences.** `check_pairable` refuses two runs by name, because aligning
  them needs *valid* time rather than forecast hour (`v2.md` 6.1.5). The view would not
  change; only the pairing rule would.
* **The 4 MiB prefix sniff in the download dialog** (`v2.md` 6.2.5) — showing a field's
  units and range before committing to 262 MB. `tools/sniff_headers.py` still does it from
  the command line.
* **No live server test.** Every downloader test drives a fake session with real `Range`
  semantics; the IMS credentials were not available here, so the transfer logic is verified
  but the NTLM handshake against the real server is not.
* Probability of exceedance and wind (`v2.md` §5), side-by-side fields (`v2.md` 6.1).

## R3.7 Gotchas found while building v3

* **G27 — a name from the server listing must never become a local path.** The listing is
  attacker-controlled in principle and sloppy in practice. `RemoteFile.local_name()`
  **rebuilds** the name from the two validated capture groups (`run`, `field`) rather than
  sanitising what was sent, so `../../evil.nc.bz2` cannot survive even in the forms
  `Path(name).name` lets through.
* **G28 — `np.clip` on a 0-d input returns an immutable `np.float32`, not an array.** An
  in-place numpy pipeline written for speed then dies with `TypeError` the moment someone
  passes scalars — which the unit tests do, since that is how you check a formula against a
  published value. Allocate the buffers with `np.empty(np.broadcast(...).shape)` and mask
  with `np.copyto(..., where=)` rather than `a[mask] = x`; both are 0-d safe.
* **G29 — G23's "difference in float64" does not apply to a plain elementwise
  difference.** G23 is right about `transform.window_value`, where the subtraction has
  float64 *intermediates* (`A[t]·h[t]`) whose bits the upcast preserves. `a − b` on two
  float32 arrays has none: the IEEE result is already correctly rounded, and the view casts
  back to float32 anyway, so a float64 intermediate cannot survive to be seen. **Measured:
  bit-identical output (max difference 0.0 K) for 3.16 ms against 0.56 ms.** The rule is
  about intermediates, not about the word "difference".
* **G30 — an unpatched modal dialog under `QT_QPA_PLATFORM=offscreen` hangs the test run
  forever.** `MainWindow(None)` schedules a `QFileDialog`, which never returns and never
  fails, so the suite times out with no failing test to point at. Any UI test that
  constructs a window without a path must patch `getOpenFileName`/`getSaveFileName` — and
  must stub `download.stored_credentials`, or a developer with `keyring` installed has
  their real IMS password read (and on some platforms prompted for) by the test suite.
  Same reasoning as **G25**.
* **A `Path.glob` in a skip guard is always truthy.**
  `if not (ROOT / 'data').glob('*.nc'): pytest.skip(...)` never skips: `glob` returns a
  generator. `test_prototype_cli_runs_without_arguments` therefore *failed* on any checkout
  without a `data/` directory instead of skipping. Fixed with `any(...)`.

## R3.8 Status — shipped and verified 2026-08-25

`298 passed, 28 skipped` (169 from R1+v2, unchanged and green, + 129 new). The skips are
the tests that need the 407 MB reference file, which is gitignored.

| item | where | verified by |
|---|---|---|
| **D1** downloader | `download.py`, `ui/downloaddialog.py` | `test_download.py` (21) — IIS/nginx/one-line listings, junk and traversal ignored, resume asks `bytes=40000-` and transfers only the remainder, a short file is refused rather than renamed, a range-ignoring server restarts cleanly, auth failure echoes neither password nor URL |
| **D2** dew point | `derived.py`, `ncwrite.py` | `test_derived.py` (41), `test_ncwrite.py` (21) — published values, `Td ≤ T` over 50 k random inputs, RH>100 clamped, RH≤0 NaN, netCDF4 reads what was written, values round-trip bit-identically |
| **D3** difference | `derived.py`, `ui/main.py` | `test_derived.py`, `test_ui_derived.py` (22) — member-by-member not aggregate-of-aggregates, °C and K give the same number (**G15**), °F scales by 1.8, symmetric colorbar, diverging map on and off |
| the design claim | `ui/main.py` `_install` | `test_ui_derived.py` — map title, colorbar, y axis, readout and status bar all read at one instant on a derived view |

End-to-end under `QT_QPA_PLATFORM=offscreen`, on a 12-step file at the real 261×161
resolution: a plain field renders; `--derive dewpoint --write` renders **and** writes a
40 MB `TD_2M.nc`; that file reopens as an ordinary `TD_2M` in °C; `--difference T_2M TD_2M
--units F` renders the depression against it; and `--difference T_2M RELHUM_2M` exits **2**
with one readable sentence.

## R3.9 Running it

```bash
venv/bin/python -m imsicon                                  # Download... / Derived field...
# headless, and scriptable:
venv/bin/python -m imsicon --derive dewpoint  --write out/TD.nc  data/ICON_ENS_..._T_2M.nc
venv/bin/python -m imsicon --derive depression --screenshot dep.png data/ICON_ENS_..._T_2M.nc
venv/bin/python -m imsicon --difference T_2M T_S --units C       data/ICON_ENS_..._T_2M.nc
```

`--derive` and `--difference` find the other field(s) of the run beside the file you name,
the same way the dialog does. Toolbar row 1 now carries **Open**, **Download…** (`Ctrl+D`),
**Derived field…** (`Ctrl+R`) and **Save field…** (`Ctrl+S`), then **Map shows**: a field
combo (the open field, `TD_2M`, `T-Td`) followed by the aggregation combo.

## R3.10 Follow-up — naming, the field selector, and the map overlay (2026-08-25)

Three requests after the first v3 review.

### The difference is called `T-Td`

`T_2M-TD_2M` is the machine name — the `QSettings` units key, the NetCDF variable name
after sanitising — and it is not what a forecaster reads. Views now carry a
**`display_name`** alongside `field`: `FieldView.display_name` is just the field, and
`dew_point_depression` sets `T-Td`. The map title, the graph's y axis, the readout subtitle
and the status summary all use `display_name`; nothing that has to be parsed back into an
identity does. Keeping the two separate is the point — renaming `field` itself would have
put a hyphenated label into the settings keys and the written variable name.

### `T-Td` is a map you can pick, not only one you can build

**Map shows** now has a *field* combo before the aggregation combo: the open field, then
`TD_2M - dew point` and `T-Td - dew point depression` when the run's `T_2M` and
`RELHUM_2M` are actually on disk. Switching between the temperature and the depression is
something a forecaster does while reading, so it belongs in the toolbar rather than behind
a dialog; the *Derived field…* dialog stays for the general `A − B`.

* `MainWindow.base_ds` holds the file-backed view, so switching back to it is instant
  rather than re-mapping 407 MB.
* Entries are checked against the disk before being offered — a menu entry that always
  fails is worse than one that is not there.
* `derivedialog.build` stamps `derived_kind` and `derived_request` on every view it makes,
  so the combo names what is on screen whether it came from the dialog, the combo, or
  `--derive`. **This was a real bug the first time round:** `--derive depression` painted
  `T-Td` while the combo still read `T_2M`.
* A view the standard entries do not cover — an ad-hoc `--difference T_2M T_S` — is added
  to the combo as its own entry rather than leaving the combo pointing at the wrong field.

### Coastlines and borders, drawn over the field

`CLAUDE.md` Phase 4.3 asked for a bundled Natural Earth overlay and R1 shipped without it,
so outside the inner box the map had **no coastline at all** (**G6**). Now:

* `tools/build_mapdata.py` clips Natural Earth 1:10 m coastline and admin-0 boundary lines
  to 30.5–39.5 E / 24.5–38 N — the full pan range, not just the domain, or the outline
  stops mid-pan and looks like a bug — rounds to 4 decimals (~11 m, against a 2.5 km grid)
  and writes `imsicon/mapdata/levant_10m.json`. **85 kB**, committed, no runtime download.
* `geo.overlay_for` returns every layer at once and falls back to the `fr_land` contour
  when the bundle is missing, so a lost data file costs the map its outlines rather than
  stopping a forecast opening.
* **Drawn over the model map by explicit z-value**, not by insertion order: field 0,
  coastline 10/11, borders 12/13, marker 20/21.
* Every outline is drawn twice — a white halo under a thin ink line. Over turbo, and over
  a diverging ramp, there is no single ink colour legible at both ends of the scale, and an
  outline that vanishes exactly where you are looking is worse than none.

**On the borders themselves.** Natural Earth's own `FEATURECLA` is carried through
unmodified. In this domain the source marks 6 lines `Disputed`, 38 `Indefinite`, 20
`Line of control` and 112 `International boundary`; the settled ones are drawn solid and
everything the source flags as less than settled is drawn dashed. Nothing in this app
decides where a border is — it renders a public-domain reference dataset with the source's
own uncertainty intact, which is the only defensible thing to do for this region.

**Packaging (Phase 9).** `imsicon/mapdata/levant_10m.json` is read by path, so a PyInstaller
build needs `--add-data "imsicon/mapdata;imsicon/mapdata"`. Without it the app still runs;
it just loses its outlines, which is exactly the failure the fallback is written to survive.

### Verified

`298 passed, 28 skipped`. `tests/test_geo.py` (16) checks the shipped file rather than a
fixture — it loads, spans the whole domain, keeps disputed lines separate, survives being
deleted or corrupted — plus the z-order, the halo contrast and the dashed styling on the
real widgets. `tests/test_ui_derived.py` grew the field-combo cases, including the one that
caught the `--derive` labelling bug.

## R3.11 Follow-up — every downloaded map is selectable (2026-08-25)

Two things reported after the R3.10 review: downloading more than three maps left only one
of them reachable, and *Select what the dew point needs* fetched `T_2M` and `RELHUM_2M` but
the viewer then offered the temperature and the dew point without the humidity.

Both were the same gap. **Map shows** listed exactly three things — the open file, `TD_2M`
and `T-Td` — so every other map of the run, however it got onto disk, could only be reached
through the file dialog. The list is now built from what is actually on disk:

* `_field_entries` scans the run and offers **every field it finds**, keyed `file:<FIELD>`,
  followed by the derived entries as before. Five downloaded maps give seven entries;
  the dew point pair gives four (`T_2M`, `RELHUM_2M`, `TD_2M`, `T-Td`).
* Entries keep **catalogue order** (`derivedialog.field_sort_key`) whichever field is open,
  so picking one does not reshuffle the list under the cursor. The open field is always the
  `base` entry, wherever it sits in that order.
* Names come from the download catalogue (`TOT_PREC - Precipitation - total`), which
  answers "what is this map" **without opening the file** — most of the list is still
  compressed, and expanding one to read its `long_name` costs 16 s. A field the catalogue
  has never heard of (a `TD_2M` written by *Save field…*) falls back to `EXTRA_FIELD_LABELS`
  and then to its own header.
* Picking a file-backed entry reuses the view if this window has already mapped it (so
  flipping between two fields of a run is free after the first look at each), decompresses
  it on the `Open…` worker if it is still a `.nc.bz2`, and otherwise loads it. The
  selection is put back if that fails or is cancelled — the combo moves the instant the
  user picks an entry, so a failed open would otherwise name a field the map is not showing.
* **One run only.** Another run shares neither the valid times nor, in principle, the grid,
  and `check_pairable` already refuses to mix them (R3.6).

**G31 — discovery must remember where the user opened from, not just where the open file
is.** `search_roots(base_ds.path)` looks beside the open file — but a `.nc.bz2` is expanded
into the cache, so the moment the user picked a downloaded map, "beside the open file"
became the cache directory and the rest of the run vanished from the list. `MainWindow`
now keeps `_roots`, the last 8 directories it has been pointed at, newest first, and
`_search_roots()` puts them ahead of `ingest.search_roots`. The *Derived field…* dialog
takes the same list (`DerivedDialog(roots=...)`) so the two selectors cannot disagree about
which files exist. Caught by `test_a_compressed_map_is_expanded_when_it_is_chosen`, which
fails on the old one-root scan.

Also: the combo sizes to its contents (the labels are longer now, and the list is rebuilt
after first show, which the default policy does not measure), and a multi-file download
reports `downloaded 4 maps (CAPE_ML, TOT_PREC, T_2M, RELHUM_2M) - choose between them under
"Map shows"` rather than a list of file names that says nothing about where they went.

### Verified

`283 passed, 30 skipped` here (the count differs from R3.10's because `netCDF4` is not
installed in this environment, so its oracle tests skip). Seven new cases in
`tests/test_ui_derived.py`: five maps give seven entries in catalogue order; picking one
moves the map, the status bar and the Rate control to that field; switching back reuses the
already-open view; another run's files are not offered; a compressed map is expanded on
selection *and the rest of the run survives it*; an unreadable map puts the label back; and
the dew point pair yields temperature, humidity, `TD_2M` and `T-Td`.

`tests/test_ui_derived.py` also grew an autouse guard that pins `ingest.search_roots` to the
test's own directory — the fixtures use run `2026082300`, the same number the 407 MB
reference file carries, so on a developer machine holding `data/` the combo assertions would
otherwise pass or fail by accident (same reasoning as **G25** and **G30**).

---

# Release 4 — the wind map (wind barbs)

Requested 2026-08-25: *an option for a wind map — it uses the u and v wind models and
translates them into wind barbs shown on the map; the resolution of the barbs changes
based on the zoom.* Sections 0 (byte math), R1 (the two-panel viewer), R2 (the one
transform layer) and R3 (derived fields) are unchanged and still the contract. This is
v2.md §5.2, built — including the two warnings it left behind (**G16** circular data,
**G17** member correspondence).

## R4.1 The shape of it

The wind is one quantity read two ways, so one view carries both:

```
    U_10M ─┐                       ┌── speed = hypot(u, v) ──► image, colorbar,
           ├──► WindView ──────────┤                           graph, readout, saving
    V_10M ─┘   (derived.py)        └── u, v vectors ─────────► MapView barb layer
                                                               (imsicon/barbs.py)
```

The colours are the **speed**, because the image, the 20 member curves and the six F4
statistics all need a scalar — and the direction cannot be any of them. **G16**: direction
is circular data, so a linear mean, min, max or percentile of degrees is wrong, and wrongly
*plausible* (the linear mean of 350° and 10° is 180°; the answer is 360°). Direction
therefore reaches the screen only as barbs, which are built from the vectors, and as a
single value under the cursor in the status bar. There is no `WDIR_10M` map and no
direction row in the readout: those are exactly the places a circular quantity would be
silently averaged.

| file | responsibility |
|---|---|
| `imsicon/barbs.py` | the glyph: WMO decomposition, staff/feather geometry in pixel space, the zoom→stride rule, `direction_from`. No Qt |
| `imsicon/derived.py` | `WindView` — speed for every panel, `wind_vectors` for the barbs, `barb_label`, `direction_at` |
| `imsicon/ui/mapview.py` | the barb layer: sample the visible grid, build, draw over the field |
| `imsicon/dataset.py` | `sub_frame(t, rows, cols)` — read the cells a barb sample actually wants |
| `imsicon/transform.py` | `WSPD_10M` in the units registry (m s-1 / kt / km h-1) |

Getting to it, all four the same view: **Map shows → `WSPD_10M - wind speed + barbs`**
(offered whenever `U_10M` and `V_10M` of the run are on disk), *Derived field…* →
**Wind map**, `--derive wind`, and **Download… → Select what the wind map needs** (ticks
both components — one of them alone is half a wind, the R3.11 lesson applied before it
could be repeated).

## R4.2 The glyph, and why it is built in pixels

Half feather 5 kt, full feather 10 kt, pennant 50 kt, open circle for calm; the speed is
rounded to the nearest 5 kt first, and a lone half feather is set in from the tip so it
cannot be misread as a full one. **Barbs are always in knots**, whatever the Units combo
is set to, because the glyph is *defined* in knots — a half feather cannot mean "5 of
whatever the toolbar says". The map title says which wind the feathers are counting.

Every glyph is built as **pixel offsets** from its grid point and converted to degrees at
the end, using the data span of one screen pixel (`ViewBox.viewPixelSize`). Two things
follow, and both are the reason:

* a barb is the same size on screen at every zoom — its size is defined in pixels, so its
  footprint in degrees shrinks as you zoom in, which is the opposite of what building it
  in degrees would do;
* the angle is right whatever the aspect. A ground direction `(u east, v north)` lands on
  screen at `(u / (cos_lat · px), v / py)`, which reduces to being parallel to `(u, v)`
  when the aspect lock of **G8** holds and self-corrects when it does not. A north wind
  draws exactly vertical and an east wind exactly horizontal, at one glyph size.

**The convention, pinned by test rather than by memory.** The staff points in the
direction the wind comes FROM (`s = -(u, v)` normalised), and the feathers sit on the
staff's **right** — with the staff drawn upward for a north wind, they extend east. That
is matplotlib's `barbs` default and the northern-hemisphere convention; the southern flip
is deliberately not offered, since this domain is 28–34.5 N. Getting the side backwards
produces a map that looks completely normal and reads as the wrong hemisphere, which is
why `test_the_feathers_sit_on_the_right_of_the_staff_northern_hemisphere_style` exists.

## R4.3 Resolution follows the zoom

This is the requested behaviour, and it is one rule: **the stride is whatever puts barbs
about 34 px apart**, given how many pixels one grid cell spans right now.

* `choose_stride(spacing_px)` snaps to a fixed ladder (1, 2, 3, 4, 5, 6, 8, 10, 12, …).
  Snapping matters: a stride creeping 7, 8, 9, 10 through a zoom would reshuffle every
  barb on the map at each step.
* `sample_indices` is anchored to the **global** lattice (`0, stride, 2·stride, …`), not
  to the window, so a pan slides the same points across the screen instead of picking a
  different set of grid cells every frame.
* The sample is clipped to the visible window plus a cell of margin — at stride 1 a
  261×161 grid is 42,000 glyphs, and a window showing a tenth of the domain must not pay
  for the other nine tenths.

Measured on the real grid (261×161, 20 members, a 1500×880 window):

| longitude on screen | stride | barbs drawn |
|---|---|---|
| 4.0° (whole domain) | 8 | 378 |
| 2.0° | 4 | 414 |
| 1.0° | 2 | 475 |
| 0.5° | 1 | 475 |
| 0.25° | 1 | 195 |
| 0.06° | 1 | 35 |

## R4.4 Measured, not assumed

On this machine, where **a plain field's aggregated map measures 7.2 ms against the
2.12 ms recorded in R3.5** — i.e. it is about 3× slower than the machine v2 and v3 were
measured on, so divide by three to compare with the tables above:

| operation | ms |
|---|---|
| plain field, aggregated map (R3.5 recorded 2.12) | 7.21 |
| wind speed, aggregated map | 12.76 |
| **wind vectors, whole grid, mean** | **10.79** |
| **wind vectors, whole grid, max** | **19.51** |
| wind vectors, sampled 33×21, mean | 0.41 |
| wind vectors, sampled 33×21, max | 0.36 |
| wind vectors, one point (the status bar) | 0.08 |
| barb geometry, 693 glyphs (2,942 strokes) | 0.66 |
| one barb redraw during a zoom or pan | ~2 |
| `refresh_map` with barbs / without | 16.8 / 13.8 |

The two bold rows are why `MapView` is handed a **source** `f(rows, cols)` rather than two
full-grid arrays (**G32**): the barbs throw away all but a few hundred of the 42,000 grid
points, and computing them anyway costs more of a frame than everything else on the map put
together.

## R4.5 Which wind the barbs show — it is not the same for every aggregation

Aggregating u and v independently is only right for the mean, so the barbs follow what the
colours under them are showing:

| Map shows | barbs | why |
|---|---|---|
| `member N` | that member's vector | |
| `mean` | the mean **vector** (**G16**) | its length is *not* the mean speed the colours show: `norm(mean(V)) ≤ mean(norm(V))` by Jensen, and the gap is exactly the ensemble's disagreement about direction — barbs all pointing one way under a strong colour means the members agree |
| `max` / `min` / `median` | the vector of the member the colour came from, per cell | `max(u)` paired with `max(v)` would invent a wind no member forecast |
| `spread` | the mean vector, and the title says so | a max-minus-min has no member and no direction of its own |

The title carries it — `wind 10m [m s-1] - Ensemble mean - … | barbs (kt): ensemble mean
vector` — because a map that shows the mean speed and the mean vector at once has to say
which is which.

**Refusals**, both inherited from R3's pairing rules and both re-tested here: components
whose `history` gives a different member order are refused (**G17** — pairing `u[i]` with
another member's `v[i]` gives a wind that never existed and looks entirely ordinary), and a
component whose units are not `m s-1` is refused rather than converted (a `km h-1` file
would draw every barb at 3.6× the real speed).

**Saving.** `Save field…` / `--write` writes `WSPD_10M` in **m s-1**, its canonical space,
so it reopens as an ordinary speed field with the registry's kt / km h-1 options. It
reopens *without* barbs, honestly: a speed file no longer knows which way the wind was
blowing, and the UI decides whether to draw barbs by asking the view for vectors.

## R4.6 Gotchas found while building v4

* **G32 — a full-grid aggregation to draw a few hundred glyphs is the whole frame budget.**
  `wind_vectors` over 20×261×161 measures 10.8 ms (mean) and 19.5 ms (max) against a
  16.7 ms frame; the same values for the points actually drawn measure 0.4 ms. So the map
  is given a **source** rather than arrays, and `EnsembleFile.sub_frame` reads only the
  sampled cells instead of copying 3.4 MB per component. The test that keeps this honest
  is `test_sampling_a_subgrid_gives_exactly_the_same_vectors_as_the_whole_one`: the
  sampling must be an optimisation, never a different answer.
* **G33 — a programmatic `setRange` in a UI test is silently undone by the domain refit.**
  `MapView` keeps re-fitting the whole domain until `_user_zoomed` is set, and that flag is
  only set by `sigRangeChangedManually` — a real wheel or drag. A test that sets a range
  directly gets it reverted by the next layout pass, so a zoom assertion reads the *domain*
  stride and passes or fails by accident. Tests set `map._user_zoomed = True` first. (Same
  family as **G25** and **G30**: the harness must not silently differ from the app.)
* **G34 — `np.rint` rounds halves to even, which is the wrong rounding for a barb.**
  12.5 kt would draw as 10 kt and 17.5 kt as 20 kt — a half feather appearing and
  disappearing depending on which side of even the value fell. `barb_counts` uses
  `floor(x + 0.5)`.
* **A `QGraphicsPathItem` added to a ViewBox lives in data coordinates**, so a pen width of
  1.2 would be 1.2 *degrees* — five times the width of the domain. The pennant layer relies
  on pyqtgraph's `mkPen` defaulting to a cosmetic pen. It is also added with
  `ignoreBounds=True`, along with the barb polylines: geometry computed *from* the view
  range must never be able to feed back into the range that produced it.

## R4.7 Verified

`364 passed, 30 skipped` (283 from R1+v2+v3, unchanged and green, + 81 new). The skips are
the tests needing the 407 MB reference file or `netCDF4`, neither present here.

| what | where | verified by |
|---|---|---|
| the glyph | `barbs.py` | `test_barbs.py` (40) — the WMO table from 0 to 100 kt, half-up rounding, calm circle, lone half feather set in, staff into the wind for all four cardinals, feathers on the right, constant pixel size across a 160× zoom range, a north wind vertical and an east wind horizontal, NaN winds dropped, a degenerate view returning nothing instead of raising |
| the view | `derived.WindView` | `test_wind.py` (23) — speed is `hypot(u, v)` everywhere, units rescale the cached range instead of rescanning, barbs stay in knots when the colours do not, `max`/`min`/`median` draw a real member's vector at every cell, the mean barb is never longer than the mean speed (Jensen), 350°+10° averages north not south (**G16**), scrambled member order refused (**G17**), `km h-1` components refused, written and reopened in m s-1 |
| the zoom rule | `ui/mapview.py` | `test_ui_wind.py` (18) — on the real widgets: strides fall 8→4→2→1 as the view narrows, the count stays bounded, panning does not reshuffle the lattice, a glyph stays `SHAFT_PX` long in pixels while shrinking in degrees, unticking removes the barbs and leaves the colours, scrubbing time turns the staffs, switching to a plain field takes the barbs down with it |

End-to-end under `QT_QPA_PLATFORM=offscreen`, on a 4-step file at the real 261×161×20
resolution: `--derive wind` renders speed + barbs over the bundled coastline; `--units kt`
moves the map, colorbar, y axis, readout and status bar together while the barbs stay in
knots; `--derive wind --write` produces a `WSPD_10M.nc` that reopens as an ordinary speed
field; and `--derive wind` with only `U_10M` beside it exits **2** with one sentence.

## R4.8 Deliberately not done

* **An on-map key.** The barb scale lives in the *Wind barbs* tooltip and the dialog rather
  than in a corner of the map. A drawn key is a good idea and a separate one.
* **Barbs over another field** (wind over CAPE, say). The plumbing is duck-typed — any view
  exposing `wind_vectors` gets barbs — so this is a pairing question, not a drawing one:
  it needs a second view alongside the one on screen, which is v2.md 6.1's shared-cursor
  workspace.
* **A direction map, and direction statistics in the readout.** Refused on purpose:
  **G16** says min / max / P10 / P90 of a direction are not meaningful, and a panel with
  four `n/a (circular)` rows teaches less than barbs do.
* **Streamlines and a gust overlay** (`VMAX_10M` is already per-interval and needs no
  de-accumulation — **G14** — so it would drop straight in as a second layer).

## R4.9 Running it

```bash
venv/bin/python -m imsicon                                    # Map shows -> WSPD_10M
venv/bin/python -m imsicon --derive wind data/ICON_ENS_..._U_10M.nc
venv/bin/python -m imsicon --derive wind --units kt --screenshot wind.png <U_10M file>
venv/bin/python -m imsicon --derive wind --write out/WSPD.nc  <U_10M file>
```

Wheel-zoom the map and the barbs fill in; zoom out and they thin. **Wind barbs** on
toolbar row 2 turns them off without losing the speed map, and hovering the map reads
`31.250°N  35.000°E   17.8 m s-1   from 289°`.

---

# Release 5 — isolines, and the T-Td sort scale

Requested 2026-08-26: *isolines for temperature (every 1 degree Celsius) and T-Td (every
0.5 degrees Celsius); for T-Td an option to "sort", meaning it will only show the colours
when the difference is under 2 — 2 white, 1 yellow-orange, 0 red.* Sections 0 (byte math),
R1 (the two-panel viewer), R2 (the one transform layer), R3 (derived fields) and R4 (the
wind map) are unchanged and still the contract.

## R5.1 The shape of it

Both halves are the same question asked twice — *what does this number mean, in degrees?*
— so both are answered by the view rather than by the map:

```
                     ┌─ isolines ──► Interval(step, emphasis, anchor)  ──┐
   FieldView /        │              1 degC on a temperature             │
   DewPointView   ────┤              0.5 degC on a difference            ├──► MapView
   DifferenceView     │                                                  │    (draws it)
                     └─ sort_scale ► SortScale(0 red, 1 orange, 2 white) ┘
                                     T-Td only
```

`MapView` and `MainWindow` ask a view for an interval and a band the same duck-typed way
they ask it for `wind_vectors` (R4.1): a view that has one gets the drawing, a view that
does not gets none, and neither panel learns what a dew point depression is.

| file | responsibility |
|---|---|
| `imsicon/isolines.py` | vectorised marching squares, level selection, the emphasis rule, and the per-field interval registry. No Qt |
| `imsicon/derived.py` | `SortScale`, `SORT_STOPS`, and the `isolines` / `sort_scale` properties on the derived views |
| `imsicon/fieldview.py` | `FieldView.isolines` — the registry entry, converted to display units |
| `imsicon/ui/mapview.py` | two isoline layers under the coastline, and `set_title` wrapping (**G36**) |
| `imsicon/ui/main.py` | the **Isolines** and **Sort** checkboxes, the band's colour scale, the title notes |
| `imsicon/geo.py` | `contour_segments` now delegates to `isolines.contour_lines` |

## R5.2 The interval: a spacing *and* an anchor, converted differently

The trap here is that half of **G15** is easy to remember and the other half is not. The
*spacing* is a difference and takes the affine's scale alone — 1 degC is 1 K and 1.8 degF.
The **anchor** is a value and takes the whole affine, and it is what decides *where* the
lines fall:

| units | spacing | anchor | first lines |
|---|---|---|---|
| °C | 1.0 | 0.0 | 15, 16, 17 |
| K | 1.0 | 273.15 | 288.15, 289.15, 290.15 |
| °F | 1.8 | 32.0 | 59.0, 60.8, 62.6 |

Those are the **same three isotherms** in three notations. Scaling the spacing alone —
the obvious reading of G15 — would have anchored the lines on whole Kelvin or whole
Fahrenheit instead, drawing a *different* set of lines every time the Units combo moved,
20.85 °C and 21.85 °C rather than 15 and 16. A view whose values are already a difference
(`T-Td`) takes the scale for both, because no difference is no difference in every unit.

Levels are always `anchor + k·step` for whole k, never "the frame minimum plus a
multiple": the 20 °C isotherm has to be at 20 °C in every frame, or scrubbing time would
slide every line across the map as the data range breathed.

**Which fields.** `T_2M`, `T_S` and `TD_2M` at 1 °C with every 5th line heavier; a
difference of any two of them — the depression above all — at 0.5 °C with every 2nd
(a round 1 °C) heavier. The finer interval is about range, not symmetry: a temperature map
spans 20 °C across the domain while the depression a forecaster reads lives in the 0–5 °C
band, where half a degree is the difference between fog and no fog. Everything else is
uncontoured and the checkbox is **disabled, not hidden** — the rule Rate and Wind barbs
already follow.

## R5.3 The sort scale

`Sort` is the requested word and it describes what it does: the depression is sorted into
the band that decides whether there is fog or cloud at the surface, and everything drier
stops competing for attention. Red at 0, yellow-orange at 1, white at 2, and **above 2 is
not a separate colour — it *is* the top stop**, so on this app's white background the
colours simply run out where the band does. The thresholds are stated in °C and rescale
like any other spacing, so the band is 0–3.6 in °F.

* **It replaces the scale, so the controls that set one are disabled**, not silently
  ignored: while Sort is on, *Colours* and *Scale* are greyed with a tooltip saying why.
* **Fixed levels, deliberately.** A cell's colour means the same depression in every frame
  and at every step, which is the entire premise of reading it as "under 2 degrees"
  instead of "reddest here".
* **`spread` is excluded**, for the reason it is excluded from the symmetric difference
  scale (R3.4): a max-minus-min across the members is a width, not a depression, and
  colouring it against the fog thresholds would read as a forecast nobody made.
* **Offered on the depression and nothing else.** `T_2M − T_S` is a temperature difference
  and *is* contoured, but it is signed, and red-at-zero-white-above-2 would hide which
  side of zero a cell is on.
* The isolines keep running through the uncoloured air, which is most of the point: they
  are what says how far past the threshold a dry area is.

It also fixes something R3 left awkward. A depression is non-negative, so the diverging
symmetric scale a difference map gets (R3.4) spends half its colorbar on values that
cannot occur — the map is one flat shade of red. Sorting is the reading that scale could
not give.

**Recognising the depression.** `DifferenceView` now decides from its operands: `T_2M`
minus `TD_2M` **is** the depression, so it gets the name `T-Td`, the 0.5 °C interval and
the sort band whether it was built by *Derived field…*, by the **Map shows** combo, by
`--derive depression`, or as an ad-hoc `--difference T_2M TD_2M` against a `TD_2M` file
written earlier by *Save field…*. Verified end to end: the written file's depression is
the same map, to 1e-5 °C, as the live computation.

## R5.4 Measured, not assumed

Contours are rebuilt on **every** redraw of the map, so the arithmetic had to fit inside
v2's 16.7 ms frame. On the real grid (261×161), on the machine R4.4 measured as roughly
3× slower than the one v2 and v3 were timed on:

| operation | ms |
|---|---|
| contour a temperature frame, 16 levels at 1 °C (6,478 segments) | **3.45** |
| contour a depression frame, 33 levels at 0.5 °C (13,085 segments) | **4.81** |
| `refresh_map`, T_2M, isolines off → on | 5.40 → **8.76** |
| `refresh_map`, T-Td, isolines off → on | 15.03 → **18.47** |
| `refresh_map`, T-Td sorted, isolines off → on | 15.03 → 18.08 |

So the lines cost about **3.4 ms a frame**, roughly 1.2 ms on the reference machine. (The
T-Td row is dominated by the derived difference itself, which R3.5 recorded at 14.05 ms
and which this release does not touch.)

Two things got it there, and the obvious spelling of either would have blown the budget:

* **The grid is walked once for the whole level set, not once per level.** A binary search
  places each cell's corner range in the level ladder, which gives the *count* of levels
  it crosses, and `repeat` expands that straight into the (cell, level) pairs. Everything
  downstream then works on the 13,000 crossings that exist rather than the 1.4 million
  (cell, level) combinations that do not. Measured: the per-level scan cost 6.56 ms for the
  0.5 °C case against 4.81 ms.
* **A float32 field is contoured in float32.** Every frame this app draws is float32, the
  comparisons and the search are the bulk of the cost, and halving their memory traffic is
  most of the rest of the difference. Crossing *positions* still land in float64 because
  the coordinates are; float32 would place a line to about 0.3 m on a 2.5 km grid anyway.

`geo.contour_segments` — the `fr_land` coastline, contoured once at startup — now
delegates to the same function. One marching squares in the codebase rather than two that
can disagree about a saddle.

## R5.5 Gotchas found while building v5

* **G35 — a title naming an interval the lines are not drawn at is worse than no title.**
  Above ~60 lines a map is a hatch pattern, not a reading, so `levels_for` coarsens the
  step by a **whole** factor (every line at the coarser step was a line at the finer one)
  and reports the step it actually used. `MainWindow` therefore builds the title *after*
  `set_frame`, from what the map drew — `isolines 6 °C (too many lines at 1 °C)` — rather
  than from what it asked for.
* **G36 — a long map title silently CROPS the map.** A pyqtgraph `LabelItem`'s minimum
  width is the width of its text, and a `GraphicsLayout` widens the whole column to honour
  it. Adding the isoline and sort notes to the title took the plot column from 582 px to
  992 px, swallowed the colorbar entirely, and — the ViewBox being aspect-locked — shrank
  the latitude on screen from 6.5° to 2.4°, showing about a third of the domain with no
  error anywhere. That is **G10** reached from the other end, and it was latent in R4's
  barb titles too. `MapView.set_title` now wraps the label to the widget width (and
  re-wraps on resize), so a long title takes a second line instead of taking the domain.
  `test_a_long_title_does_not_crop_the_map` fails on the old code.
* **A `cap=MAX_LINES` default argument cannot be monkeypatched.** Python binds default
  arguments at import, so a test that lowers `isolines.MAX_LINES` to force the coarsening
  path changes nothing. `levels_for` and `contour_set` take `cap=None` and read the module
  attribute at call time. The same trap as any "constant" a test needs to move.
* **The saddle decision is not cosmetic.** Cases 5 and 10 have all four edges crossing and
  two valid pairings; picking the wrong one joins two lobes that are not connected. It is
  resolved from the mean of the four corners — whichever side of the level the middle of
  the cell is on is the side that stays connected through it — and
  `test_a_saddle_is_resolved_by_the_middle_of_the_cell` pins both branches, because the
  wrong one draws a perfectly plausible-looking map.

## R5.6 Verified

`419 passed, 30 skipped` (364 from R1–R4, unchanged and green, + 55 new). The skips are the
tests needing the 407 MB reference file or `netCDF4`, neither present here.

| what | where | verified by |
|---|---|---|
| the geometry | `isolines.py` | `test_isolines.py` (36) — **the vectorised path matches a deliberately naive cell-by-cell loop written independently in the test**, on random fields with ties and NaNs; a ramp's contour lies exactly on its level; a peak gives a closed ring; a saddle gives two segments that do not cross, resolved by the cell centre; a NaN corner leaves a hole rather than a line around one; mismatched coordinates are refused |
| the interval | `isolines.py`, `fieldview.py`, `derived.py` | same file — levels anchored, not floated with the frame; the same isotherms in °C, K and °F; a difference anchored at 0; the cap coarsens by a whole factor to a subset of the fine lines; only the temperatures are contoured |
| the sort band | `derived.py` | same file — red/orange/white at 0/1/2, 0–3.6 in °F, offered on the depression and refused on `TD_2M`, `T_2M`, the wind and `T_2M − T_S`; a depression built from a *saved* `TD_2M` is the same map as the live one |
| the map | `ui/mapview.py`, `ui/main.py` | `test_ui_isolines.py` (19) — on the real widgets: lines drawn every whole °C with every 5th heavier, following the time step and the aggregation; unticking leaves the colours; °F keeps the same isotherms; the band fixes the colorbar at 0–2 and disables *Colours* and *Scale*; `spread` is never sorted; Sort clears itself on a field that has none; and G36's long title no longer crops the map |

End-to-end under `QT_QPA_PLATFORM=offscreen`, on a 6-step run at the real 261×161×20
resolution: `T_2M` renders with 1 °C isolines over the bundled coastline; `--derive
depression` renders with 0.5 °C isolines; `--sort` colours only the band and greys the
two controls it replaces; `--units F` moves the band to 0–3.6 °F and the label to 1.8 °F
while the lines stay put; `--isolines off` takes them down; `--sort` on a plain
temperature prints one sentence and carries on; and `--difference T_2M TD_2M --sort`
against a `TD_2M` written by `--write` opens as `T-Td` with the band.

## R5.7 Deliberately not done

* **Labels along the contours.** The standard way to read a contour map is to have the
  value written into a gap in the line. It needs label placement (which gap, which angle,
  how many per line) and re-placement on every zoom, which is a piece of work in its own
  right; for now the heavier every-5th line, the colorbar and the hover readout carry the
  values.
* **Smoothing before contouring.** The lines show the data, ragged or not. Real ICON
  output is spatially coherent; a filter that tidied the lines would also move them.
* **Isolines on the other fields** (CAPE at 250 J kg⁻¹, precipitation at 1 mm). The
  registry takes one line each — the reason they are absent is that nobody asked, not that
  anything stops them.
* **A sort band on other quantities.** The thresholds 0/1/2 are the dew point depression's
  physics. The mechanism is general; the numbers are not.

## R5.8 Running it

```bash
venv/bin/python -m imsicon data/ICON_ENS_..._T_2M.nc            # isolines every 1 °C
venv/bin/python -m imsicon --derive depression --sort data/ICON_ENS_..._T_2M.nc
venv/bin/python -m imsicon --derive depression --sort --units F --screenshot sorted.png <T_2M file>
venv/bin/python -m imsicon --isolines off data/ICON_ENS_..._T_2M.nc
```

Toolbar row 2 carries **Isolines** (on wherever the field has them) and **Sort** (enabled
on `T-Td`, which **Map shows** offers whenever the run's `T_2M` and `RELHUM_2M` are on
disk). The map title names both: `T-Td [°C] - Ensemble mean - 2026-08-23 02:00Z (+2 h) |
isolines 0.5 °C | sorted: colour only below 2 °C`.
