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
  and writes `imsicon/mapdata/levant_10m.json`. **85 kB** (133 kB once R6 added the land
  polygons), committed, no runtime download.
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
* ~~**Barbs over another field**~~ (wind over CAPE, say). The plumbing is duck-typed — any
  view exposing `wind_vectors` gets barbs — so this is a pairing question, not a drawing
  one: it needs a second view alongside the one on screen. **Built in Release 8**, and it
  was exactly that: `MapView` and `barbs.py` did not change at all.
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


---

## R5.9 Follow-up — the isoline switch, and how close the lines are (2026-09-01)

Requested 2026-09-01: *make it possible to choose if the user wants the isolines or not,
and if it does, add a slider on top that lets the user choose how close they are — 0.5, 1,
2, 3, 4.*

The switch was already there (R5.2's **Isolines** tick, on by default where a field has
lines, disabled where it does not) and is unchanged. What is new is the spacing beside it,
and pairing the two is what makes the tick worth having: 1 °C is the right interval for
reading a gradient across the domain and the wrong one for reading the 0–5 °C air a fog
forecast lives in, so "off or on" was really "off, or on at whatever R5 chose".

```
Isolines [x]  ──|────  2 °C          the notches are 0.5, 1, 2, 3, 4 CANONICAL degrees
                                      the label is the same spacing in the units on screen
```

**The notches are canonical (Celsius) degrees, and that is G15 applied to a user's own
choice.** Picking 2 with the Units combo on °F draws the lines picking 2 on °C drew,
labelled 3.6 °F — not a new set anchored on whole Fahrenheit. Anything else would move
every line on the map as a side effect of relabelling the colorbar, which is exactly what
R5.2 went to some trouble to prevent. The slider therefore sits still through a unit
change while its label follows the colorbar; the map title, which already reported the
interval actually drawn, needed no change at all.

**The anchor is kept and only the emphasis follows the step.** Re-spacing has to re-read
the map, not redraw it somewhere else, so every choice still puts a line on 0 °C. The
heavy line cannot keep "every 5th" across all five, though — at 0.5 °C that lands on
2.5 °C, which is no landmark — so `isolines.emphasis_for` is a small table chosen to make
the heavy line a number a forecaster would name:

| spacing | 0.5 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| every Nth line heavier | 2 | 5 | 5 | 2 | 5 |
| i.e. a heavy line every | 1 °C | 5 °C | 10 °C | 6 °C | 20 °C |

**The choice is remembered per field**, in the window rather than in `QSettings`. 2 °C on
a temperature map and 2 °C on a depression are different readings, so flipping between
them under *Map shows* must neither carry one choice onto the other nor throw the first
one away. Each field opens on its registry interval — 1 °C on `T_2M`/`T_S`/`TD_2M`,
0.5 °C on `T-Td` — until it is moved.

Where it lives is the R2/R5 rule again: the spacing is pushed onto the **view**
(`FieldView.set_isoline_step`, `DerivedView.set_isoline_step`), never held beside the map,
so the map, the title, the tooltip and the CLI all read one number and the affine that
turns 2 °C into 3.6 °F is applied in exactly one place. The slider is disabled while
Isolines is unticked, for the same reason *Colours* and *Scale* are disabled under Sort: a
control that moves and changes nothing is worse than one that is visibly out of reach.

**Fields that are not contoured are still not contoured** — CAPE, precipitation, humidity
and the wind leave both controls greyed, as R5.2 set out and R5.7 recorded. The five
spacings are degrees, so extending them to a `J kg-1` or `mm` map is a different choice of
numbers per field (R5.7 still has it as a one-line registry entry each), not this control.

### Measured

On the real 261×161 grid, 20 members, a 1500×880 window — the machine R4.4 measured as
roughly 3× slower than the one v2 and v3 were timed on, against v2's 16.7 ms frame:

| spacing | levels | segments | `contour_set` | `refresh_map` |
|---|---|---|---|---|
| isolines off | — | — | — | **0.83 ms** |
| 0.5 °C | 32 | 5,999 | 0.91 ms | **1.99 ms** |
| 1 °C | 16 | 3,015 | 0.67 ms | 1.67 ms |
| 2 °C | 8 | 1,432 | 0.50 ms | 1.50 ms |
| 3 °C | 5 | 937 | 0.45 ms | 1.45 ms |
| 4 °C | 4 | 736 | 0.40 ms | 1.36 ms |

So the closest spacing the slider offers costs about **1.2 ms a frame** over drawing no
lines, and the vectorised marching squares R5.4 built is why choosing it is not a decision
about performance. (Measuring this wrongly is easy and was done first: whichever variant
runs first is charged a warm-up of ~1.5 ms, which reported "isolines off" as the slowest
of the six. Each figure above is a steady-state mean of 30 redraws after 5 warm-up ones —
the same trap R6.5 hit from the paint side.)

### G39 — a delegating decorator needs a class-level default for every new instance
attribute

`DerivedView.__getattr__` hands any name it cannot find to its first operand, which is a
`FieldView` — and `FieldView` now has an `_iso_step` of its own. So a chosen spacing lives
in the same-named attribute on both, and a `DerivedView` that had not set one yet would
have silently contoured `T-Td` at the spacing chosen for the `T_2M` underneath it, at the
wrong emphasis, with the slider showing the right number. `_iso_step = None` is a **class**
attribute on `DerivedView` for that reason, not an `__init__` line — the same shape as
`isoline_interval`, `rate_hours` and the rest of the block above it. Any future state
added to `FieldView` needs the same treatment.
`test_a_derived_view_is_spaced_on_its_own_and_not_on_its_operand_s` fails without it.

### Verified

`519 passed` (504 from R1–R6, unchanged and green, + 15 new). Eight in
`tests/test_isolines.py`: the five offered spacings are the registry's own plus four more;
re-spacing keeps the anchor and moves the emphasis onto a round number; a broken step
(0, NaN) leaves the interval alone rather than costing the map its lines; a chosen spacing
still anchors on whole degrees; the same choice in °C and °F is the same isotherms;
clearing it restores the registry's; an uncontoured field refuses one; and G39. Seven in
`tests/test_ui_isolines.py`, on the real widgets: the slider starts on the field's own
interval; moving it spaces the lines and retitles the map; 0.5 °C draws more lines and
heavies every whole degree; unticking puts the slider out of reach and re-ticking gives
back the spacing rather than the default; the label follows the units while the lines stay
put; each field keeps what it was last read at across *Map shows*; and a humidity map
leaves the slider greyed with a tooltip that says why.

End-to-end under `QT_QPA_PLATFORM=offscreen`, on a 6-step run at the real 261×161×20
resolution: `--isoline-step 0.5/1/2/4` renders the same map at four densities with the
title and the toolbar label agreeing at each; `--units F --isoline-step 2` reads
`isolines 3.6 °F` over the 2 °C isotherms; `--derive depression --sort --isoline-step 1`
keeps the sort band and re-spaces the lines running through it; `--isoline-step` on
`RELHUM_2M` prints one sentence and carries on; and an unoffered value is refused by
argparse with the five choices listed.

### Running it

```bash
venv/bin/python -m imsicon --isoline-step 2 data/ICON_ENS_..._T_2M.nc
venv/bin/python -m imsicon --isolines off data/ICON_ENS_..._T_2M.nc
venv/bin/python -m imsicon --derive depression --isoline-step 1 --sort <T_2M file>
```

Toolbar row 2: **Isolines** ticks them on, the slider beside it chooses 0.5, 1, 2, 3 or
4 °C, and the label to its right says that spacing in whatever units are on screen.

---

# Release 6 — a transparent zero, a vivid ramp, and the land underneath

Requested 2026-08-26: *for each map, for the turbo colours, make 0 be transparent and make
everything else pop up more; and when the map is empty, make the land light grey (keep the
sea white).* Sections 0 (byte math), R1 (the two-panel viewer), R2 (the one transform
layer), R3 (derived fields), R4 (the wind map) and R5 (isolines and the sort scale) are
unchanged and still the contract.

## R6.1 The shape of it

The two halves are one idea. Painting nothing where the field is zero is only an
improvement if there is something worth seeing underneath, so the transparency and the
grey land ship together and neither makes sense alone:

```
   geo.load_mapdata()['land'] ──► MapView.land   (grey fill, z = -10)   the sea is the
                                       ▲                                widget's own white
   FieldView / derived view ─────► MapView.img   (z = 0)  ◄── ColorMap from ui/colors.py
                                       ▲                        vivid + alpha fade at zero
   MainWindow.refresh_map ── lo, hi ───┘  _zero_is_the_floor(lo, hi) decides the fade
```

| file | responsibility |
|---|---|
| `imsicon/ui/colors.py` | the two colour transforms (`vivid`, `fade_in_from_zero`), the cache, and the sort band's transparent top stop |
| `imsicon/ui/main.py` | `_zero_is_the_floor` — the one predicate; `_apply_colormap` gained the flag |
| `imsicon/ui/mapview.py` | the land layer at `Z_LAND = -10`, filled odd-even |
| `imsicon/geo.py` | `land` in the overlay dict, as one array pair per closed ring |
| `tools/build_mapdata.py` | schema 2: `ne_10m_land` fetched and Sutherland-Hodgman clipped |

## R6.2 Zero is an absence, and only at the floor

A CAPE map at 03 UTC is zero nearly everywhere, and turbo paints zero as a near-black
navy: the whole domain reads as a dark rectangle with the coastline lost underneath it.
Zero is not a small value of CAPE, it is the *absence* of CAPE, and the honest way to draw
an absence is to draw nothing.

**The fade is at the value zero, not at position 0.0 of the ramp**, and that distinction is
the whole safety of it. `_zero_is_the_floor(lo, hi)` asks whether the bottom of the colour
scale *is* zero — which means the field cannot go lower, so a zero cell is an absence. On a
2 m temperature map the bottom of the scale is the coldest air in the domain, which is a
reading with nothing missing about it, and fading it would hide the coldest place on the
map. So:

| map | floor | faded |
|---|---|---|
| CAPE, precipitation, snow depth, wind speed | 0 | yes |
| any field under `spread` (max − min) | 0 | yes — no spread means the members agree |
| `T_2M` / `T_S` / `TD_2M` | the coldest air | no |
| a difference map (`T-Td`, `A − B`) | −max\|v\| | no — its centre is a reading, not an absence |

The tolerance is relative to the span, because a floor is a float that has been through a
units affine: 0 m of snow read as mm is not exactly 0.0.

**It fades rather than switches.** A model field is not "0 or 1400"; it is a floor of exact
zeros with a smooth skirt of small values around every active cell. Cutting at exactly 0.0
would leave that skirt painted solid and the map barely changed, so alpha climbs from
nothing to opaque across the bottom 5 % of the scale (~160 J kg⁻¹ on the reference CAPE
file — below anything a forecaster acts on). **Nothing above the fade is touched at all**:
the transform can lighten a map, never move a value's colour.

## R6.3 Vivid — and why the near-black end had to go

With the floor gone the rest has to carry the map on its own. `vivid` does two things to
every stop: a saturation gain about the value (V = max(r,g,b) is held, the other channels
are pulled down — an HSV saturation multiply without the round trip), and a lift of V
itself to `floor + (1 − floor)·V`, which is ≥ V everywhere and exactly V at V = 1, so it
only ever brightens and leaves a fully bright colour where it was.

The lift is not decoration. turbo starts at a navy of V = 0.23 and viridis at a purple of
V = 0.27, and against a pale map both read as *black* — which is the reading the
transparency now gives to zero, and must give to nothing else. A low-but-nonzero cell that
looks like an empty cell is the bug this release exists to fix, reappearing 5 % further up
the scale.

Applied to the six sequential ramps. **The three diverging ones are left exactly as they
are**: a difference map's centre is a reading ("no difference"), its neutral colour is
already pale, and symmetry about zero is the only thing that map is for.

## R6.4 The land, and why lines could not do it

`geo` already bundled Natural Earth coastline and border *lines*, and a line has no inside:
a coastline clipped to a box is a set of open polylines, so there is nothing to fill. Schema
2 of `levant_10m.json` therefore carries `ne_10m_land` **rings**, clipped with
Sutherland-Hodgman — which keeps a ring a ring, unlike `clip_line`, which may split a
polyline into several runs. The degenerate edges S-H can leave running along the box
enclose no area, so an odd-even fill is unaffected by them, and interior rings are kept
rather than dropped so a hole punches its own hole with no further bookkeeping.

* **The land is under the field, not over it** (`Z_LAND = -10`), by explicit z-value like
  every other layer since R3.10. It is there to be seen *through* the map.
* **The sea is not drawn at all** — the widget background is already white, so "keep the
  sea white" is the absence of a layer rather than a second one.
* Grey `#e7e7e2`: pale enough to sit under the lowest values a ramp carries, dark enough to
  read as land at a glance.
* An older bundle, a missing one or a corrupt one costs the map its grey land and nothing
  else — the same failure mode `load_mapdata` already had for the outlines.

**The sort band's top stop changed with it.** R5 ended the band at white on the reasoning
that above 2 °C "the colours simply run out", which was true while the background was plain
white. With grey land underneath, an opaque white top stop would paint over the coastline
in exactly the dry air the band is trying to say nothing about. The stop keeps its RGB —
the colorbar still reads white at the top — and loses its alpha, so the colours run out for
real. It also fixes something R3 left awkward from the other side: a depression is
non-negative, so its diverging symmetric scale spent half a colorbar on values that cannot
occur.

## R6.5 Measured, not assumed

Timings on the real CAPE file (261×161, 20 members), each `refresh_map` followed by a full
widget grab so the paint — where a lookup table is actually applied — is inside the number.
This is the machine R4.4 measured as roughly 3× slower than the one v2 and v3 were timed on.

| operation | ms |
|---|---|
| `refresh_map` + paint, opaque vivid turbo | 15.2 |
| `refresh_map` + paint, transparent-zero turbo | **15.4** |
| the land fill, added to the same paint | **+0.6** |
| one cold colormap build (256 stops) | 1.33 |
| a cached `map_colormap` lookup | 0.03 |

So an RGBA lookup table costs about 0.2 ms — inside the noise, and nowhere near v2's
16.7 ms frame. Measuring it wrongly is easy and was tried first: **timing `refresh_map`
alone reported the transparent path as 3.7 ms slower**, because `ImageItem.setImage` only
marks the item dirty and the LUT is applied at paint. The extra 3.7 ms was a first-call
warm-up being charged to whichever variant ran first.

Other numbers:

| claim | measurement |
|---|---|
| the land bundle stays committable | 19 rings, 2,763 points, **133 kB** total (from 85 kB) — clipped down from the 82,076 points those rings carry globally |
| the clip is correct, not just small | ray-cast against the clipped rings at 12 places nobody can be wrong about: Jerusalem, Damascus, Cairo, Sinai and Cyprus land; the Med off Haifa, west of Cyprus, the NW corner, the Gulf of Suez and the Eilat gulf sea |
| the existing layers did not drift | the regenerated coastline and border blobs are **byte-identical** to the committed ones; only the `land` key is new |

## R6.6 Gotchas found while building v6

* **G37 — a `pg.ColorMap` built from floats in 0..1 is black from end to end.**
  `ColorMap.__init__` runs every colour through `mkColor`, which reads a 4-tuple as 0–255
  integers, so `[0.19, 0.07, 0.23, 1.0]` becomes `(0, 0, 0, 1)`. `getLookupTable(mode=FLOAT)`
  hands *back* 0..1, so the natural spelling of "read the ramp, edit it, put it back" is
  silently wrong — no exception, just a black map and a black colorbar. `colors._build`
  converts to bytes, and `test_the_stops_survive_the_trip_through_mkColor` fails on the
  float version.
* **G38 — `QMessageBox.critical` reached from a worker's failure signal wedges the whole
  offscreen test run.** **G30** covered the modal file dialog; this is the same trap
  arriving from `_on_derive_failed` *inside* `app.processEvents()`, so pytest-timeout's
  default signal method cannot unwind it either — the run dies at the shell timeout with no
  failing test to point at, and `--timeout-method=thread` is what prints the stack that
  names it. UI tests now collect `QMessageBox.critical` into a list and assert it is empty,
  which turns a wedged run into a one-line failure quoting the app's own message. What it
  caught first time was a *fixture* bug of the G17 family: a humidity array built from `x`
  and `step` but not `y` broadcast to `(NT, 1, NX)`, so the file was written on a
  1-row grid and `check_pairable` correctly refused to pair it with the temperature.
* **An empty `QPainterPath` does not have zero elements.** `path().elementCount()` is 1 for
  a fresh path (a `moveTo` at the origin), so a "the layer was cleared" assertion written
  against `== 0` fails on a layer that really is clear. `QPainterPath.isEmpty()` is the
  predicate that means what it says.

## R6.7 Verified

`504 passed` (469 from R1–R5, unchanged and green apart from the one sort-band assertion
this release deliberately changes, + 35 new). Nothing skips in this environment because
both `netCDF4` and the 407 MB reference file are present here.

| what | where | verified by |
|---|---|---|
| the colour transforms | `ui/colors.py` | `test_colors.py` (20) — the bottom stop is not painted; everything above the fade is fully opaque and the fade is a fade, not a step; **the RGB of every stop is identical with and without the flag**, and alpha only ever decreases; the ramp gains saturation and loses its near-black end without any stop getting darker; a fully bright colour is left where it was; an unknown name falls back instead of raising; the cache returns the same object; G37 |
| the predicate | `ui/main.py` | same file — CAPE, an all-zero field and `spread` are floors; a temperature map, a scale with zero inside it and a symmetric difference are not; a floor that has been through an affine still counts; a NaN range is not a floor (**G20**'s family) |
| the wiring | `ui/main.py`, `ui/mapview.py` | `test_ui_colors.py` (9) — on real widgets: a zero-floor field is not painted at zero and a temperature map is opaque all the way down; the decision follows the aggregation (`spread` on a temperature map fades, `mean` does not) and the Scale control; a units change keeps it; a difference map's diverging ramp is **bit-identical to the stock pyqtgraph one**; the land is filled, sits under the field, and survives switching fields; a 121-step scrub rebuilds no lookup table |
| the land data | `geo.py`, `mapdata/levant_10m.json` | `test_geo.py` (+6) — the shipped bundle carries rings with no NaN breaks, spanning the whole pan range; ray casting agrees with the coastline at 6 places; an older schema-1 bundle still loads; the fill is pale, pen-less, under the field, and cleared by an empty overlay |
| the sort band | `derived.py`, `ui/colors.py` | `test_ui_isolines.py` — red at 0 and yellow-orange at 1 stay painted, white at 2 keeps its RGB and loses its alpha |

End-to-end under `QT_QPA_PLATFORM=offscreen`, on the real 407 MB CAPE file: at +110 h the
convective plume is vivid over grey land and white sea with the coastline legible through
it; at +20 h, when the domain is almost entirely zero, the map is the grey land, the white
sea and the outlines, with a faint haze exactly where a little CAPE exists. `H_SNOW` in
summer — all zeros — renders as a bare map. A 2 m temperature map and a wind-speed map stay
fully opaque, and the sorted depression shows the land through the dry air the band leaves
uncoloured.

## R6.8 Deliberately not done

* **A transparent centre on the diverging ramps.** "No difference" is a reading, and the
  neutral colour is already pale. If a difference map ever wants to disappear where it is
  zero, that is a different request with a different justification.
* **A land shade that varies with terrain.** `topo_icon_web.nc` has `topography_c`, and
  CLAUDE.md Phase 4.3 still lists an optional hillshade — but it only covers the inner box
  (**G6**), so it would fade out mid-domain. The flat fill covers the whole pan range.
* **Lakes and rivers.** `ne_10m_lakes` would put the Sea of Galilee and the Dead Sea back
  in white. It is another layer in the same bundle and the same clip; nobody asked yet.
* **A configurable fade width or land colour.** Both are single constants in
  `ui/colors.py` and `ui/mapview.py`, deliberately not on the toolbar: the toolbar already
  carries seven controls, and these are calibration, not a reading.

## R6.9 Running it

```bash
venv/bin/python -m imsicon data/ICON_ENS_..._CAPE_ML.nc     # zero is not painted
venv/bin/python -m imsicon --derive depression --sort data/ICON_ENS_..._T_2M.nc
python tools/build_mapdata.py --fetch                       # rebuild the bundle (dev only)
```

Nothing new on the toolbar. **Colours** still chooses the ramp, and the transparency
follows the scale rather than a switch — which is the point: it appears on the maps where
zero means nothing happened, and nowhere else.

---

# Release 7 — the deterministic run, and pressure levels

Requested 2026-08-31: *when downloading IMS ICON maps, rather than ensembles, the up and
down button will move through pressure levels, and instead of single members, it will say
the correct pressure level.* Section 0 (byte math), R1 (the two-panel viewer), R2 (the one
transform layer), R3 (derived fields), R4 (the wind map), R5 (isolines and the sort scale)
and R6 (the transparent zero and the grey land) are unchanged and still the contract.

## R7.0 What the second product is — and which of this is measured

`IMS_ICON_manual.pdf` describes the **deterministic ICON-LAM run**, which is the same
model as the ensemble, published separately:

| | ensemble (R1–R6) | deterministic (this release) |
|---|---|---|
| file | `ICON_ENS_<YYYYMMDDHH>_<FIELD>.nc.bz2` | `IE_<YYYYMMDDHH>_<field>.nc.bz2` |
| runs | 00Z only, one a day | **00Z and 12Z** |
| range | +120 h | +90 h |
| fields | 15, all 2-D | **48**, six of them 3-D |
| second axis | **20 ensemble members** (G1) | **20 pressure levels**, or nothing |
| levels | — | 1000 975 950 925 900 875 850 825 800 750 700 650 600 500 400 350 300 250 200 150 hPa |

**Section 0 is measured; this section was not, when it was written.** The 15 ensemble
fields, their units and their byte layout were read off real files (0.2, 0.4). Everything
about the deterministic product here — the 48 field names, their units, which of them
accumulate, and the server folder — was **transcribed from the manual**, because the
credentials for that folder were not available at the time. **They arrived on 2026-08-31
and all of it has since been checked against the real server: see R7.13, which corrects
the four things the manual gets wrong.** That difference is why the release leans so hard on
guards rather than on assumptions:

* a units string this build does not expect means **warn and offer no conversion** (v2
  1.3), so a wrong registry row costs a conversion and can never corrupt a reading;
* the level axis is read out of the **file**, never out of the catalogue, except in the
  one fallback below — which says so on screen;
* the folder `/ims/IMS_ICON/` is **inferred** from the server's own layout (the ensemble
  is in `/ims/IMS_ICON_ENSEMBLE/`, the ICON manuals and topography in
  `/ims/MANUALS/IMS_ICON/`). It could not be confirmed: the server answers **401 to every
  unauthenticated path, real or not**, so a probe cannot tell a wrong folder from a
  private one. `IMS_ICON_URL` overrides it without a new build, and the download dialog
  says so when a listing comes back empty.

## R7.1 The shape of it

One question runs through every part of this release: **what is the second axis of this
file?** It is answered once, in one small module, and everything else asks that object.

```
   IE_<run>_<field>.nc ──nc3──► field var + its 2nd dim's coordinate
                                            │
                                            ▼
                                levels.axis_for(...)  ──►  LevelAxis
                                            │              kind = member | pressure | single
   ICON_ENS_<run>_<FIELD>.nc ──nc3──────────┘              labels, values, note
                                            │
                    ┌───────────────────────┼─────────────────────────┐
                    ▼                       ▼                         ▼
              EnsembleFile            MainWindow                  PlotView
          n_members, member_labels   Level combo, ▲▼, agg list   curves, mean?
                                      title, readout rows
```

`n_members` and `member_labels` keep their names and are now *the axis'* length and
labels, so every R1–R6 reader of them — the map, the graph, the readout, `ncwrite`, the
derived views — keeps working with no change and no special case.

| file | responsibility |
|---|---|
| `imsicon/products.py` | the two families: name grammar, server folder, catalogue, and which field plays which role. **New, and the only place that knows how these files are named** |
| `imsicon/levels.py` | `LevelAxis`, the detection rules, and `step` — "up" as a physical direction |
| `imsicon/nc3.py` | `data_variables`/`field_name` accept a 3-D or 4-D variable of any name; `level_coordinate` reads the second dim's coordinate |
| `imsicon/dataset.py` | `EnsembleFile.axis`, a 3-D surface field, `level_stats`, per-level ranges |
| `imsicon/transform.py` | `field_key` (one registry for both spellings) + the 48 fields' units and accumulation kinds |
| `imsicon/ui/main.py` | the Level combo, the ▲▼ buttons, Up/Down, the aggregation list, the title, the readout choice |
| `imsicon/ui/readout.py` | a second row set for a column |
| `imsicon/ui/plotview.py` | no mean/envelope across levels; the chosen level drawn heavier |
| `imsicon/ui/downloaddialog.py` | a **Product** combo, and a Levels column |

## R7.2 The axis is decided from the file, not from the file name

`levels.axis_for` tries four things, in descending order of how much they are worth
trusting:

1. **One plane ⇒ `single`.** The deterministic run's surface fields are `(time, lat, lon)`
   with no vertical dimension at all; `EnsembleFile` inserts one so every accessor keeps
   its shape, and the picker has nothing to offer.
2. **The coordinate values say it.** Distinct positive values with pressure units — or
   with no units but unmistakably in a pressure range — *are* the levels. `Pa` is scaled
   to hPa, because a chart is read in hPa and `85000 hPa` is not a pressure. This is the
   only branch that can label a file this build has never seen, and it is the one that
   normally fires.
3. **The catalogue says the field is 3-D and the count matches the manual's ladder.** Only
   when the file's own coordinate is degenerate — the ensemble's `sfc` axis is 20 zeros
   (**G1**), and a deterministic file merged the same way would be too. The levels are
   then the manual's, in the manual's order, and **`axis_note` says so out loud**: the
   status bar reads `⚠ levels assumed` with the full reason in its tooltip. An assumed
   order that happened to be reversed would label every map wrongly while looking entirely
   normal, which is exactly the class of failure this codebase's gotchas are about.
4. **Otherwise it is the ensemble's members**, named from `history` (**G1**), unchanged.

## R7.3 Up is up the atmosphere — which is not the same as `+1`

This is the requested behaviour and the one piece of it that is easy to get wrong.

* The **Level combo lists the top of the atmosphere first** (150 hPa … 1000 hPa), so the
  list reads like a vertical profile and "up" means up the list *and* up the column
  instead of fighting the combo's own keyboard behaviour.
* **`LevelAxis.step` orders by pressure, not by index** (**G40**). Up from 850 hPa is
  825 hPa whichever way round the file stores its coordinate; a file in ascending pressure
  and a file in descending pressure behave identically. Stepping **clamps** at both ends
  rather than wrapping, so holding the key stops at the top of the column instead of
  jumping back to the ground.
* Two toolbar buttons (**▲ ▼**, auto-repeating) and the **Up / Down arrow keys** do the
  same thing — left/right still walk time, so the four arrows are "where" and "when".
* A file opens on **850 hPa**, the conventional low-level chart, and the map title always
  names the level, so it is a starting point rather than a hidden assumption.
* On an **ensemble**, Up/Down step the member and switch the map to *Single member* first.
  Stepping "to the next member" while the map shows the ensemble mean would change nothing
  visible; switching makes the key do what it was asking for, and the title then says
  which member is on screen.

The map title, the graph, the readout and the status bar all name the position the same
way, from `ds.level_label(i)`: `temp [°C] - 850 hPa - 2026-08-31 03:00Z (+3 h)`.

## R7.4 What must not be computed across a column

An ensemble is 20 samples of one quantity, so a mean, a spread and a percentile across it
are the whole point. **A column of pressure levels is not.** The mean of the temperature
at 1000 hPa and at 150 hPa is not a temperature anyone forecasts, and a "P90 across
levels" is meaningless in exactly the way a linear mean of compass directions is
(**G16**) — and just as plausible-looking. So `LevelAxis.aggregatable` is False for a
pressure axis, and:

* the **aggregation combo** carries one entry, *Single level*, and is disabled with a
  tooltip saying why (disabled, not hidden — the rule Rate, Wind barbs and Isolines
  already follow);
* the **graph** still draws all 20 curves, which is a time-height section of one point and
  worth reading, but **not** the ensemble mean or the min–max envelope. The selected
  level's curve is drawn heavier instead, so the up/down keys show as movement;
* the **readout** switches to a row set that reports the column honestly: the level shown,
  the value there, and the highest and lowest in the column **named by the level they
  occur at** (`Highest in column  6.65 °C @ 1000 hPa`). A surface field, having one level,
  drops those last two rather than repeating the value above them;
* the **wind barbs** say `barbs (kt): the wind at this level` rather than naming a member
  that does not exist.

## R7.5 One level's colours, not the whole column's

A temperature column spans about 60 °C between 1000 and 150 hPa, so colouring one level
against the file's range paints every map a single flat shade — measured on the test file:
the 850 hPa map came out uniformly red across the whole domain. **Each position on the
axis therefore carries its own cached range**, collected in the same background pass as
the global one, and *Dataset range* uses it. Scrubbing time at one level is still fixed,
which is what R1's fixed scale is for; switching level rescales, which is the only way a
level's own gradient can be seen. The sidecar gains an optional `levels` list inside its
schema-2 entry, so an existing cache still loads and simply has no per-level detail.

## R7.6 The downloader

**Download…** grows a **Product** combo: *IMS ICON ensemble* or *IMS ICON deterministic
(ICON-LAM)*. Switching re-lists the other folder **on the session already open**, so the
password is typed once. The field list gains a **Levels** column — `20 pressure levels` or
`surface` — which is the difference the user is choosing between, said before 262 MB is
spent rather than after. *Select what the wind map needs* ticks that family's components.

Everything else is the R3 downloader unchanged, including **G27**: the name written to
disk is rebuilt from the validated `run` and `field` captures, never echoed back from the
listing, for `IE_` names exactly as for `ICON_ENS_` ones.

## R7.7 What the two families share, and what they must never share

The field name is the registry key, and the two families spell the same quantity
differently (`TOT_PREC` and `tot_prec`, `T_2M` and `t_2m`). Since they are the same
physical quantity out of the same model, `products.field_key` folds the case and **one**
registry serves both: the deterministic `t_2m` gets °C by default, `temp` on pressure
levels is contoured at the same 1 °C a 2 m temperature is, `clct` gets the **G22** cloud
gate, and `tot_prec` gets the **G14** de-accumulation. What is *not* folded is anything a
name is parsed back out of — the file name, the NetCDF variable, the settings key —
because those identify a file rather than a quantity.

What they must never share is a *forecast*. The two products publish the **same run id**
with different contents and, in principle, different grids, so:

* `ingest.scan_for_fields` keys on `(family, run, field)`, and **Map shows** lists one
  family's maps only;
* `derived.check_pairable` refuses two files whose axes are of different kinds, before it
  compares anything else, with a message that says an ensemble member is not a pressure
  level.

Within the deterministic family the derived fields work as they always did, and two of
them are new in substance rather than in code: `t_2m − td_2m` is recognised as the
**depression** (through `field_key`, so the run's *published* dew point lands on the same
name, the same 0.5 °C isolines and the same R5 sort band as the ensemble's computed one),
and the **wind map can be built from the 3-D `u`/`v`** — barbs at 850 or 300 hPa, from
the R4 machinery unchanged, under the name `WSPD` rather than `WSPD_10M` because saying
"10m" over a 300 hPa map would be a plain misstatement.

## R7.8 Measured, not assumed

On this machine, on a synthetic file at the real spatial resolution (261×161, 20 levels,
6 steps). The last row is the calibration point: a plain ensemble field's aggregated map,
measured in the same session, so these numbers can be read against R3.5's and R5.4's.

| operation | ms |
|---|---|
| read one level's frame | **0.01** |
| read the whole column at one time step | 2.39 |
| point time series (20 levels × 6 steps) | 0.004 |
| `refresh_map`, one level, isolines off | 0.48 |
| `refresh_map`, one level, isolines on | 2.92 |
| **`set_level` — map, graph highlight and readout together** | **2.96** |
| ensemble `refresh_map`, mean of 20 members (calibration) | 3.59 |
| ensemble `refresh_map`, single member (calibration) | 0.38 |

So stepping a level costs about what redrawing one frame costs, well inside v2's 16.7 ms
budget, and a level map is *cheaper* than an ensemble mean because it reads one plane
instead of aggregating twenty.

| claim | measurement |
|---|---|
| the per-level ranges are nearly free | one scan collecting both: **12 ms against 9 ms** for the global range alone, over 6 steps — a third more on a pass that is 0.6 s for a real 407 MB file |
| a level's own range has contrast the column's does not | on the test file: global −50.7…13.1 °C, **850 hPa −12.2…4.8 °C**, 500 hPa −31.5…−14.5 °C |
| a written pressure file reopens as itself | values equal to 1e-3, labels identical, `plev` in hPa, and **no fabricated member history** |
| the sampling of a level is the file's own bytes | every level's frame equals what was written, and `series` equals the column at that point |
| the two families do not collide on one run | `('ens', run, 'T_2M')` and `('icon', run, 't_2m')` in one directory scan, each listed only under its own product |

## R7.9 Gotchas found while building v7

* **G39 — the second axis cannot be identified from its dimension's NAME.** The ensemble's
  is called `sfc` and tagged `axis="Z"`, `long_name="surface"` — and holds 20 members
  (**G1**). A pressure file's may be called anything. So the decision is made from the
  coordinate *values* and their units, with the dimension name only allowed to break a tie
  when the units are missing, and the field catalogue only as a last resort that announces
  itself. Deciding from the name would have read the ensemble as 20 levels of 0 hPa.
* **G40 — "up" is not `+1`.** A level coordinate may ascend or descend in the file, and
  both are ordinary. Stepping by index therefore walks *down* the atmosphere in half the
  files it meets, silently, and a map labelled `700 hPa` would still be correct — only the
  key would be wrong. `LevelAxis.step` orders by pressure, and the combo is listed in the
  same order so the control and the key cannot disagree.
* **G41 — a column's dataset range flattens every map in it.** 60 °C of range across the
  troposphere against ~17 °C within one level: the first render of an 850 hPa map was one
  uniform red rectangle, with the colorbar spanning −50…+13 °C. Fixed with a per-level
  cached range (R7.5). Worth noticing that this is the *same* failure R1's fixed scale
  exists to prevent, arriving from the other direction — a scale can be too stable.
* **G42 — a transient status message must put the standing warning BACK.** The status bar
  is used both for a file's standing warning (`⚠ incomplete file`, and now `⚠ levels
  assumed`) and for progress (`scanning for dataset range...`). `_on_scan_done` cleared it
  to `''`, so the truncation warning **G26** exists to raise had been vanishing on every
  file whose range was not already cached — since R1, unnoticed, because the tests that
  cover it happen to hit the cached path. There is now one `_show_standing_note()` and the
  transient message restores it.
* **G43 — `<FIELD>_eps` is a claim, not a suffix.** `eps` means ensemble. `ncwrite` used
  to append it to every variable it wrote, which on a saved pressure-level or surface
  field would tell the next reader — and `nc3.field_name`, and a human with `ncdump` —
  that the file holds an ensemble it does not hold. It is now written only for a file
  whose axis really is one, and the same rule governs the `history` line: a level file
  gets no fabricated `ICON_ENS_<run>_<member>_` tokens for `member_labels` to read back.

## R7.10 Status — shipped and verified 2026-08-31

`530 passed, 28 skipped` — 476 from R1–R6, unchanged and green, + 54 new (one assertion in
`test_rate.py` was re-scoped, not weakened: it pinned the accumulation table exactly, and
now pins the ensemble's 15 fields within it while `test_levels.py` pins the deterministic
run's). The skips are the tests that need the 407 MB reference file, which is gitignored.

| item | where | verified by |
|---|---|---|
| the axis, read from the file | `levels.py`, `nc3.py`, `dataset.py` | `test_levels.py` (30) — hPa and Pa coordinates; the ensemble's `sfc` still members; a 3-D surface field; the manual fallback *and its note*; a non-pressure coordinate refused; values read back per level |
| **up and down** | `levels.LevelAxis.step`, `ui/main.py` | `test_levels.py`, `test_ui_levels.py` (24) — up is lower pressure in a file stored either way round; clamping at both ends; the ▲▼ buttons and the keys agree; the map really changes to that level's data |
| **the level, not the member** | `ui/main.py`, `ui/readout.py`, `ui/plotview.py` | `test_ui_levels.py` — the picker lists `850 hPa` top-first, the title says it, the status bar counts levels, the readout switches row sets, the graph drops the mean and highlights the level |
| nothing aggregated across a column | `levels.py`, `ui/main.py` | `test_ui_levels.py` — one aggregation entry, disabled, with the reason in its tooltip |
| the ensemble is untouched | `ui/main.py` | `test_ui_levels.py` — six aggregations, member names, the mean curve, the six F4 rows; and switching between the two products in one window restores every control |
| the downloader | `products.py`, `download.py`, `ui/downloaddialog.py` | `test_levels.py`, `test_ui_levels.py` — the `IE_` listing parsed, `local_name` rebuilt (**G27**), the folder overridable, the Product combo, the Levels column |
| pairing and derived fields | `derived.py` | `test_levels.py` — an ensemble and a column refused as a pair; two different ladders refused; the wind map on levels; the native `t_2m − td_2m` recognised as T-Td |
| writing one back | `ncwrite.py` | `test_levels.py` — levels survive, a surface field stays `(time, lat, lon)`, no member history invented |

End to end under `QT_QPA_PLATFORM=offscreen`, on a 261×161×20 deterministic run: `temp`
opens at 850 hPa with 1 °C isolines over the bundled coastline and a colorbar that fits
the level; `--level 500` and the ▲▼ buttons move it; `--derive wind` on `u` draws barbs at
300 hPa in knots; `--difference t_2m td_2m --sort` opens as `T-Td` with the R5 band;
`--level 500 --write` produces a file that reopens as the same 20 levels; and the
synthetic ensemble still renders exactly as it did in R6.

## R7.13 Measured against the live server — 2026-08-31

R7.0 said the deterministic product was transcribed from the manual and not measured,
because there were no credentials. There are now, and everything in R7 has been checked
against the real server and real files. **Four of the manual's facts are wrong, and the
code was right about all four** — not by luck: each is a case the design refused to take
on trust.

| what R7 assumed | what the server says | how it landed |
|---|---|---|
| the folder is `/ims/IMS_ICON/`, inferred from the layout | **confirmed** — HTTP 200 with NTLM; 2,900 files across 58 runs, 00Z and 12Z | the inference held, and `IMS_ICON_URL` stayed unnecessary |
| 48 fields, from the manual's Table 1 | **50** — `sob_s` and `sou_s` are published and undocumented | they parsed and listed already (an unknown field goes to the end of the catalogue); both are now named, and in the registry as `W m-2` |
| 20 pressure levels, 1000 hPa first | **22 levels, 150 hPa first, stored in Pa** | the axis is read from the file's own `plev` (**G39**), so the labels were right anyway; `PRESSURE_LEVELS` now carries the measured ladder |
| a level index counts down the atmosphere | it counts **up**: 15000 Pa is index 0 | **G40** — "up" is defined by pressure, so ▲ gives 825 hPa from 850 hPa in this file exactly as it would in a file stored the other way |

Had the fallback ladder been used, every map would have been labelled upside down. It was
never reached: the real files carry a proper coordinate, which is the branch that fires.

**Measured on run `2026083012`:**

| | |
|---|---|
| 3-D fields | `(time, plev, lat, lon)`, `plev` in **Pa**, `standard_name=air_pressure`, `positive=down`, 22 values ascending 15000..100000 |
| grid | **281 x 201**, lat 29.000..36.000, lon 32.000..37.000, 0.025° — a **different domain from the ensemble's** 261x161 (28..34.5 N, 33..37 E) |
| steps | 91, hourly, +0..+90 h (the manual's range, confirmed) |
| record stride | 4,970,336 B for a 3-D field (**G2** again: `time` is a record variable too) |
| surface fields | two shapes: `(time, lat, lon)`, and `(time, height, lat, lon)` with height = **2 m** for `t_2m`/`td_2m`/`rh_2m` and **10 m** for `u_10m`/`gust10` |
| units | all 12 fields sniffed match the registry, `W/m**2` (**G21**) and `%` included. No row was stale -- and the same product uses BOTH spellings: `asodird_s` says `W/m**2` while `sob_s` says `W m-2` |
| accumulation | measured, not inferred: `tot_prec`, `rain_gsp`, `snow_gsp` and `graupel_gsp` are non-decreasing over 91 steps (**`sum`**); `asodird_s` and `asob_t` are not (a running mean, **`mean`**); `sob_s` is not either, which is why it is not in the table (**G14**) |
| sizes | 3-D 259..424 MB compressed; surface 12..19 MB; `h_snow` **2,123 B** — all-zero summer snow, as in the ensemble |

Both shapes of surface field already opened (`levels.axis_for` answers "one plane" before
it looks at anything else), and the height coordinate is now used for the label: the
picker reads **`2 m`** or **`10 m`** rather than "surface", because calling a 10 m gust a
surface field is a small lie the file itself can correct.

**The downloader, live, for the first time** (R3.6 and R7.11 both listed this as not
done):

* both listings parse — 435 ensemble files across 29 runs, 2,900 deterministic across 58
  — with sizes, the run picker, the Levels column (`22 pressure levels` / `surface`) and
  the per-family wind shortcut all filled from the real HTML;
* a real **resumable** transfer: seeded with 3,000,000 bytes of a `.part`, the app sent
  `Range: bytes=3000000-`, the server answered **206**, and the finished file was
  **byte-identical (SHA-256) to the whole one**;
* a completed file is not refetched, and a size mismatch is what would have refused it.

`tools/sniff_headers.py --product icon` is what produced most of the table above: it
takes a 4 MiB prefix (16 MiB with `--deep`) of each field instead of 259..424 MB, and it
now reports the level axis as well as the units, the range and the monotonicity. Six
fields, deep, settled the whole accumulation question in one command.

**G44 — a truncated `.nc.bz2` is readable data, not a broken file.** Fetching a 24 MiB
prefix of a 271 MB field is the cheap way to look at a 3-D product, and it is also what an
interrupted download leaves behind. `ingest.decompress` raised `EOFError` from
`bz2.read()` at the end of the stream and threw away everything it had already written,
reporting "could not decompress" about several hundred MB of good forecast. bz2
decompresses incrementally, so the bytes already written are real: the loop now stops at
the truncation and lets **G26** clamp the header to the records that survived, which is
exactly the path the app already had for a short `.nc`. An empty or corrupt stream is
still an error. Measured: a 24 MiB prefix of `temp` opens as 7 of 91 steps, all 22 levels,
with `⚠ incomplete file` in the status bar.

## R7.11 Deliberately not done

* **A dew point on pressure levels.** `temp` + `rh` would give one, and the formula does
  not care what the second axis is — but the roles that feed *Derived field…* name the
  2 m pair, and an upper-air dew point is read as a depression against `temp` rather than
  on its own. It is a table entry away when someone asks for it.
* **A cross-section or a tephigram.** The graph is already a time-height section at a
  point; a *pressure*-height section (value against level, at one time) is the natural
  next panel and a different piece of work.
* **Renaming the app.** The window still says "IMS ICON Ensemble Viewer" and the settings
  still live under `IconEnsembleViewer`. It opens both products now, so the name is half
  right — but what an app is called, and the settings key that goes with it, is the user's
  call, not a side effect of adding a product.
* ~~**A live test against the deterministic folder.**~~ **Done 2026-08-31 — see R7.13.**
  The folder, the catalogue, the units, the level ladder and a real resumable transfer are
  all measured now, and `tools/sniff_headers.py --product icon` is what settles a field
  from a 4 MiB prefix instead of a 262 MB download.
* **12Z vs 00Z run comparison.** The deterministic run publishes twice a day — measured:
  58 runs on the server, 00Z and 12Z — which makes run-to-run consistency newly cheap to
  look at, and it needs alignment on *valid* time, which `check_pairable` still refuses
  (R3.6).

## R7.12 Running it

```bash
venv/bin/python -m imsicon                                   # Download... -> Product: deterministic
venv/bin/python -m imsicon data/IE_2026083100_temp.nc        # opens at 850 hPa
venv/bin/python -m imsicon --level 500 data/IE_2026083100_temp.nc
venv/bin/python -m imsicon --derive wind --level 300 --units kt data/IE_2026083100_u.nc
venv/bin/python -m imsicon --difference t_2m td_2m --sort data/IE_2026083100_t_2m.nc
IMS_ICON_URL=https://.../ims/SOME_OTHER_FOLDER/ venv/bin/python -m imsicon   # if the folder moved
```

Toolbar row 1 now reads **Map shows: [field] [aggregation] Level: [850 hPa] ▲ ▼**. On a
pressure file the aggregation is fixed at *Single level*; on an ensemble it is the six
R1 choices and the picker names members instead. **Up** and **Down** step the axis from
anywhere in the window: up the atmosphere on a column, and to the next member (switching
the map to that member) on an ensemble.

---

# Release 8 — wind barbs over any map

Requested 2026-08-31: *make it possible to add wind barbs to any map, in addition to the
current map.* Sections 0 and R1–R7 are unchanged and still the contract. This is R4.8's
one deliberate omission, built:

> **Barbs over another field** (wind over CAPE, say). The plumbing is duck-typed — any
> view exposing `wind_vectors` gets barbs — so this is a pairing question, not a drawing
> one: it needs a second view alongside the one on screen.

That is exactly what it turned out to be. Nothing in `MapView` or `barbs.py` changed.

## R8.1 The shape of it

R4 drew barbs from *the view on screen*, so only a `WindView` could have them. Now the
window can hold a **second** view beside the one being drawn:

```
   MainWindow.ds ────────────────────────────► colours, graph, readout   (any field)
                                                        │
   MainWindow.wind_overlay  (a WindView) ───────────────┼──► MapView.set_wind(...)
        built from the run's u/v, on demand             │
                                                        ▼
                            wind_source() -> (view, is_overlay)
```

`wind_source()` is the whole of the new logic: the map's own vectors if it has them (so
the wind map is untouched), else the overlay if one is open *and still fits*, else none.
`_push_wind` asks it and hands `MapView` the same `f(rows, cols)` source R4 defined, so
the zoom-dependent stride, the pixel-space glyphs and the knots are all unchanged.

| where | what changed |
|---|---|
| `ui/main.py` | `wind_overlay`, `wind_source`, `_overlay_request`, `_overlay_shape`, and the Wind barbs checkbox becoming live over any map |
| `derived.py` | `wind_pair_for(..., levels=)` picks the pair by SHAPE; `barb_label(mode, over=)` names the source; `WindView.source_label` |
| `ui/mapview.py`, `barbs.py` | **nothing** |

## R8.2 Which wind, over which map

The pair is chosen by the shape of the map it is going over, from the catalogue, before
any file is opened:

* over a **surface or ensemble** map — the 10 m pair (`U_10M`/`V_10M`, `u_10m`/`v_10m`);
* over a **pressure level** — the run's 3-D `u`/`v`, so barbs over an 850 hPa temperature
  are the wind *at 850 hPa*. Anything else would draw a wind from the wrong place and look
  entirely normal.

The barbs then follow the map: the same time step, the same aggregation (an ensemble mean
map gets the mean *vector*, **G16**), and the same position on the second axis — so
stepping up the column with ▲ moves the barbs with it.

**The title says the feathers are a different quantity**, because they now can be:

```
CAPE_ML [J kg-1] - Ensemble mean - 2026-08-23 11:00Z (+11 h)  |  barbs (kt) from U_10M/V_10M: ensemble mean vector
t_2m [°C] - 2 m - 2026-08-31 00:00Z (+12 h)  |  barbs (kt) from u_10m/v_10m: the wind at 10 m  |  isolines 1 °C
```

## R8.3 Opt-in, and honest about what it costs

**Wind barbs** is now enabled whenever barbs are *possible* — on the wind map as before,
and over any other map of a run whose wind components are on disk — with the tooltip
naming the two files it would use, or saying to download them.

It is **unticked** on a plain map and ticked on a wind map. That asymmetry is deliberate:
a wind map is barbs by definition, while an overlay opens two more files (up to 262 MB
each, decompressed on the `Open…` worker with the progress the app already has), and a
field map that silently opened two more files on every launch would be a surprise nobody
asked for. Once built, the overlay is **kept** while it fits, so flipping between the maps
of one run is free after the first tick.

The tick always reflects what is actually drawn — there is never a ticked box with no
barbs under it, and never barbs the box does not admit to.

## R8.4 The overlay has to fit the map, not just itself (G45)

**This is the one that mattered, and only real data found it.** The two components are
checked against *each other* inside `derived.wind` (**G17** — member order, grid, run,
time axis). That says nothing about whether they fit the map they are being drawn over.

On the real 2026-08-30 12Z run, fetched as prefixes, `temp` held 7 forecast steps and
`u`/`v` held 5 — an interrupted download stops each file at its own point (**G26**), and
the two are separate files. The overlay built happily, drew correct-looking barbs at
+3 h, and **wedged the app** at +6 h: `wind_vectors(6, …)` indexed past the end of a
5-step wind, inside a paint handler, where the exception never reached a `try`. No
traceback, no error dialog, just a window that stopped redrawing.

So `check_pairable(self.ds, overlay)` now runs against **the view on screen** before the
overlay is accepted, and `_overlay_shape` — the key that decides whether an open overlay
can be reused on the next map — carries `n_times` along with the run, the axis, the level
count and the grid. A wind that does not fit is refused with the reason in the status bar
(`⚠ no barbs over this map`, the full sentence in its tooltip) and the tick put back:
a status note rather than a modal, because the barbs are a secondary thing asked for on
top of a map that is perfectly fine to read.

## R8.4b Two more things real data found

Both were latent before this release and both were reached by the same session:

* **A one-plane field's label is a name, not an identity.** R7 started labelling a
  single-level axis from its `height` coordinate (`2 m`, `10 m`) instead of "surface",
  which is better -- and `check_pairable` compared those labels the way it compares member
  numbers, so a 10 m wind was refused over a surface precipitation map, and `t_2m - t_g`
  would have been refused too. **G17**'s danger is pairing position *i* of one file with a
  different position *i* of another, which needs there to be more than one position: the
  label check now applies only when the axis has more than one.
* **G46 — a range that arrives after the view moved on.** A range scan is per transform
  signature (**G19**), so changing the Rate while one is running leaves the view with no
  range for the new signature; the finished scan then handed `_apply_range` a `None`,
  which it unpacked. It took `--rate 3h --barbs on` together to see it: opening the wind
  held the event loop long enough for the two to cross. `_apply_range(None)` is now a
  no-op, because `_ensure_range` has already started the scan for the signature that is
  actually on screen.

## R8.5 Verified

`546 passed, 28 skipped`. `tests/test_ui_barbs.py` (13) is new and drives the real
widgets:

| what | how |
|---|---|
| the run's wind draws over a CAPE map | the box is enabled and **unticked**; ticking it opens the pair and draws; the map is still CAPE |
| the title names the source | `barbs (kt) from U_10M/V_10M: ensemble mean vector` |
| it follows the map | switching to a single member changes the label and the glyphs; scrubbing time turns the staffs |
| the right pair, by shape | over a pressure level it takes `u`/`v` and moves with ▲; over a 2 m temperature it takes `u_10m`/`v_10m` |
| kept, and dropped | reused unchanged across two fields of one run; dropped when the shape changes, and rebuilt for the new one |
| **G45** | a 3-step wind over a 6-step map is refused, the status bar says so, and scrubbing to a step the wind never had is safe |
| the wind map is untouched | its own vectors still win over an open overlay |
| no wind on disk | disabled, unticked, tooltip says what to download |

On real IMS data (run 2026083012): the 2 m temperature at +12 h with the 10 m wind over
it, and the 850 hPa zonal wind with the wind at that level over it — both rendered
headlessly with `--barbs on`.

## R8.6 Deliberately not done

* **Two coloured fields at once.** The barbs are a second *quantity*, not a second colour
  scale; a proper two-field workspace with a shared cursor is v2.md 6.1 and still open.
* **Barbs from another run.** `check_pairable` refuses two runs (R3.6), and comparing a
  06Z wind against a 00Z map needs alignment on valid time, which this build does not do.
* **Gusts as a second layer.** `VMAX_10M`/`gust10` is per-interval already (**G14**) and
  would drop straight in as a second glyph, but two overlapping glyph sets need a legend
  and a colour rule of their own.
* **Remembering the tick between sessions.** `QSettings` could carry it, but "open two
  more files" is not a preference to restore silently on a launch the user did not ask
  for it on.

## R8.7 Running it

```bash
venv/bin/python -m imsicon data/ICON_ENS_..._CAPE_ML.nc     # tick "Wind barbs"
venv/bin/python -m imsicon --barbs on --screenshot cape.png data/ICON_ENS_..._CAPE_ML.nc
venv/bin/python -m imsicon --barbs on --level 850 data/IE_..._temp.nc
```

The checkbox is on toolbar row 2 where it always was; what changed is that it is no
longer greyed out unless the map is the wind map.

---

# Release 9 — terrain, spacing on both products, heights, and the humidity profile

Requested 2026-10-01: *for both: add an optional topographic map, isolines difference
choosing. For single: pressure to geopot height (both optional isolines and level), make
the date and value more visible, add geopot height next to them (from model). For RH in
single only: instead of levels height is Y and value is X for the single point.* "Both"
is the ensemble (R1–R6) and the deterministic run (R7, "single"). Sections 0 and R1–R8 are
unchanged and still the contract.

**Housekeeping first.** This checkout was behind `origin/main` by R7 and R8 while the R5.9
isoline-spacing slider sat here uncommitted, so the two were merged by hand (five hunks in
`__main__.py`, two in `ui/main.py`); the R5.9 work is the "isolines difference choosing"
half of the request, now applied to both products, and the merged tree passed its 589
tests before anything new was added.

## R9.1 The shape of it

Four things, and the fourth is the only one that needed a new panel:

```
  mapdata/levant_etopo1.npz ──► terrain.py ──► MapView.terrain  (Multiply, z = 1)   [both]
                                   hillshade      "Topography" tick, remembered

  isolines.Ladder  ─► Interval.ladder ─► the slider's notches  (degrees | kft)      [both]
                                        GEOPOT: 0.2 kft, every 5th heavier          [single]

  the run's geopot ─► MainWindow.height_companion ─► ReadoutPanel 'Height (geopot)' [single]
                            (opened beside any column, G45-checked)   ─► ProfileView y axis

  ProfileView  (x = value, y = geopotential height)  ◄── "Profile" tick, on for `rh`  [single]
```

| file | responsibility |
|---|---|
| `imsicon/terrain.py` | load the elevation bundle, the hillshade, the RGBA image, `height_at`. No Qt |
| `imsicon/mapdata/levant_etopo1.npz` | ETOPO1 1-arc-minute elevation, 901×601, clipped to the pan range (592 kB, committed) |
| `tools/build_terrain.py` | dev-only: fetch the subset from NOAA ERDDAP and write the bundle |
| `imsicon/isolines.py` | `Ladder` (the slider's notches per quantity), `DEGREES`, `HEIGHT`, the `GEOPOT` interval |
| `imsicon/ui/profileview.py` | the vertical profile panel |
| `imsicon/ui/readout.py` | the `height` row; the time and value rows a size up |
| `imsicon/ui/mapview.py` | the relief layer, the land mask rasteriser, an unpinned colorbar axis |
| `imsicon/ui/main.py` | **Topography** and **Profile** ticks, the per-ladder slider, the height companion, the graph stack, `app_settings` |
| `imsicon/products.py` | the `height` role (`geopot`) on the deterministic family |
| `tools/build_mapdata.py` | the clip box widened to the union of both products' pan ranges |

## R9.2 Topography: shaded relief, multiplied over any map

The land was already pale grey and the sea white (R6); this is the next thing a
forecaster wants to see through the field: *where the hills are*. A CAPE plume over the
Judean hills and one over the coastal plain are two different forecasts.

* **The data is ETOPO1**, NOAA's 1-arc-minute global relief, public domain, fetched once
  as a NetCDF-3 subset that this app's own `nc3` parses, and committed. Not the model's
  `topo_icon_web.nc`: that covers the inner box only (**G6**), and a relief that faded out
  mid-domain is exactly what R6.8 declined to ship. The sea floor is clamped at −450 m
  (below the lowest land on Earth) because it is never drawn and compresses worst: 751 kB
  without the clamp, 592 kB with it, and not one land cell touched.
* **Shadow only, by construction.** The layer is composited with the painter's *Multiply*
  mode, which can darken and never brighten, and the hillshade is normalised to flat
  ground: flat terrain and slopes facing the north-west light come out white (the field
  under them is untouched), slopes facing away darken it to no less than `AMBIENT` (0.55).
  That is the classic way to put relief under a thematic map without washing its colours
  out; a hypsometric tint would fight the colour scale for the same pixels, and the colour
  scale is the reading.
* **The sea stays white** because the relief is masked to the *same land polygons the grey
  fill uses*, rasterised with the same odd-even rule and the same painter, so the two end on
  exactly the same coastline — and the Dead Sea shore, 430 m below sea level, is land.
  Without rings (an older bundle) it falls back to "above sea level", losing that shore and
  nothing else.
* **Exaggeration 6.** A 100 m rise over a 1.85 km cell is a 3° slope, which shades by
  about 4 % at a 45° sun — invisible. The z-factor is the usual trick and is stated as one:
  the shading says *where* the slopes are, never how steep.
* Off by default and **remembered** (`display/topography`): a backdrop that appears unasked
  competes with the field, and a reader who wants the hills wants them on every map. The
  title says `terrain shading` while it is on, because the shadows change what a colour
  looks like and a screenshot has to say they are not the field.
* Built **once per window**, lazily: a reader who never ticks it pays nothing, and the
  image depends on the bundle and the land polygons, neither of which changes with the
  field, the step or the zoom.

The clip box of the coastline bundle was widened at the same time, from 30.5–39.5 E /
24.5–38 N to **29.5–39.5 E / 24.5–39.5 N**: the deterministic domain is bigger than the
ensemble's (R7.13) and its pan range reached north of the old box, where the coast simply
stopped. Regenerated from Natural Earth: 141 kB, 24 land rings, the same six disputed and
38 indefinite lines as before.

## R9.3 Isoline spacing on both products, with a ladder per quantity

R5.9's slider offered five spacings in degrees. On the deterministic run that already
covered `temp`, `t_2m`, `td_2m`, `t_g`, `tmax_2m` and `tmin_2m` through `field_key`; what it
could not do was contour a **height chart**, where "every 2" is a hatch pattern and the
conventional spacing is 4 dam. So an `Interval` now carries a `Ladder`:

| ladder | notches | natural unit | canonical scale | heavy line |
|---|---|---|---|---|
| `DEGREES` | 0.5, 1, 2, 3, 4 | °C | 1 K per degree | 1, 5, 10, 6, 20 °C (the R5.9 table) |
| `HEIGHT` | 0.1, 0.2, 0.25, 0.5, 1, 2 | kft | 304.8 × 9.80665 m² s⁻² per kft | every 10th, 5th, 4th, 2nd, 5th, 5th — a whole kilofoot from every notch |

The slider re-ranges to the field's ladder and its label stays in display units, so a
500 hPa `geopot` map opens at `0.2 kft` with six notches, reads `60.96 gpm` or `6.096 dam`
when the Units combo says so, and draws the **same lines** either way (**G15** — the
ladder is canonical, the label is not). `--isoline-step` takes the spacing in the field's
natural unit and snaps to the nearest notch, saying so (`--isoline-step 0.3` on a height →
"offers 0.1, 0.2, 0.25, 0.5, 1, 2 kft; using 0.25"); it used to refuse anything off the
degree list, which would have made every height spacing an error.

**Geopotential → height, in kilofeet by default** (asked for 2026-10-01, after the first
cut had used gpm). `GEOPOT` offers `kft` first, then `gpm`, `dam` and the raw `m² s⁻²`;
1 kft is 304.8 m exactly, so the affine is `1 / (304.8 g)`. It is contoured at
`0.2 kft` in its canonical m² s⁻², anchored at 0 so the lines sit on round kilofeet with
every 5th (a whole kilofoot) heavier. The height row and the profile axis read in the same
unit the chart is set to — one choice, remembered under `units/geopot`, and the companion
view is the very object "Map shows" installs as the chart, so changing the unit on the
chart moves the row. The isolines slider, the Level control, the units combo and the map
title all work on it unchanged.

## R9.4 The date, the value, and the height beside them

On the deterministic readout the **Time** and **Value here** rows are set four points
larger and bold — a size up, which the eye reads as the answer, rather than bold alone,
which it reads as a heading. The ensemble's six-row panel is untouched.

Between them and the column extremes sits **Height (geopot)**: the geopotential height of
the level shown, at the chosen point and time, from the run's own `geopot` file. The
companion is opened automatically beside any pressure-level map whose run has `geopot` on
disk — automatic rather than opt-in, unlike the wind overlay, because the height is a
reading *of the map on screen* and it was asked for next to the value — off the UI thread
on the same `BuildWorker`, under a new request kind `FIELD` that opens one file as itself
and does **not** stamp it as derived (so "Map shows" can later install the very same view
as the `geopot` map, and does: the companion is reused, not reopened). It is kept while it
fits the map (`_overlay_shape`) and checked with `check_pairable` before it is accepted
(**G45**): a `geopot` with fewer steps than the map is refused with the reason on the row's
tooltip, and the profile falls back to pressure.

The row is hidden where it would lie: on a 2 m field (no level to be the height of), on
the ensemble, and on `geopot` itself, whose value *is* the height.

## R9.5 The humidity profile

On `rh` the right-hand panel is a **vertical profile**: the value across, the geopotential
height up, one point per level, at the time step on the slider — the question a forecaster
asks of one point and one time ("where in the column is it moist?") rather than the time
graph's ("how does it change"). Specifically:

* **y is the model's height**, from the companion above, so the levels sit where the model
  says they are today. Without `geopot` the levels are drawn against **pressure**, inverted
  so up is still up, and the axis says so — a profile against a guessed height would look
  just as convincing and be off by hundreds of metres.
* The right-hand axis names the levels in hPa, **thinned to what fits** (`MIN_TICK_GAP_PX`):
  the lowest levels are 25 hPa (~220 m) apart, which is a few pixels on a 15 km panel, and
  22 labels printed over each other name nothing. The map's current level is never thinned.
* The map's level is marked (a ring and a line); **▲▼** move it; **clicking** a level on
  the profile moves the map to it; **hovering** one reports that level's value and height
  to the readout, the way hovering the time graph reports another time, and leaving
  restores the map's.
* The value axis is pinned to the dataset range (A1), so a profile at 03Z and one at 15Z
  are read on one scale.
* A **Profile** tick on toolbar row 2 is live on any column of pressure levels, ticked by
  default on `rh` — the field it was asked for — and remembered per field within the
  window, like the isoline spacing. Unticked, the graph is the R7 time-height section.

## R9.6 Measured, not assumed

On this machine (the one R4.4 measured as roughly 3× slower than v2's), against the
16.7 ms frame:

| operation | ms |
|---|---|
| load the elevation bundle (901×601 int16, once per process) | 9.2 |
| hillshade of the whole bundle | 7.7 |
| land-mask rasterise (the painter, 541,501 cells) | 0.9 |
| the relief image, first build, all in | **8.7** |
| the cached check on every later `set_terrain` | 0.005 |
| CAPE +110 h, `refresh_map` + paint, terrain off → on | 15.19 → **15.54** |
| 21-step scrub with terrain on | 3.75 per step |
| profile refresh (22 levels, with heights) | 0.58 |
| height lookup at one level (the readout) | 0.004 |
| `set_time`, time graph / profile showing | 1.25 / 1.93 |

So the relief costs about a third of a millisecond a frame once built, and the profile
about 0.7 ms a step over the time graph.

| claim | measurement |
|---|---|
| the hillshade is right-way-up | a flat field is exactly 1.0 everywhere; a ramp falling to the north-west is 1.0 (lit) and the same ramp the other way up 0.73; an east-facing ramp is darker than a north-facing one of equal degrees per cell, because a degree of longitude is shorter |
| the mask is the coastline | Jerusalem and the Dead Sea shore opaque; the Mediterranean off Haifa, Gaza and Beirut clear — 402,649 land cells of 541,501 |
| the bundle is the real relief | Hermon > 2000 m, Jerusalem 600–1000 m, the Dead Sea < −300 m, the sea off Haifa < 0 |
| the same lines in kft, gpm, dam and m² s⁻² | 18 kft = 5,486.4 gpm = 548.64 dam each land on one level of their own ladder |
| the profile is the column | x equals `series(iy, ix)[t]` and y the companion's `geopot / (304.8 g)` in kft, at two points and two times |

## R9.7 Gotchas found while building v9

* **G47 — the test suite's `QSettings` isolation did not isolate on macOS.** `conftest.py`
  set `setDefaultFormat(IniFormat)` and `setPath(...)` (G25), but on macOS
  `QSettings(org, app)` keeps **NativeFormat** regardless and goes straight to
  `~/Library/Preferences/com.ims.IconEnsembleViewer.plist`. MEASURED: a marker written
  natively before a test file was gone after it — every test run since R2 has been wiping
  the developer's real preferences (`last_dir`, the units choices), and it was found only
  because a remembered Topography tick vanished between two screenshots while the suite
  ran in the background. The app now opens its preferences through one seam,
  `ui.main.app_settings()`, and `clean_settings` points it at a throw-away `.ini` per test.
  Re-measured: the marker survives.
* **G48 — pyqtgraph pins the colorbar axis to 45 px, and G36's margin was too small for it
  anyway.** 45 px fits three digits; a CAPE scale read `0, 100, 200, 300` for 0..3000 and a
  500 hPa height `552` for 5,520 gpm. Unpinning the axis (`setWidth(None)`) was necessary
  and not sufficient: with `TITLE_MARGIN_PX = 110` the layout measured 738 px in a 718 px
  widget — the wrapped title plus 41 px of plot frame plus a 63 px colorbar — so the
  colorbar's last 20 px, and the last digit of every label over 999, were off the edge of
  the widget with no error anywhere. The margin is 150 px now, and
  `test_a_four_digit_colorbar_fits_inside_the_map_widget` holds the layout inside the widget.
* **The python.org macOS build ships no root certificates for `urllib`.** Both build tools
  fetch with `certifi`'s bundle when it is importable (it comes with `requests`, which the
  downloader already needs), else the default context.
* **A `Map shows` entry for the open file is keyed `base`, not `file:<field>`.** A test that
  switches fields twice has to recompute the keys after each switch, because the open file
  changes which entry is `base`.

## R9.8 Verified

`639 passed` (589 from R1–R8 and the merged R5.9, unchanged and green except one row-set
assertion that gained the height row, + 50 new). Nothing skips here: the 407 MB reference
file and `netCDF4` are both present.

| what | where | verified by |
|---|---|---|
| the relief arithmetic and the bundle | `terrain.py`, `mapdata/levant_etopo1.npz` | `test_terrain.py` (13) — flat is 1.0, lit stays white, away darkens, cos(lat) in the gradient, the mask in the alpha, known places, the clamp, a missing or descending bundle refused |
| the option | `ui/mapview.py`, `ui/main.py` | `test_ui_terrain.py` (8) — off until ticked; Multiply at z between field and isolines; sea clear and land shaded at named places; built once across time and fields; unticked takes the note down; remembered for the next window; disabled with the reason when the bundle is missing |
| the ladder | `isolines.py`, `fieldview.py`, `derived.py` | `test_isolines.py` (+4) and `test_ui_isolines.py` (+5) — kft/gpm/dam/raw are the same lines; the slider re-ranges to six notches and back to five; metres and decametres relabel without moving; the CLI snaps and says so |
| the readout and the height | `ui/readout.py`, `ui/main.py` | `test_ui_profile.py` — the height row is the run's `geopot / (304.8 g)` at the level, point and time, kft by default and following the chart's unit; larger date and value; hidden on `geopot`, on a 2 m field and on the ensemble; a short `geopot` refused (G45); the companion kept across fields and reused as the `geopot` map |
| the profile | `ui/profileview.py` | `test_ui_profile.py` (19 in all) — `rh` opens as a profile with height up; the column at the point and time; hPa on the right axis; the level marked and following ▲▼; click moves the map, hover the readout; the slider moves it; pinned value axis; `temp` opens on the time graph and can switch; the choice kept per field; pressure fallback without `geopot` |
| the colorbar | `ui/mapview.py` | `test_ui_profile.py` — a four-digit scale fits inside the widget |

End to end under `QT_QPA_PLATFORM=offscreen`: the real CAPE file at +110 h with
`--topo on` shows the plume vivid over the shaded Judean hills, white sea and legible
coastline; a synthetic deterministic run at the real 281×201×22 resolution renders `rh` as
a profile against its `geopot` heights with the readout's height row in kft (4.35 kft at 850 hPa), `temp`
at 850 hPa with 1 °C isolines over the relief, `geopot` at 500 hPa with `--isoline-step
60` and again with `--isoline-step 500 --units dam` (snapped to 12 dam, with the sentence);
`--profile on` on the ensemble prints one sentence and carries on.

## R9.9 Deliberately not done

* **Elevation contours.** A "topographic map" can also mean height contours; the isolines
  machinery would draw ETOPO1 at 250 m in a few lines, but they would compete with the
  field's own isolines for the same ink. The shading carries the relief for now.
* **A terrain height for surface fields.** The readout's height row is the height *of a
  pressure level*; a 2 m field would want the ground height instead, which `terrain.
  height_at` already answers from ETOPO1 — but that is not "from model", so it was left out
  rather than labelled as something it is not.
* **A profile of the other 3-D fields by default.** The Profile tick is live on all of
  them; only `rh` opens that way, as asked.
* **Isobars on `pres_msl`.** One registry line (a hPa ladder of 1, 2, 4, 5, 10), not asked
  for.

## R9.10 Running it

```bash
venv/bin/python -m imsicon --topo on data/ICON_ENS_..._CAPE_ML.nc            # relief under any map
venv/bin/python -m imsicon --level 500 --isoline-step 0.5 data/IE_..._geopot.nc  # a height chart, kft
venv/bin/python -m imsicon --units gpm data/IE_..._geopot.nc                    # or dam, or m2 s-2
venv/bin/python -m imsicon --point 31.8 35.2 data/IE_..._rh.nc                 # the profile, if geopot is beside it
venv/bin/python -m imsicon --profile on --level 700 data/IE_..._temp.nc
venv/bin/python tools/build_terrain.py --fetch                                 # rebuild the elevation bundle (dev only)
```

Toolbar row 2 gained **Topography** (after Wind barbs) and **Profile** (at the end); the
Isolines slider's notches are now the field's own. The readout on a pressure-level map
reads, in order: time, level, value, **height**, highest and lowest in the column.
