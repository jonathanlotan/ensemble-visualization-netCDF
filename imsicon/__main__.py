"""python -m imsicon [file] - launch the ensemble viewer."""
import argparse
import sys

from PySide6 import QtCore, QtWidgets

from . import derived, ingest, isolines, ncwrite, products, transform
from .ui import derivedialog
from .ui.main import MainWindow


def main(argv=None):
    ap = argparse.ArgumentParser(prog='imsicon',
                                 description='View IMS ICON ensemble NetCDF files.')
    ap.add_argument('path', nargs='?', help='.nc or .nc.bz2 file (default: ask at startup)')
    ap.add_argument('--screenshot', metavar='PNG',
                    help='dev: load, render, save a PNG and exit (no interaction)')
    ap.add_argument('--point', nargs=2, type=float, metavar=('LAT', 'LON'),
                    help='dev: grid point to select for --screenshot')
    ap.add_argument('--time', type=int, default=None, help='dev: time index for --screenshot')
    ap.add_argument('--units', metavar='LABEL',
                    help="display units, e.g. C, F, K, mm, kt (default: the field's own)")
    ap.add_argument('--rate', metavar='WINDOW', default=None,
                    help='de-accumulate to a window: 1h, 3h, or 0 for the stored values '
                         '(accumulated fields only)')
    ap.add_argument('--derive', choices=('dewpoint', 'depression', 'wind'), default=None,
                    help='show a derived field instead of the file itself: dewpoint is '
                         'TD_2M from T_2M and RELHUM_2M, depression is T_2M - TD_2M, and '
                         'wind is the speed map with wind barbs from U_10M and V_10M. The '
                         'other input files are found beside the given file, by the run '
                         'in its name')
    ap.add_argument('--difference', nargs=2, metavar=('A', 'B'), default=None,
                    help='show the difference between two fields of this run, e.g. '
                         '--difference T_2M T_S')
    ap.add_argument('--write', metavar='PATH', default=None,
                    help='write the field on screen to a NetCDF-3 file, then exit')
    ap.add_argument('--isolines', choices=('on', 'off'), default=None,
                    help='contour lines over the map: every 1 degC on a temperature, '
                         'every 0.5 degC on a difference such as T-Td (default: on '
                         'wherever the field has them)')
    ap.add_argument('--level', metavar='HPA', default=None,
                    help='which position on the file\'s second axis to show: a pressure '
                         'in hPa on a 3-D field of the deterministic run (e.g. --level '
                         '850, snapped to the nearest level), or a member number on an '
                         'ensemble file')
    ap.add_argument('--barbs', choices=('on', 'off'), default=None,
                    help="draw the run's wind barbs over the map, whatever field it "
                         'shows (needs the run\'s wind components beside the file); the '
                         'wind map draws its own either way')
    ap.add_argument('--isoline-step', type=float, default=None, metavar='SPACING',
                    help='how close the isolines are, snapped to the nearest spacing the '
                         'field offers: '
                         + ', '.join(f'{step:g}' for step in isolines.DEGREES.steps)
                         + ' degrees Celsius on a temperature or a difference such as '
                           'T-Td; '
                         + ', '.join(f'{step:g}' for step in isolines.HEIGHT.steps)
                         + " ft on a geopotential height chart (default: the field's "
                           'own, 1 degC on a temperature, 0.5 on a difference, 200 ft '
                           'on a height)')
    ap.add_argument('--topo', choices=('on', 'off'), default=None,
                    help='shaded relief under the map, from the bundled ETOPO1 elevation '
                         'grid (default: as last used)')
    ap.add_argument('--profile', choices=('on', 'off'), default=None,
                    help='on a pressure-level file, draw the right-hand panel as a '
                         'vertical profile (value across, geopotential height up) '
                         'instead of the time graph (default: on for rh, off otherwise)')
    ap.add_argument('--sort', action='store_true',
                    help='T-Td only: colour the map only where the depression is under '
                         '2 degC (2 white, 1 yellow-orange, 0 red), leaving drier air '
                         'uncoloured')
    ap.add_argument('--settings', action='store_true',
                    help='open the Settings window on its own (credentials, points on '
                         'the map, per-map defaults) and exit when it closes')
    args = ap.parse_args(argv)
    if args.settings:
        from .ui.settingsdialog import SettingsDialog
        app = QtWidgets.QApplication(sys.argv[:1])
        app.setApplicationName('IMS ICON Ensemble Viewer')
        dialog = SettingsDialog()
        dialog.exec()
        return 0
    if args.derive and args.difference:
        ap.error('--derive and --difference choose the same thing; give only one')

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('IMS ICON Ensemble Viewer')
    window = MainWindow(args.path)
    window.show()

    wants_post = any((args.units, args.rate is not None, args.derive, args.difference,
                      args.write, args.isolines is not None, args.isoline_step is not None,
                      args.sort, args.level is not None, args.barbs is not None,
                      args.topo is not None, args.profile is not None))
    if wants_post and not args.path:
        ap.error('--units/--rate/--derive/--difference/--write/--isolines/--isoline-step'
                 '/--sort/--level/--barbs/--topo/--profile need a file path')
    if args.screenshot:
        if not args.path:
            ap.error('--screenshot needs a file path')
        _shoot(app, window, args)               # _shoot runs the same post-load steps
    elif wants_post:
        QtCore.QTimer.singleShot(0, lambda: _apply_display(window, args, ap))
    return app.exec()


# The degree sign is awkward to type at a shell prompt.
UNIT_ALIASES = {'C': '°C', 'c': '°C', 'F': '°F', 'f': '°F', 'degC': '°C', 'degF': '°F'}


def _derive(window, args):
    """Replace the loaded file with a field derived from its run (--derive/--difference).

    The second (or both) input files are found beside the one that was opened, by the run
    in its name -- the same discovery the Derived field dialog does.
    """
    run = f'{window.ds.run_init:%Y%m%d%H}'
    family = getattr(window.ds, 'family', None) or products.ENSEMBLE
    available = ingest.fields_of_run(
        ingest.scan_for_fields(ingest.search_roots(window.ds.path)), family, run)
    if args.difference:
        kind, wanted = derivedialog.DIFFERENCE, list(args.difference)
    elif args.derive == 'wind':
        kind = derivedialog.WIND
        # Prefer the pair the open file belongs to: opening `u` and asking for a wind map
        # means the wind on those pressure levels, not the 10 m wind of the same run.
        pair = derived.wind_pair_for(family, available, window.ds.field)
        wanted = list(pair) if pair else [derived.role_field(family, k)
                                          for k in ('zonal', 'meridional')]
    else:
        kind = (derivedialog.DEW_POINT if args.derive == 'dewpoint'
                else derivedialog.DEPRESSION)
        wanted = [derived.role_field(family, k) for k in ('temperature', 'humidity')]
    if not all(wanted):
        print(f'the {family.short} product has no field for one of the inputs this '
              'derived field needs', file=sys.stderr)
        return False
    missing = [field for field in wanted if field not in available]
    if missing:
        print(f'{" and ".join(missing)} for run {run} was not found beside '
              f'{window.ds.path.name}', file=sys.stderr)
        return False
    paths = [ingest.resolve(available[field]) for field in wanted]
    request = derivedialog.DerivedRequest(kind, paths, ' - '.join(wanted))
    try:
        view = derivedialog.build(request, dict(window.opened))
    except derived.PairError as exc:
        print(str(exc), file=sys.stderr)
        return False
    window._install(view, request.title)
    if window.scan is not None and window.scan.isRunning():
        window.scan.wait(30000)
        window._on_scan_done(window.ds.value_range)
    return True


def _write(window, path):
    """--write: save what is on screen, in the view's own canonical units."""
    written = ncwrite.write_canonical(path, window.ds)
    print(f'wrote {written} ({window.ds.field} in {window.ds.canonical_units})')


def _apply_display(window, args, ap, tries=0):
    """Apply --derive/--difference/--units/--rate/--write once the file has loaded.

    Order matters: the derived field is built first, because --units then applies to what
    is actually on screen rather than to the file that was opened.
    """
    if window.ds is None:
        if tries < 60:
            QtCore.QTimer.singleShot(200, lambda: _apply_display(window, args, ap, tries + 1))
        return
    if getattr(window, '_display_applied', False):
        return                      # --screenshot and the timer both call this
    window._display_applied = True
    if (args.derive or args.difference) and not _derive(window, args):
        raise SystemExit(2)
    if args.units:
        label = UNIT_ALIASES.get(args.units, args.units)
        if not window.ds.set_units(label):
            print(f'--units {args.units!r}: {window.ds.field} offers '
                  f'{window.ds.unit_labels}', file=sys.stderr)
        else:
            window.units_combo.setCurrentText(label)
    if args.rate is not None:
        hours = int(str(args.rate).lower().replace('h', '').strip() or 0)
        if not window.ds.can_rate and hours:
            print(f'--rate {args.rate!r}: {window.ds.field} is not an accumulated field, '
                  'so it has no window rate', file=sys.stderr)
        else:
            window.rate_combo.setCurrentText(transform.rate_label(hours))
    if args.isolines is not None:
        if not window.isolines_check.isEnabled() and args.isolines == 'on':
            print(f'--isolines on: {window.ds.display_name} is not a contoured field',
                  file=sys.stderr)
        window.isolines_check.setChecked(args.isolines == 'on')
    if args.isoline_step is not None:
        # Set after --isolines, and deliberately even when the lines are switched off:
        # the spacing is a property of the field, so ticking Isolines afterwards draws
        # the interval that was asked for rather than the default.
        if not window.isolines_check.isEnabled():
            print(f'--isoline-step: {window.ds.display_name} is not a contoured field',
                  file=sys.stderr)
        else:
            # In the field's natural unit -- degrees on a temperature, gpm on a height --
            # and snapped to the ladder the slider offers, as the slider itself would.
            ladder = window.ds.isoline_ladder
            chosen = ladder.nearest(args.isoline_step)
            if abs(chosen - args.isoline_step) > 1e-9:
                print(f'--isoline-step {args.isoline_step:g}: {window.ds.display_name} '
                      f'offers {", ".join(f"{s:g}" for s in ladder.steps)} {ladder.unit}; '
                      f'using {chosen:g}', file=sys.stderr)
            window.isoline_step_slider.setValue(ladder.steps.index(chosen))
    if args.topo is not None:
        if args.topo == 'on' and not window.topo_check.isEnabled():
            print('--topo on: the bundled elevation grid is missing, so there is no '
                  'relief to draw', file=sys.stderr)
        window.topo_check.setChecked(args.topo == 'on')
    if args.profile is not None:
        if args.profile == 'on' and not window.profile_check.isEnabled():
            print(f'--profile on: {window.ds.display_name} is not on pressure levels, so '
                  'there is no column to draw as a profile', file=sys.stderr)
        window.profile_check.setChecked(args.profile == 'on')
        if window._height_builder is not None and window._height_builder.isRunning():
            window._height_builder.wait(60000)
            QtWidgets.QApplication.processEvents()
    if args.level is not None:
        _apply_level(window, args.level)
    if args.barbs is not None:
        if args.barbs == 'on' and not window.barbs_check.isEnabled():
            print(f'--barbs on: neither {window.ds.display_name} nor the rest of this run '
                  'on disk has a wind to draw', file=sys.stderr)
        window.barbs_check.setChecked(args.barbs == 'on')
        if window._overlay_builder is not None and window._overlay_builder.isRunning():
            window._overlay_builder.wait(60000)
            QtWidgets.QApplication.processEvents()
    if args.sort:
        if not window.sort_check.isEnabled():
            print(f'--sort: {window.ds.display_name} has no sort band; it applies to the '
                  'dew point depression (--derive depression)', file=sys.stderr)
        window.sort_check.setChecked(True)
    if args.write:
        _write(window, args.write)
        if not args.screenshot:
            QtCore.QTimer.singleShot(0, QtWidgets.QApplication.quit)


def _apply_level(window, request):
    """--level: a pressure in hPa on a level axis, a member number on an ensemble."""
    axis = window.ds.axis
    try:
        value = float(str(request).lower().replace('hpa', '').strip())
    except ValueError:
        print(f'--level {request!r}: expected a number', file=sys.stderr)
        return
    if axis.is_pressure:
        index = axis.nearest(value)
        window.set_level(index)
        if abs(axis.values[index] - value) > 1e-6:
            print(f'--level {value:g}: nearest level in the file is '
                  f'{axis.labels[index]}', file=sys.stderr)
    elif axis.n > 1:
        window.set_level(int(value) - 1)             # member numbers are 1-based on screen
    else:
        print(f'--level: {window.ds.display_name} has {axis.describe()}, so there is '
              'nothing to choose', file=sys.stderr)


def _shoot(app, window, args):
    """Headless-ish verification: settle, pick a point, grab the window, quit."""
    def step():
        if window.ds is None:
            QtCore.QTimer.singleShot(200, step)
            return
        if window.scan is not None and window.scan.isRunning():
            window.scan.wait(5000)
            window._on_scan_done(window.ds.value_range)
        if window._height_builder is not None and window._height_builder.isRunning():
            window._height_builder.wait(60000)
            app.processEvents()
        if any((args.units, args.rate is not None, args.derive, args.difference,
                args.write, args.isolines is not None, args.isoline_step is not None,
                args.sort, args.level is not None, args.barbs is not None,
                args.topo is not None, args.profile is not None)):
            _apply_display(window, args, None)
        if args.point:
            window.select_point(*window.ds.nearest_index(*args.point))
        if args.time is not None:
            window.set_time(args.time)
        window.plot.hovered.emit(window.t)
        app.processEvents()
        QtCore.QTimer.singleShot(400, save)

    def save():
        window.grab().save(args.screenshot)
        print(f'wrote {args.screenshot}')
        app.quit()

    QtCore.QTimer.singleShot(300, step)


if __name__ == '__main__':
    raise SystemExit(main())
