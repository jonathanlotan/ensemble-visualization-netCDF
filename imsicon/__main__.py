"""python -m imsicon [file] - launch the ensemble viewer."""
import argparse
import sys

from PySide6 import QtCore, QtWidgets

from . import derived, ingest, ncwrite, transform
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
    args = ap.parse_args(argv)
    if args.derive and args.difference:
        ap.error('--derive and --difference choose the same thing; give only one')

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('IMS ICON Ensemble Viewer')
    window = MainWindow(args.path)
    window.show()

    wants_post = any((args.units, args.rate is not None, args.derive, args.difference,
                      args.write))
    if wants_post and not args.path:
        ap.error('--units/--rate/--derive/--difference/--write need a file path')
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
    available = ingest.scan_for_fields(ingest.search_roots(window.ds.path))
    if args.difference:
        kind, wanted = derivedialog.DIFFERENCE, list(args.difference)
    elif args.derive == 'wind':
        kind = derivedialog.WIND
        wanted = [derived.WIND_INPUTS[k][0] for k in ('zonal', 'meridional')]
    else:
        kind = (derivedialog.DEW_POINT if args.derive == 'dewpoint'
                else derivedialog.DEPRESSION)
        wanted = list(derived.DEW_POINT_INPUTS[k][0] for k in ('temperature', 'humidity'))
    missing = [field for field in wanted if (run, field) not in available]
    if missing:
        print(f'{" and ".join(missing)} for run {run} was not found beside '
              f'{window.ds.path.name}', file=sys.stderr)
        return False
    paths = [ingest.resolve(available[(run, field)]) for field in wanted]
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
    if args.write:
        _write(window, args.write)
        if not args.screenshot:
            QtCore.QTimer.singleShot(0, QtWidgets.QApplication.quit)


def _shoot(app, window, args):
    """Headless-ish verification: settle, pick a point, grab the window, quit."""
    def step():
        if window.ds is None:
            QtCore.QTimer.singleShot(200, step)
            return
        if window.scan is not None and window.scan.isRunning():
            window.scan.wait(5000)
            window._on_scan_done(window.ds.value_range)
        if any((args.units, args.rate is not None, args.derive, args.difference,
                args.write)):
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
