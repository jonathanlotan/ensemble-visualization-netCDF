"""python -m imsicon [file] - launch the ensemble viewer."""
import argparse
import sys

from PySide6 import QtCore, QtWidgets

from . import transform
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
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('IMS ICON Ensemble Viewer')
    window = MainWindow(args.path)
    window.show()

    if args.units or args.rate is not None:
        if not args.path:
            ap.error('--units/--rate need a file path')
        QtCore.QTimer.singleShot(0, lambda: _apply_display(window, args, ap))
    if args.screenshot:
        if not args.path:
            ap.error('--screenshot needs a file path')
        _shoot(app, window, args)
    return app.exec()


# The degree sign is awkward to type at a shell prompt.
UNIT_ALIASES = {'C': '°C', 'c': '°C', 'F': '°F', 'f': '°F', 'degC': '°C', 'degF': '°F'}


def _apply_display(window, args, ap, tries=0):
    """Apply --units/--rate once the file has finished loading."""
    if window.ds is None:
        if tries < 60:
            QtCore.QTimer.singleShot(200, lambda: _apply_display(window, args, ap, tries + 1))
        return
    if getattr(window, '_display_applied', False):
        return                      # --screenshot and the timer both call this
    window._display_applied = True
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


def _shoot(app, window, args):
    """Headless-ish verification: settle, pick a point, grab the window, quit."""
    def step():
        if window.ds is None:
            QtCore.QTimer.singleShot(200, step)
            return
        if window.scan is not None and window.scan.isRunning():
            window.scan.wait(5000)
            window._on_scan_done(window.ds.value_range)
        if args.units or args.rate is not None:
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
