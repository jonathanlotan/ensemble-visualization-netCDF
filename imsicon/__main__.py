"""python -m imsicon [file] - launch the ensemble viewer."""
import argparse
import sys

from PySide6 import QtCore, QtWidgets

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
    args = ap.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('IMS ICON Ensemble Viewer')
    window = MainWindow(args.path)
    window.show()

    if args.screenshot:
        if not args.path:
            ap.error('--screenshot needs a file path')
        _shoot(app, window, args)
    return app.exec()


def _shoot(app, window, args):
    """Headless-ish verification: settle, pick a point, grab the window, quit."""
    def step():
        if window.ds is None:
            QtCore.QTimer.singleShot(200, step)
            return
        if window.scan is not None and window.scan.isRunning():
            window.scan.wait(5000)
            window._on_scan_done(window.ds.value_range)
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
