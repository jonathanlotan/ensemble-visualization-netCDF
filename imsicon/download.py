"""Fetching a run's field straight from the IMS server (CLAUDE.md Phase 8, v2.md 6.2).

Turns "there is a new run" -> "I am looking at it" into one dialog. This module is the
non-Qt half: the field catalogue, credential lookup, the directory listing, and a
resumable download. `ui/downloaddialog.py` drives it.

Security rules, all of them load-bearing:

* **G7 -- the server speaks NTLM/Negotiate, not HTTP Basic.** `curl --ntlm` works and a
  plain `-u` returns 401, so `requests` alone is not enough; `requests_ntlm` is required.
  Both are imported lazily so the app (and its frozen exe) does not carry an HTTP stack it
  only needs for this one dialog.
* **Credentials never touch `QSettings`** -- that is plaintext on disk. Keyring (Windows
  Credential Manager / macOS Keychain) first, `IMS_USER`/`IMS_PASS` second, prompt last.
  They are never logged, never put in a URL or query string, and a failure says
  "authentication failed" without echoing what was tried.
* **The listing is UNTRUSTED input.** Only `ICON_ENS_<10 digits>_<FIELD>.nc.bz2` is
  accepted, by a strict regex; everything else in the HTML is ignored, and a local path is
  never built from a server string without `Path(name).name` on top of that (G27).
"""
import os
import re
import shutil
from pathlib import Path

BASE = 'https://data.israel-meteo-service.org/ims/IMS_ICON_ENSEMBLE/'
CHUNK = 1 << 20
TIMEOUT = 120
MIN_FREE_BYTES = 1 << 30        # refuse to start a 262 MB download under 1 GB free

# Keyring service name. The password is stored under the username, never under a fixed key.
KEYRING_SERVICE = 'ims-icon'


class DownloadError(Exception):
    """Anything that stopped a download, in words a dialog can show."""


class AuthError(DownloadError):
    """Credentials were rejected. Deliberately says nothing about what was tried."""


class MissingDependency(DownloadError):
    """requests / requests-ntlm are not installed."""


class Cancelled(Exception):
    """The user cancelled. Not an error, and the `.part` file is deliberately kept."""


# ---- the catalogue: "choose a map (cape, precipitation or other)" ----------------------
# One file per field per run (CLAUDE.md 0.2). `group` only drives how the dialog sorts the
# list; `field` is what the file name is built from.
class Product:
    __slots__ = ('field', 'label', 'group', 'note')

    def __init__(self, field, label, group, note=''):
        self.field, self.label, self.group, self.note = field, label, group, note

    def __repr__(self):
        return f'<Product {self.field}>'


PRODUCTS = (
    Product('CAPE_ML', 'CAPE - instability', 'Convection',
            'CAPE of the mean surface layer parcel [J kg-1] - the "cape index"'),
    Product('TOT_PREC', 'Precipitation - total', 'Convection',
            'accumulated since model start [kg m-2 = mm]; the viewer can show 1 h / 3 h rates'),
    Product('T_2M', 'Temperature - 2 m', 'Temperature and humidity',
            'stored in K, shown in °C by default'),
    Product('T_S', 'Temperature - surface', 'Temperature and humidity',
            'weighted surface temperature, stored in K'),
    Product('RELHUM_2M', 'Relative humidity - 2 m', 'Temperature and humidity',
            'needed, with T_2M, to derive the dew point'),
    Product('U_10M', 'Wind - zonal component (U) 10 m', 'Wind', 'm s-1, positive eastward'),
    Product('V_10M', 'Wind - meridional component (V) 10 m', 'Wind',
            'm s-1, positive northward'),
    Product('VMAX_10M', 'Wind - gust at 10 m', 'Wind',
            'max since the previous full hour - already per-interval, never de-accumulate'),
    Product('CLCT', 'Cloud cover - total', 'Cloud', '%'),
    Product('CLCL', 'Cloud cover - low', 'Cloud', '%'),
    Product('CLCM', 'Cloud cover - medium', 'Cloud', '%'),
    Product('CLCH', 'Cloud cover - high', 'Cloud', '%'),
    Product('ASWDIFD_S', 'Solar radiation - diffuse downward', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('ASWDIR_S', 'Solar radiation - direct downward', 'Radiation',
            'W m-2, stored as a MEAN since model start'),
    Product('H_SNOW', 'Snow depth', 'Surface', 'm; ~all zero in summer'),
)
BY_FIELD = {p.field: p for p in PRODUCTS}
GROUPS = tuple(dict.fromkeys(p.group for p in PRODUCTS))

# The two fields the dew point is derived from (derived.dew_point).
DEW_POINT_FIELDS = ('T_2M', 'RELHUM_2M')
# ...and the two the wind map is (derived.wind). Both pairs are useless one file at a
# time, which is why the dialog offers a one-click tick for each.
WIND_FIELDS = ('U_10M', 'V_10M')


def product_label(field):
    product = BY_FIELD.get(field)
    return product.label if product else field


# ---- credentials -----------------------------------------------------------------------
def keyring_module():
    """The `keyring` module, or None. Optional: the app must open without it."""
    try:
        import keyring
        return keyring
    except Exception:
        return None


def stored_credentials():
    """-> (user, password) from the OS keychain or the environment, else None.

    Never reads QSettings, and never returns a partially-filled pair.
    """
    user, password = os.environ.get('IMS_USER'), os.environ.get('IMS_PASS')
    if user and password:
        return user, password
    keyring = keyring_module()
    if keyring is None:
        return None
    try:
        user = user or keyring.get_password(KEYRING_SERVICE, 'username')
        if not user:
            return None
        password = keyring.get_password(KEYRING_SERVICE, user)
    except Exception:
        return None
    return (user, password) if user and password else None


def remember_credentials(user, password):
    """Store in the OS keychain (Windows Credential Manager / Keychain). -> bool."""
    keyring = keyring_module()
    if keyring is None:
        return False
    try:
        keyring.set_password(KEYRING_SERVICE, 'username', user)
        keyring.set_password(KEYRING_SERVICE, user, password)
        return True
    except Exception:
        return False


def forget_credentials():
    keyring = keyring_module()
    if keyring is None:
        return False
    try:
        user = keyring.get_password(KEYRING_SERVICE, 'username')
        if user:
            keyring.delete_password(KEYRING_SERVICE, user)
        keyring.delete_password(KEYRING_SERVICE, 'username')
        return True
    except Exception:
        return False


def make_session(user, password):
    """An authenticated `requests.Session`. G7: NTLM, so plain Basic auth returns 401."""
    try:
        import requests
        from requests_ntlm import HttpNtlmAuth
    except ImportError as exc:
        raise MissingDependency(
            'The built-in downloader needs `requests` and `requests-ntlm`:\n\n'
            '    pip install requests requests-ntlm\n\n'
            'They are optional because the viewer itself needs no HTTP stack. '
            f'({exc})') from exc
    session = requests.Session()
    session.auth = HttpNtlmAuth(user, password)
    return session


# ---- the listing -----------------------------------------------------------------------
# Strict: 10-digit run id, upper-case field, exactly this extension. Anything else in the
# server's HTML is ignored rather than interpreted.
NAME_RE = re.compile(r'^ICON_ENS_(?P<run>\d{10})_(?P<field>[A-Z0-9_]+)\.nc\.bz2$')
_ANCHOR_RE = re.compile(r'<a\b[^>]*?href\s*=\s*["\']([^"\']*)["\'][^>]*>(.*?)</a>',
                        re.IGNORECASE | re.DOTALL)
# A size is a standalone integer. The lookarounds exclude the digits of the timestamp that
# sits next to it on the same line -- without them "24-Aug-2026 12:20 261123456" parses as
# a 12-byte file, which is exactly the kind of plausible-looking wrong number that makes a
# size check useless.
_SIZE_TOKEN = re.compile(r'(?<![\d.,:/\-])(\d[\d,]*)(?![\d.,:/\-])')


class RemoteFile:
    """One `ICON_ENS_<run>_<FIELD>.nc.bz2` offered by the server."""
    __slots__ = ('name', 'run', 'field', 'size')

    def __init__(self, name, run, field, size=None):
        self.name, self.run, self.field, self.size = name, run, field, size

    @property
    def url(self):
        return BASE + self.name

    @property
    def label(self):
        return product_label(self.field)

    def local_name(self):
        """G27: the file name used on disk is re-derived, never taken from the server.

        `Path(name).name` alone is not enough -- it still passes through `..%2f` style
        oddities on some platforms -- so the name is REBUILT from the two validated
        capture groups. A server string can therefore never choose a path.
        """
        return f'ICON_ENS_{self.run}_{self.field}.nc.bz2'

    def __repr__(self):
        return f'<RemoteFile {self.name} size={self.size}>'


def _int_or_none(text):
    try:
        value = int(text.replace(',', ''))
    except (AttributeError, ValueError):
        return None
    return value if value > 0 else None


# A listing row ends at a newline or a <br>. Both matter: a row with no size of its own
# must not inherit the size that ended the row above it.
_ROW_BREAK = re.compile(r'\n|<br\s*/?>', re.IGNORECASE)


def _size_before(text):
    """IIS: `8/23/2026 12:20 PM   274989056 <A HREF=...>` -- the size ends the row."""
    text = _ROW_BREAK.split(text)[-1].rstrip()
    last = None
    for last in _SIZE_TOKEN.finditer(text):
        pass
    if last is None or last.end() != len(text):
        return None
    return _int_or_none(last.group(1))


def _size_after(text):
    """nginx/apache: `<a ...>NAME</a>   24-Aug-2026 12:20   261123456` -- size trails."""
    last = None
    for last in _SIZE_TOKEN.finditer(_ROW_BREAK.split(text)[0]):
        pass
    return _int_or_none(last.group(1)) if last is not None else None


def parse_listing(html):
    """IIS/nginx/apache directory HTML -> [RemoteFile], newest run first.

    Untrusted input (v2.md 6.2.3): every candidate must satisfy NAME_RE, and the name that
    ends up on disk is rebuilt from the captures rather than echoed back.
    """
    text = html if isinstance(html, str) else html.decode('utf8', 'replace')
    found = {}
    for match in _ANCHOR_RE.finditer(text):
        href, inner = match.group(1), match.group(2)
        # The anchor text is the friendlier source, but IIS truncates long names with an
        # ellipsis, so fall back to the href's last segment.
        for candidate in (re.sub(r'<[^>]*>', '', inner).strip(),
                          href.rstrip('/').rpartition('/')[2]):
            name_match = NAME_RE.match(candidate)
            if name_match:
                break
        else:
            continue
        before = text[max(0, match.start() - 90):match.start()]
        after = text[match.end():match.end() + 90]
        size = _size_before(before)
        if size is None:
            size = _size_after(after)
        entry = RemoteFile(candidate, name_match.group('run'), name_match.group('field'),
                           size)
        # A listing can name the same file twice (anchor text and href); keep the richer.
        previous = found.get(candidate)
        if previous is None or (previous.size is None and size is not None):
            found[candidate] = entry
    return sorted(found.values(), key=lambda f: (f.run, f.field), reverse=True)


def fetch_listing(session, url=BASE):
    """Download and parse the ensemble directory index."""
    try:
        response = session.get(url, timeout=TIMEOUT)
    except Exception as exc:
        raise DownloadError(f'Could not reach the IMS server: {exc}') from exc
    _raise_for_auth(response)
    if getattr(response, 'status_code', 200) >= 400:
        raise DownloadError(f'The server returned HTTP {response.status_code} '
                            'for the ensemble directory.')
    files = parse_listing(response.text)
    if not files:
        raise DownloadError('The ensemble directory listing held no '
                            'ICON_ENS_<run>_<FIELD>.nc.bz2 entries.')
    return files


def runs_in(files):
    """Run ids present in a listing, newest first."""
    return sorted({f.run for f in files}, reverse=True)


def index_by_run(files):
    """-> {run: {field: RemoteFile}}."""
    out = {}
    for entry in files:
        out.setdefault(entry.run, {})[entry.field] = entry
    return out


def _raise_for_auth(response):
    if getattr(response, 'status_code', 200) in (401, 403):
        # Never echo the username, the password, or the URL that was tried.
        raise AuthError('Authentication failed. Check the IMS user name and password '
                        '(the server uses NTLM, not a browser login).')


# ---- the transfer ----------------------------------------------------------------------
def part_path(target):
    return Path(target).with_name(Path(target).name + '.part')


def _total_from(response, already):
    """Total size of the whole file, from Content-Range if the server honoured the range."""
    headers = getattr(response, 'headers', {}) or {}
    content_range = headers.get('Content-Range') or headers.get('content-range')
    if content_range:
        match = re.search(r'/\s*(\d+)\s*$', content_range)
        if match:
            return int(match.group(1))
    length = headers.get('Content-Length') or headers.get('content-length')
    try:
        return int(length) + already
    except (TypeError, ValueError):
        return None


def download(session, url, target, expected_size=None, progress=None, cancel=None):
    """Resumable GET into `<target>.part`, then one atomic rename. -> Path.

    Resume is the point: 262 MB over a flaky link should not restart from zero, so a
    cancelled or failed transfer deliberately LEAVES the `.part` file and the next call
    continues it with a `Range` header. Only a completed, size-checked transfer is
    renamed into place, so a `.nc.bz2` in the download directory is always whole.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        if progress is not None:
            size = target.stat().st_size
            progress(size, size)
        return target

    partial = part_path(target)
    have = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and have > expected_size:
        have = 0                              # a stale .part from a different file
        partial.unlink(missing_ok=True)
    if expected_size is not None:
        free = shutil.disk_usage(target.parent).free
        if free < MIN_FREE_BYTES + max(0, expected_size - have):
            raise DownloadError(
                f'Not enough free space in {target.parent}: the download needs about '
                f'{(expected_size - have) >> 20} MB plus a 1 GB margin.')

    headers = {'Range': f'bytes={have}-'} if have else {}
    try:
        response = session.get(url, headers=headers, stream=True, timeout=TIMEOUT)
    except Exception as exc:
        raise DownloadError(f'Download failed: {exc}') from exc
    _raise_for_auth(response)
    status = getattr(response, 'status_code', 200)
    if status == 416:                         # range past the end: the .part is complete
        partial.replace(target)
        return _verify(target, expected_size, progress)
    if status >= 400:
        raise DownloadError(f'The server returned HTTP {status} for this file.')
    if have and status != 206:
        # The server ignored the Range header, so what follows is the whole file again.
        have = 0
    mode = 'ab' if have else 'wb'
    total = _total_from(response, have) or expected_size

    written = have
    try:
        with open(partial, mode) as fh:
            for block in response.iter_content(CHUNK):
                if cancel is not None and cancel():
                    raise Cancelled
                if not block:
                    continue
                fh.write(block)
                written += len(block)
                if progress is not None:
                    progress(written, total or 0)
    except Cancelled:
        raise                                 # keep the .part: that is what makes it resume
    except Exception as exc:
        raise DownloadError(f'Download failed after {written >> 20} MB: {exc}') from exc

    if expected_size is not None and written != expected_size:
        # Do not rename a short file into place: nc3 would open it as a truncated dataset
        # (G26) and quietly show a partial forecast.
        raise DownloadError(
            f'{target.name} came back as {written:,} bytes but the listing says '
            f'{expected_size:,}. The partial file is kept, so retrying will resume it.')
    partial.replace(target)
    return _verify(target, expected_size, progress)


def _verify(target, expected_size, progress):
    size = target.stat().st_size
    if expected_size is not None and size != expected_size:
        raise DownloadError(f'{target.name} is {size:,} bytes, expected {expected_size:,}.')
    if progress is not None:
        progress(size, size)
    return target


def resume_state(target):
    """Bytes already fetched for `target`, for a dialog that wants to say "resume"."""
    partial = part_path(target)
    return partial.stat().st_size if partial.exists() else 0
