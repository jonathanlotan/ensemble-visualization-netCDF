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
* **The listing is UNTRUSTED input.** Only a name matching the family's grammar --
  `ICON_ENS_<10 digits>_<FIELD>.nc.bz2`, or `IE_<10 digits>_<field>.nc.bz2` for the
  deterministic run -- is accepted, by a strict regex; everything else in the HTML is
  ignored, and a local path is never built from a server string without rebuilding it from
  the validated captures (G27).

Two families are offered (see `products.py`): the **ensemble**, whose second data axis is
20 members, and the **deterministic ICON-LAM run**, six of whose fields are 3-D on 20
pressure levels. They differ only in the folder, the name grammar and the catalogue, so
everything below takes a `family` and nothing below knows which one it is holding.
"""
import os
import re
import shutil
from pathlib import Path

from . import products
from .products import ENSEMBLE, ICON, FAMILIES     # noqa: F401  (re-exported)

BASE = ENSEMBLE.base_url
# The deterministic folder is inferred rather than measured (products.py). This env var
# points the app at the right one without a code change if the inference is wrong.
ICON_URL_ENV = 'IMS_ICON_URL'
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
# One file per field per run (CLAUDE.md 0.2, and Table 1 of IMS_ICON_manual.pdf). Both
# catalogues live in `products.py`, because `ingest.py` needs the same naming rules to
# find those files again on disk and two copies would drift.
Product = products.Product


# The ensemble catalogue, under the names the rest of the app has always imported.
PRODUCTS = ENSEMBLE.products
BY_FIELD = ENSEMBLE.by_field
GROUPS = ENSEMBLE.groups

# The two fields the dew point is derived from (derived.dew_point).
DEW_POINT_FIELDS = ('T_2M', 'RELHUM_2M')
# ...and the two the wind map is (derived.wind). Both pairs are useless one file at a
# time, which is why the dialog offers a one-click tick for each.
WIND_FIELDS = ('U_10M', 'V_10M')


def role_fields(family, roles):
    """The fields playing `roles` in `family`, or () when it does not have them all."""
    fields = tuple(family.roles.get(role) for role in roles)
    return fields if all(fields) else ()


def dew_point_fields(family=ENSEMBLE):
    return role_fields(family, ('temperature', 'humidity'))


def wind_fields(family=ENSEMBLE):
    return role_fields(family, ('zonal', 'meridional'))


def base_url(family=ENSEMBLE):
    """Where a family's files are listed. `IMS_ICON_URL` overrides the deterministic one.

    The ensemble folder is measured (CLAUDE.md 0.1); the deterministic folder is inferred
    from the server's layout and could not be confirmed without credentials, so it is the
    one that can be pointed elsewhere without a new build.
    """
    if family is ICON:
        override = (os.environ.get(ICON_URL_ENV) or '').strip()
        if override:
            return override if override.endswith('/') else override + '/'
    return family.base_url


def product_label(field, family=ENSEMBLE):
    return family.label_for(field)


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
NAME_RE = ENSEMBLE.name_re
_ANCHOR_RE = re.compile(r'<a\b[^>]*?href\s*=\s*["\']([^"\']*)["\'][^>]*>(.*?)</a>',
                        re.IGNORECASE | re.DOTALL)
# A size is a standalone integer. The lookarounds exclude the digits of the timestamp that
# sits next to it on the same line -- without them "24-Aug-2026 12:20 261123456" parses as
# a 12-byte file, which is exactly the kind of plausible-looking wrong number that makes a
# size check useless.
_SIZE_TOKEN = re.compile(r'(?<![\d.,:/\-])(\d[\d,]*)(?![\d.,:/\-])')


class RemoteFile:
    """One `ICON_ENS_<run>_<FIELD>.nc.bz2` / `IE_<run>_<field>.nc.bz2` on the server."""
    __slots__ = ('name', 'run', 'field', 'size', 'family')

    def __init__(self, name, run, field, size=None, family=ENSEMBLE):
        self.name, self.run, self.field, self.size = name, run, field, size
        self.family = family

    @property
    def url(self):
        return base_url(self.family) + self.name

    @property
    def label(self):
        return self.family.label_for(self.field)

    @property
    def levels(self):
        """True for a field the catalogue says is 3-D on pressure levels."""
        return self.family.has_levels(self.field)

    def local_name(self):
        """G27: the file name used on disk is re-derived, never taken from the server.

        `Path(name).name` alone is not enough -- it still passes through `..%2f` style
        oddities on some platforms -- so the name is REBUILT from the two validated
        capture groups. A server string can therefore never choose a path.
        """
        return self.family.local_name(self.run, self.field)

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


def parse_listing(html, family=ENSEMBLE):
    """IIS/nginx/apache directory HTML -> [RemoteFile], newest run first.

    Untrusted input (v2.md 6.2.3): every candidate must satisfy the family's name grammar,
    and the name that ends up on disk is rebuilt from the captures rather than echoed back.
    """
    text = html if isinstance(html, str) else html.decode('utf8', 'replace')
    found = {}
    for match in _ANCHOR_RE.finditer(text):
        href, inner = match.group(1), match.group(2)
        # The anchor text is the friendlier source, but IIS truncates long names with an
        # ellipsis, so fall back to the href's last segment.
        for candidate in (re.sub(r'<[^>]*>', '', inner).strip(),
                          href.rstrip('/').rpartition('/')[2]):
            name_match = family.name_re.match(candidate)
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
                           size, family)
        # A listing can name the same file twice (anchor text and href); keep the richer.
        previous = found.get(candidate)
        if previous is None or (previous.size is None and size is not None):
            found[candidate] = entry
    return sorted(found.values(), key=lambda f: (f.run, f.field), reverse=True)


def fetch_listing(session, url=None, family=ENSEMBLE):
    """Download and parse one family's directory index."""
    url = base_url(family) if url is None else url
    try:
        response = session.get(url, timeout=TIMEOUT)
    except Exception as exc:
        raise DownloadError(f'Could not reach the IMS server: {exc}') from exc
    _raise_for_auth(response)
    if getattr(response, 'status_code', 200) >= 400:
        raise DownloadError(f'The server returned HTTP {response.status_code} for '
                            f'{url} - the {family.short} directory.')
    files = parse_listing(response.text, family)
    if not files:
        raise DownloadError(
            f'{url} held no {family.prefix}<run>_<field>.nc.bz2 entries.'
            + (f' The deterministic folder is inferred from the server layout rather than '
               f'measured: set {ICON_URL_ENV} to its real address if this is the wrong one.'
               if family is ICON else ''))
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
