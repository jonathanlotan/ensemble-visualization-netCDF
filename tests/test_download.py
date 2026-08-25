"""The built-in downloader: listing an untrusted index, and resuming a transfer.

No network. A fake session serves bytes with real `Range` semantics, which is what the
resume path actually depends on -- a test that mocks `download()` itself would prove
nothing about the one behaviour worth having.
"""
import pytest

from imsicon import download, ingest

IIS = """<html><head><title>data.israel-meteo-service.org - /ims/IMS_ICON_ENSEMBLE/</title>
</head><body><H1>/ims/IMS_ICON_ENSEMBLE/</H1><hr>
<pre><A HREF="/ims/">[To Parent Directory]</A><br><br>
 8/23/2026 12:20 PM    274989056 <A HREF="/ims/IMS_ICON_ENSEMBLE/ICON_ENS_2026082300_CAPE_ML.nc.bz2">ICON_ENS_2026082300_CAPE_ML.nc.bz2</A><br>
 8/23/2026 12:22 PM    198765432 <A HREF="/ims/IMS_ICON_ENSEMBLE/ICON_ENS_2026082300_T_2M.nc.bz2">ICON_ENS_2026082300_T_2M.nc.bz2</A><br>
 8/23/2026 12:25 PM      125,788 <A HREF="/ims/IMS_ICON_ENSEMBLE/ICON_ENS_2026082300_RELHUM_2M.nc.bz2">ICON_ENS_2026082300_RELHUM_2M.nc.bz2</A><br>
 8/22/2026 12:21 PM    274000000 <A HREF="/ims/IMS_ICON_ENSEMBLE/ICON_ENS_2026082200_CAPE_ML.nc.bz2">ICON_ENS_2026082200_CAPE_ML.nc.bz2</A><br>
 8/23/2026 12:30 PM         4096 <A HREF="/ims/IMS_ICON_ENSEMBLE/readme.txt">readme.txt</A><br>
 8/23/2026 12:30 PM         4096 <A HREF="/ims/IMS_ICON_ENSEMBLE/../../etc/passwd">../../etc/passwd</A><br>
</pre><hr></body></html>"""

NGINX = """<html><head><title>Index of /ims/IMS_ICON_ENSEMBLE/</title></head><body>
<h1>Index of /ims/IMS_ICON_ENSEMBLE/</h1><hr><pre><a href="../">../</a>
<a href="ICON_ENS_2026082400_TOT_PREC.nc.bz2">ICON_ENS_2026082400_TOT_PREC.nc.bz2</a>   24-Aug-2026 12:20   261123456
<a href="ICON_ENS_2026082400_CLCT.nc.bz2">ICON_ENS_2026082400_CLCT.nc.bz2</a>        24-Aug-2026 12:21           -
</pre><hr></body></html>"""


# ---- the listing, treated as untrusted input -------------------------------------------
def test_iis_listing_yields_run_field_and_size():
    files = {f.name: f for f in download.parse_listing(IIS)}
    assert len(files) == 4
    cape = files['ICON_ENS_2026082300_CAPE_ML.nc.bz2']
    assert (cape.run, cape.field, cape.size) == ('2026082300', 'CAPE_ML', 274989056)
    assert files['ICON_ENS_2026082300_RELHUM_2M.nc.bz2'].size == 125788   # thousands sep


def test_listing_ignores_everything_that_is_not_an_ensemble_file():
    names = {f.name for f in download.parse_listing(IIS)}
    assert not any('passwd' in name or 'readme' in name for name in names)


def test_a_local_name_is_rebuilt_from_the_validated_captures_not_echoed():
    """G27: a server string must never be able to choose where a download lands."""
    entry = download.RemoteFile('../../evil.nc.bz2', '2026082300', 'CAPE_ML', 1)
    assert entry.local_name() == 'ICON_ENS_2026082300_CAPE_ML.nc.bz2'
    assert '/' not in entry.local_name() and '..' not in entry.local_name()


def test_nginx_style_listing_reads_the_trailing_size():
    files = {f.field: f for f in download.parse_listing(NGINX)}
    assert files['TOT_PREC'].size == 261123456
    # A timestamp on the same row must not be mistaken for a size, and a row with no size
    # must not inherit the one above it.
    assert files['CLCT'].size is None


def test_runs_are_newest_first_and_indexable():
    files = download.parse_listing(IIS)
    assert download.runs_in(files) == ['2026082300', '2026082200']
    index = download.index_by_run(files)
    assert set(index['2026082300']) == {'CAPE_ML', 'T_2M', 'RELHUM_2M'}


def test_the_catalogue_covers_every_ims_field_and_names_the_dew_point_inputs():
    """The download menu and the sniffer must agree on what the server publishes."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
    import sniff_headers

    assert {p.field for p in download.PRODUCTS} == set(sniff_headers.FIELDS)
    assert set(download.DEW_POINT_FIELDS) <= {p.field for p in download.PRODUCTS}
    assert download.DEW_POINT_FIELDS == derived_inputs()


def derived_inputs():
    from imsicon import derived
    return tuple(field for field, _units in derived.DEW_POINT_INPUTS.values())


# ---- a fake server ----------------------------------------------------------------------
class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self.content = body
        self.status_code = status
        self.headers = headers or {}
        self.text = body.decode('utf8', 'replace') if isinstance(body, bytes) else body

    def iter_content(self, chunk):
        for i in range(0, len(self.content), chunk):
            yield self.content[i:i + chunk]


class FakeSession:
    """Serves one payload with real Range semantics, and counts bytes actually sent."""

    def __init__(self, payload, status=200, honour_range=True, cut_after=None):
        self.payload = payload
        self.status = status
        self.honour_range = honour_range
        self.cut_after = cut_after          # bytes to send before "the link drops"
        self.sent = 0
        self.requests = []

    def get(self, url, headers=None, stream=False, timeout=None):
        headers = headers or {}
        self.requests.append(headers.get('Range'))
        if self.status != 200:
            return FakeResponse(b'', status=self.status)
        start = 0
        if self.honour_range and headers.get('Range'):
            start = int(headers['Range'].split('=')[1].split('-')[0])
        if start >= len(self.payload):
            return FakeResponse(b'', status=416)
        body = self.payload[start:]
        if self.cut_after is not None:
            body = body[:self.cut_after]
        self.sent += len(body)
        if start:
            head = {'Content-Range': f'bytes {start}-{len(self.payload) - 1}/'
                                     f'{len(self.payload)}'}
            return FakeResponse(body, status=206, headers=head)
        return FakeResponse(body, headers={'Content-Length': str(len(self.payload))})


PAYLOAD = bytes(range(256)) * 400            # 102,400 bytes


def test_a_whole_file_downloads_and_is_renamed_into_place(tmp_path):
    session = FakeSession(PAYLOAD)
    target = tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc.bz2'
    out = download.download(session, 'https://x/f', target, expected_size=len(PAYLOAD))
    assert out == target and target.read_bytes() == PAYLOAD
    assert not download.part_path(target).exists()


def test_an_interrupted_download_keeps_its_part_file_and_resumes(tmp_path):
    """The whole point of the resumable path: 262 MB must not restart from zero."""
    target = tmp_path / 'ICON_ENS_2026082300_CAPE_ML.nc.bz2'
    stopped = FakeSession(PAYLOAD, cut_after=40_000)
    with pytest.raises(download.DownloadError):
        download.download(stopped, 'https://x/f', target, expected_size=len(PAYLOAD))
    assert not target.exists()
    partial = download.part_path(target)
    assert partial.stat().st_size == 40_000
    assert download.resume_state(target) == 40_000

    rest = FakeSession(PAYLOAD)
    out = download.download(rest, 'https://x/f', target, expected_size=len(PAYLOAD))
    assert out.read_bytes() == PAYLOAD
    assert rest.requests == ['bytes=40000-']          # asked to continue, not restart
    assert rest.sent == len(PAYLOAD) - 40_000         # and only the remainder crossed


def test_a_server_that_ignores_the_range_header_restarts_cleanly(tmp_path):
    target = tmp_path / 'f.nc.bz2'
    download.part_path(target).write_bytes(b'\x00' * 1000)
    session = FakeSession(PAYLOAD, honour_range=False)
    out = download.download(session, 'https://x/f', target, expected_size=len(PAYLOAD))
    # Appending to the stale 1000 bytes would corrupt the file with no error anywhere.
    assert out.read_bytes() == PAYLOAD


def test_a_short_file_is_refused_rather_than_renamed(tmp_path):
    """A truncated .nc opens as a partial forecast (G26), so it must never land."""
    target = tmp_path / 'f.nc.bz2'
    session = FakeSession(PAYLOAD[:-10])
    with pytest.raises(download.DownloadError, match='listing says'):
        download.download(session, 'https://x/f', target, expected_size=len(PAYLOAD))
    assert not target.exists() and download.part_path(target).exists()


def test_a_complete_part_file_is_finished_by_a_416(tmp_path):
    target = tmp_path / 'f.nc.bz2'
    download.part_path(target).write_bytes(PAYLOAD)
    session = FakeSession(PAYLOAD)
    out = download.download(session, 'https://x/f', target, expected_size=len(PAYLOAD))
    assert out.read_bytes() == PAYLOAD


def test_an_already_complete_file_is_not_refetched(tmp_path):
    target = tmp_path / 'f.nc.bz2'
    target.write_bytes(PAYLOAD)
    session = FakeSession(PAYLOAD)
    download.download(session, 'https://x/f', target, expected_size=len(PAYLOAD))
    assert session.requests == []


def test_cancelling_raises_cancelled_and_keeps_the_partial(tmp_path):
    target = tmp_path / 'f.nc.bz2'
    with pytest.raises(download.Cancelled):
        download.download(FakeSession(PAYLOAD), 'https://x/f', target,
                          expected_size=len(PAYLOAD), cancel=lambda: True)
    assert not target.exists()


def test_progress_reaches_the_full_size(tmp_path):
    seen = []
    download.download(FakeSession(PAYLOAD), 'https://x/f', tmp_path / 'f.nc.bz2',
                      expected_size=len(PAYLOAD),
                      progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))


# ---- credentials ------------------------------------------------------------------------
def test_authentication_failure_never_echoes_what_was_tried(tmp_path):
    session = FakeSession(PAYLOAD, status=401)
    with pytest.raises(download.AuthError) as excinfo:
        download.download(session, 'https://user:secret@x/f', tmp_path / 'f.nc.bz2')
    message = str(excinfo.value)
    assert 'secret' not in message and 'https://' not in message
    assert 'NTLM' in message


def test_a_forbidden_listing_is_an_auth_error_not_a_parse_error():
    with pytest.raises(download.AuthError):
        download.fetch_listing(FakeSession(b'', status=403))


def test_credentials_come_from_the_environment_not_from_qsettings(monkeypatch):
    monkeypatch.setenv('IMS_USER', 'someone')
    monkeypatch.setenv('IMS_PASS', 'a-password')
    assert download.stored_credentials() == ('someone', 'a-password')

    from PySide6 import QtCore
    settings = QtCore.QSettings('IMS', 'IconEnsembleViewer')
    blob = ' '.join(str(settings.value(key)) for key in settings.allKeys())
    assert 'a-password' not in blob        # a password on disk in plaintext is the bug


def test_half_a_credential_is_no_credential(monkeypatch):
    monkeypatch.setenv('IMS_USER', 'someone')
    monkeypatch.delenv('IMS_PASS', raising=False)
    monkeypatch.setattr(download, 'keyring_module', lambda: None)
    assert download.stored_credentials() is None


def test_a_missing_http_stack_says_what_to_install(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_requests(name, *args, **kwargs):
        if name in ('requests', 'requests_ntlm'):
            raise ImportError(f'no module named {name}')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_requests)
    with pytest.raises(download.MissingDependency, match='requests-ntlm'):
        download.make_session('u', 'p')


# ---- where downloads land ---------------------------------------------------------------
def test_downloads_have_their_own_directory_beside_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, 'cache_dir', lambda: tmp_path / 'cache')
    (tmp_path / 'cache').mkdir()
    assert ingest.download_dir() == tmp_path / 'downloads'
    assert ingest.download_dir().is_dir()


def test_download_eviction_spares_part_files(monkeypatch, tmp_path):
    """An interrupted transfer is meant to resume; evicting its .part defeats that."""
    downloads = tmp_path / 'downloads'
    downloads.mkdir()
    monkeypatch.setattr(ingest, 'download_dir', lambda: downloads)
    (downloads / 'old.nc.bz2').write_bytes(b'x' * 4000)
    (downloads / 'big.nc.bz2.part').write_bytes(b'y' * 4000)
    ingest.evict_downloads(budget=100)
    assert not (downloads / 'old.nc.bz2').exists()
    assert (downloads / 'big.nc.bz2.part').exists()
