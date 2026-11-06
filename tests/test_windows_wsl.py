"""windows/nvs_repair_gui/wsl.py: path mapping, decoding, command lines, streaming."""

import io
import subprocess

import pytest

from nvs_repair_gui import wsl as W


@pytest.mark.parametrize("win, posix", [
    (r"C:\data\my scene", "/mnt/c/data/my scene"),
    ("D:/x/y", "/mnt/d/x/y"),
    ("c:\\", "/mnt/c"),
    (r"\\wsl.localhost\Ubuntu-24.04\home\u\scene", "/home/u/scene"),
    (r"\\wsl$\Ubuntu-24.04\home", "/home"),
    ("/home/u/already", "/home/u/already"),
])
def test_win_to_wsl(win, posix):
    assert W.win_to_wsl(win) == posix


@pytest.mark.parametrize("bad", ["relative\\dir", r"\\server\share\x", ""])
def test_win_to_wsl_rejects_non_drive_paths(bad):
    with pytest.raises(ValueError):
        W.win_to_wsl(bad)


def test_wsl_to_win_round_trip():
    assert W.wsl_to_win("/mnt/c/data/my scene") == r"C:\data\my scene"
    assert W.wsl_to_win("/home/u/x", "Ubuntu-24.04") == r"\\wsl.localhost\Ubuntu-24.04\home\u\x"
    for win in (r"C:\data\x", r"E:\a b\c"):
        assert W.wsl_to_win(W.win_to_wsl(win)) == win


def test_decode_utf16_and_utf8():
    assert W.decode("Ubuntu-24.04\r\n".encode("utf-16-le")) == "Ubuntu-24.04\r\n"
    assert W.decode(b"\xff\xfe" + "Ubuntu".encode("utf-16-le")) == "Ubuntu"
    assert W.decode("ünicode ok".encode()) == "ünicode ok"


def test_argv_and_job_script():
    w = W.WSL()
    assert w.argv("echo hi") == ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "bash", "-lc", "echo hi"]
    assert w.argv("id", root=True)[3:5] == ["-u", "root"]
    job = W.job_script("python run.py all")
    assert job.startswith("set -m;") and W.PIDFILE in job and "python run.py all" in job
    assert W.q("a b'c") == "'a b'\"'\"'c'"


class _FakeProc:
    def __init__(self, data: bytes, rc: int = 0):
        self.stdout = io.BytesIO(data)
        self.returncode = rc

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode

    def communicate(self, timeout=None):
        return self.stdout.read(), b""


def test_run_streams_lines_and_treats_cr_as_newline():
    calls = []

    def popen(argv, **kw):
        calls.append(argv)
        return _FakeProc(b"step 1\r\nprogress 10%\rprogress 50%\nlast line without newline", rc=3)

    lines = []
    rc = W.WSL(popen=popen).run("bash ./bootstrap.sh", lines.append)
    assert rc == 3
    assert lines == ["step 1", "progress 10%", "progress 50%", "last line without newline"]
    assert calls[0][-1].startswith("set -m;")             # wrapped so stop() can kill the group


def test_missing_wsl_exe():
    def popen(argv, **kw):
        raise FileNotFoundError(argv[0])

    w = W.WSL(popen=popen)
    lines = []
    assert w.run("true", lines.append) == 127 and "not found" in lines[0]
    assert w.capture("true")[0] == 127
    assert w.meta("--status")[0] == 127


def test_capture_ignores_startup_warnings_and_requests_utf8():
    calls = []

    class WarningProc(_FakeProc):
        def communicate(self, timeout=None):
            return b"NVIDIA RTX 5060 Ti, 16311 MiB\n", "wsl: proxy warning\n".encode()

    def popen(argv, **kw):
        calls.append(kw)
        return WarningProc(b"")

    w = W.WSL(popen=popen)
    assert w.capture("nvidia-smi") == (0, "NVIDIA RTX 5060 Ti, 16311 MiB\n")
    assert calls[0]["stderr"] == subprocess.PIPE
    assert calls[0]["env"]["WSL_UTF8"] == "1"
    w.meta("--status")
    w.run("true", lambda line: None)
    assert all(call["env"]["WSL_UTF8"] == "1" for call in calls)


def test_capture_preserves_error_diagnostics():
    class ErrorProc(_FakeProc):
        def communicate(self, timeout=None):
            return b"partial output\n", b"startup failed\n"

    w = W.WSL(popen=lambda *args, **kw: ErrorProc(b"", rc=1))
    assert w.capture("false") == (1, "partial output\nstartup failed\n")


def test_capture_and_meta_decode():
    w = W.WSL(popen=lambda argv, **kw: _FakeProc("Ubuntu-24.04\r\n".encode("utf-16-le")))
    assert w.meta("-l", "-q") == (0, "Ubuntu-24.04\r\n")
    w = W.WSL(popen=lambda argv, **kw: _FakeProc(b"NVIDIA RTX 4090, 24564 MiB\n"))
    assert w.capture("nvidia-smi") == (0, "NVIDIA RTX 4090, 24564 MiB\n")
