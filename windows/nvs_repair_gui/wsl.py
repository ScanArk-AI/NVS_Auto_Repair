"""Talk to WSL2 from Windows: run commands in the Ubuntu distro and map paths.

Everything that can be is a plain function of strings, so it is tested on any
OS; only ``WSL.run`` / ``WSL.capture`` actually start ``wsl.exe``.
"""

from __future__ import annotations

import os
import posixpath
import re
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Callable

from .steps import HomePath, Layout

DISTRO = "Ubuntu-24.04"
# The background job's process-group id, so "Stop" can end the whole tree
# (python, its CUDA children) rather than just the wsl.exe that started it.
# One file per user: install steps run as root and as the user, and /tmp's
# sticky bit would stop the user overwriting a file root created.
PIDFILE = "/tmp/nvs_repair_gui.$(id -u).pid"
PIDGLOB = "/tmp/nvs_repair_gui.*.pid"


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def win_to_wsl(path: str) -> str:
    r"""``C:\data\scene`` -> ``/mnt/c/data/scene``; ``\\wsl.localhost\<distro>\home\u`` -> ``/home/u``.

    Paths that are already POSIX are returned unchanged.
    """
    path = str(path).strip()
    if path.startswith("/"):
        return path
    m = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)?$", path, re.IGNORECASE)
    if m:
        return (m.group(1) or "\\").replace("\\", "/")
    p = PureWindowsPath(path)
    if not p.drive or not re.fullmatch(r"[A-Za-z]:", p.drive):
        raise ValueError(f"not a drive path: {path!r}")
    rest = "/".join(p.parts[1:])
    return f"/mnt/{p.drive[0].lower()}" + (f"/{rest}" if rest else "")


# Where bootstrap.sh / setup.py install things inside WSL (its defaults).
WSL_LAYOUT = Layout("wsl", HomePath("nvs_repair"), HomePath("nvs_repair_runtime/envs/nvs/bin/python"),
                    HomePath("nvs_repair_scenes"), lambda p: win_to_wsl(p).rstrip("/") or "/", posixpath)


def wsl_to_win(path: str, distro: str = DISTRO) -> str:
    r"""``/mnt/c/data`` -> ``C:\data``; ``/home/u/x`` -> ``\\wsl.localhost\<distro>\home\u\x``."""
    m = re.match(r"^/mnt/([a-zA-Z])(/.*)?$", path)
    if m:
        return f"{m.group(1).upper()}:" + ((m.group(2) or "/").replace("/", "\\"))
    return f"\\\\wsl.localhost\\{distro}" + path.replace("/", "\\")


def decode(raw: bytes) -> str:
    """``wsl.exe``'s own messages are UTF-16LE; what runs inside Linux prints UTF-8."""
    if raw[:2] == b"\xff\xfe" or (len(raw) > 1 and raw[1:2] == b"\x00"):
        return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def job_script(script: str) -> str:
    """Run ``script`` as its own process group and remember it for ``stop``."""
    return f"set -m; ( {script} ) & echo $! > {PIDFILE}; set +m; wait $!"


@dataclass
class WSL:
    layout = WSL_LAYOUT          # not a field: where runner finds the repo and python
    distro: str = DISTRO
    exe: str = "wsl.exe"
    popen: Callable = subprocess.Popen
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def argv(self, script: str, root: bool = False) -> list[str]:
        return [self.exe, "-d", self.distro, *(["-u", "root"] if root else []), "--exec", "bash", "-lc", script]

    def _flags(self) -> int:
        # No console window flashing up behind the GUI on Windows.
        return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0

    def meta(self, *args: str, timeout: float = 60) -> tuple[int, str]:
        """``wsl.exe <args>`` itself (``--status``, ``-l -q``); its output is UTF-16."""
        try:
            proc = self.popen([self.exe, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, creationflags=self._flags(),
                              env={**os.environ, "WSL_UTF8": "1"})
            out, _ = proc.communicate(timeout=timeout)
            return proc.returncode, decode(out or b"")
        except FileNotFoundError:
            return 127, f"{self.exe} not found"
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124, "timed out"

    def capture(self, script: str, root: bool = False, timeout: float = 120) -> tuple[int, str]:
        """Run a short command; ``(returncode, output)``."""
        try:
            proc = self.popen(self.argv(script, root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              stdin=subprocess.DEVNULL, creationflags=self._flags(),
                              env={**os.environ, "WSL_UTF8": "1"})
            out, err = proc.communicate(timeout=timeout)
            # Successful probes consume stdout as data, not WSL startup warnings.
            data = out or b""
            if proc.returncode != 0:
                data += err or b""
            return proc.returncode, decode(data)
        except FileNotFoundError:
            return 127, f"{self.exe} not found -- is WSL installed?"
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124, f"timed out after {timeout:.0f} s"

    def run(self, script: str, on_line: Callable[[str], None], root: bool = False) -> int:
        """Run a long job, streaming each output line to ``on_line``; returns its exit code."""
        try:
            proc = self.popen(self.argv(job_script(script), root), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=self._flags(),
                              env={**os.environ, "WSL_UTF8": "1"})
        except FileNotFoundError:
            on_line(f"{self.exe} not found -- is WSL installed?")
            return 127
        with self._lock:
            self._proc = proc
        buf = b""
        for chunk in iter(lambda: proc.stdout.read1(4096) if hasattr(proc.stdout, "read1")
                          else proc.stdout.read(4096), b""):
            buf += chunk
            # progress bars redraw with \r; treat it as a line break
            *lines, buf = re.split(rb"\r\n|\n|\r", buf)
            for line in lines:
                on_line(decode(line))
        if buf:
            on_line(decode(buf))
        rc = proc.wait()
        with self._lock:
            self._proc = None
        return rc

    def stop(self) -> None:
        """End the running job's whole process group inside WSL."""
        self.capture(f'for f in {PIDGLOB}; do test -s "$f" && kill -TERM -- -"$(cat "$f")" 2>/dev/null; done; '
                     f'sleep 3; for f in {PIDGLOB}; do test -s "$f" && kill -KILL -- -"$(cat "$f")" 2>/dev/null; '
                     f'rm -f "$f"; done; true', root=True, timeout=30)
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()


def q(value) -> str:
    """Shell-quote for a ``bash -lc`` script."""
    return shlex.quote(str(value))
