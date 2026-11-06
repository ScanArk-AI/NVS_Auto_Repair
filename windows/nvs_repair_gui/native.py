"""Run the pipeline's steps directly on Windows with the bundle's own Python.  Owner: C.

The native counterpart of ``wsl.WSL``: an ``Executor`` (see steps.py) that
starts each ``Step``'s argv as a process, streams its output line by line, and
on "Stop" ends the whole process tree (the stage's Python and the processes it
spawns). Works on any OS so it can be tested anywhere; on Windows it hides the
console windows and uses ``taskkill /T``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from .steps import Layout, Step

WINDOWS = sys.platform == "win32"


def base_env(layout: Layout, environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every step starts from.

    UTF-8 mode and unbuffered output so the log arrives live and non-ASCII
    names survive; the bundle's Python (and its Scripts, where ninja.exe would
    live) first on PATH; no user site-packages, so a Python the user installed
    separately cannot leak modules into the bundle's.
    """
    env = dict(os.environ if environ is None else environ)
    pydir = str(Path(layout.python).parent)
    env["PATH"] = os.pathsep.join([pydir, str(Path(pydir) / "Scripts"), env.get("PATH", "")])
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


class Native:
    def __init__(self, layout: Layout, popen: Callable = subprocess.Popen, killer: Callable | None = None):
        self.layout = layout
        self._popen = popen
        self._killer = killer          # injected in tests; None = the real platform call
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # -- process plumbing ----------------------------------------------------
    def _spawn(self, argv: list[str], cwd: str | None, env: dict[str, str]) -> subprocess.Popen:
        kwargs = dict(cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL)
        if WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True     # own process group, for stop()
        return self._popen([str(a) for a in argv], **kwargs)

    def capture(self, argv: list[str], timeout: float = 120, env: dict[str, str] | None = None) -> tuple[int, str]:
        """A short command; ``(returncode, output)``."""
        try:
            proc = self._spawn(argv, self.layout.repo if Path(self.layout.repo).is_dir() else None,
                               {**base_env(self.layout), **(env or {})})
            out, _ = proc.communicate(timeout=timeout)
            return proc.returncode, (out or b"").decode("utf-8", errors="replace")
        except FileNotFoundError as exc:
            return 127, f"not found: {exc.filename or argv[0]}"
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124, f"timed out after {timeout:.0f} s"

    # -- Executor ------------------------------------------------------------
    def run_step(self, step: Step, on_line: Callable[[str], None]) -> int:
        env = {**base_env(self.layout), **step.env}
        try:
            proc = self._spawn(step.argv, step.cwd or self.layout.repo, env)
        except FileNotFoundError as exc:
            on_line(f"not found: {exc.filename or step.argv[0]}")
            return 127
        with self._lock:
            self._proc = proc
        buf = b""
        read = proc.stdout.read1 if hasattr(proc.stdout, "read1") else proc.stdout.read
        for chunk in iter(lambda: read(4096), b""):
            buf += chunk
            *lines, buf = re.split(rb"\r\n|\n|\r", buf)     # \r: progress bars redraw in place
            for line in lines:
                on_line(line.decode("utf-8", errors="replace"))
        if buf:
            on_line(buf.decode("utf-8", errors="replace"))
        rc = proc.wait()
        with self._lock:
            self._proc = None
        return rc

    def stop(self) -> None:
        """End the running step's whole process tree."""
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if self._killer is not None:
            self._killer(proc.pid)
        elif WINDOWS:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
