"""What runner builds and a backend executes: ``Step`` and ``Layout``.  Owner: A.

A ``Step`` is one program to run -- argv, working directory, extra
environment -- with no shell in it.  A ``Layout`` says where a backend keeps
the repository and its Python, and how a Windows path the user typed is
written for that backend:

- WSL (``wsl.WSL_LAYOUT``): paths under the WSL user's $HOME, Windows paths
  mapped to /mnt/<drive>; each step is turned into one bash script by
  ``runner.to_bash``.
- native: a repository and a ``python.exe`` on Windows, paths used as typed;
  the native executor starts argv directly.

This module imports nothing from wsl or runner, so both can import it.
"""

from __future__ import annotations

import ntpath
import posixpath
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Protocol


class HomePath(str):
    """A path relative to the WSL user's home directory.

    Its string value is ``$HOME/<rel>`` (readable in logs); ``runner.to_bash``
    writes it as ``"$HOME"/'<rel>'`` so bash expands $HOME and nothing else.
    Only layouts create these -- never anything the user typed.
    """

    rel: str

    def __new__(cls, rel: str):
        rel = rel.strip("/")
        self = super().__new__(cls, f"$HOME/{rel}")
        self.rel = rel
        return self


@dataclass(frozen=True)
class Step:
    label: str                        # "copy" | "config" | "caption" | "pipeline"
    argv: list[str]                   # argv[0] is the program; items may be HomePath
    cwd: str | None = None            # None = the backend's default
    env: dict[str, str] = field(default_factory=dict)   # on top of the backend's base environment


@dataclass(frozen=True)
class Layout:
    name: str                         # "wsl" | "native"
    repo: str                         # the nvs_repair checkout (where run.py is)
    python: str                       # the interpreter with every stage installed
    scenes: str | None                # where copy_to_wsl puts scenes; None = this backend does not copy
    to_local: Callable[[str], str]    # a Windows path the user typed -> this backend's path; ValueError if unusable
    pathmod: ModuleType               # posixpath or ntpath

    def join(self, base: str, *parts: str) -> str:
        """``pathmod.join`` that keeps a HomePath a HomePath."""
        if isinstance(base, HomePath):
            return HomePath(posixpath.join(base.rel, *parts))
        return self.pathmod.join(base, *parts)

    def basename(self, path: str) -> str:
        seps = "/\\" if self.pathmod is ntpath else "/"
        return self.pathmod.basename(path.rstrip(seps))


class Executor(Protocol):
    layout: Layout

    def run_step(self, step: Step, on_line: Callable[[str], None]) -> int: ...

    def stop(self) -> None: ...


# --------------------------------------------------------------------------- #
# native layouts
# --------------------------------------------------------------------------- #
def windows_path(path: str) -> str:
    r"""An absolute Windows path, normalised (``C:/a/../b`` -> ``C:\b``); ValueError otherwise."""
    path = str(path).strip()
    drive, _ = ntpath.splitdrive(path)
    if not drive or not ntpath.isabs(path):
        raise ValueError(f"not an absolute Windows path: {path!r}")
    return ntpath.normpath(path)


def dev_layout(repo: str, python: str) -> Layout:
    """Any checkout and interpreter, e.g. a clone plus a dev venv on the build machine."""
    return Layout("native", windows_path(repo), windows_path(python), None, windows_path, ntpath)


def native_layout(root: str) -> Layout:
    r"""The offline bundle: ``<root>\app`` and ``<root>\runtime\python.exe``; root = the exe's directory."""
    root = windows_path(root)
    return dev_layout(ntpath.join(root, "app"), ntpath.join(root, "runtime", "python.exe"))
