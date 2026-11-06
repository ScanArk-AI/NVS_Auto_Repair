"""PyInstaller entry point for the Windows GUI.

    nvs_repair.exe                       open the window
    nvs_repair.exe --version             print the version
    nvs_repair.exe --version-file <p>    write the version to <p> (CI: a --windowed exe has no console)

The package's own modules use relative imports, so the frozen entry has to be
a script outside it; this is that script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nvs_repair_gui import __version__  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nvs_repair", add_help=True)
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    ap.add_argument("--version-file", metavar="PATH", help="write the version to PATH and exit")
    a = ap.parse_args(argv)
    if a.version_file:
        Path(a.version_file).write_text(__version__ + "\n", encoding="utf-8")
        return 0
    if a.version:
        if sys.stdout is not None:      # None in a --windowed exe
            print(__version__)
        return 0
    from nvs_repair_gui.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
