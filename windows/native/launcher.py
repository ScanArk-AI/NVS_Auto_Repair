"""nvs_repair.exe for the offline Windows bundle: start the GUI with the bundled Python.

Frozen with PyInstaller by build_bundle.ps1 into a tiny exe next to runtime\\
and app\\. It carries no dependencies of its own; it only runs
``runtime\\pythonw.exe -m nvs_repair_gui`` with ``app\\windows`` on the path.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    pythonw = root / "runtime" / "pythonw.exe"
    if not pythonw.is_file():
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None, f"{pythonw} is missing.\n\nUnzip every volume of the download into one folder "
                  "and start nvs_repair.exe from there.", "nvs_repair", 0x10)
        return 1
    env = dict(os.environ, PYTHONPATH=str(root / "app" / "windows"), PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONHOME", None)
    subprocess.Popen([str(pythonw), "-m", "nvs_repair_gui"], cwd=str(root), env=env,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
