"""Windows front end for nvs_repair: a small GUI that runs the pipeline inside WSL2.

Modules and owners (see docs/windows-*.md for the user view):

    wsl.py     run commands in the WSL distro, map Windows <-> WSL paths   (C)
    setup.py   detect WSL / GPU / install state, first-time install steps  (C)
    runner.py  turn the GUI's Options into pipeline commands, parse progress (A)
    steps.py   Step / Layout: what runner builds and a backend executes     (A)
    app.py     the tkinter window                                          (B)

windows/launch.py is the PyInstaller entry point.
"""

__version__ = "0.1.0"
