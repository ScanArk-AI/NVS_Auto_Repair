r"""``python -m nvs_repair_gui``: how the native bundle's launcher opens the window
(``runtime\pythonw.exe -m nvs_repair_gui`` with ``app\windows`` on PYTHONPATH)."""

from .app import main

raise SystemExit(main())
