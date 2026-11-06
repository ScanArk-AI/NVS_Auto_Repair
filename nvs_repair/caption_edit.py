"""Write the scene description into a generated workflow.yaml.

    python -m nvs_repair.caption_edit <workflow.yaml> <description>

The Windows app runs this between ``make-config`` and ``run.py all``.  The
text arrives as one argv item, so nothing in it is shell- or YAML-parsed --
which ``--set workflow.caption_text=...`` would do to a colon or a quote.
An empty (or blank) description means "run without one": auto_caption off.
The rest of the file is kept as it was.  Owner: A.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def set_caption(path, text: str) -> None:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    wf = cfg.get("workflow") or {}
    if text.strip():
        wf["caption_text"] = text
        wf.pop("auto_caption", None)
    else:
        wf.pop("caption_text", None)
        wf["auto_caption"] = False
    cfg["workflow"] = wf
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    set_caption(argv[0], argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
