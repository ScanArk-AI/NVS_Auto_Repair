"""Command line: ``select`` / ``repair`` / ``all``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import dump, load_config
from .pipeline import Pipeline


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", required=True, help="scene config yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override, e.g. --set model.variant=14b")
    ap.add_argument("--limit", type=int, default=None, help="only the first N trajectories")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="repair_pipeline", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("select", "choose reference views and export them"),
        ("repair", "render opacity (if missing) and run the repair model"),
        ("all", "select + repair"),
    ):
        p = sub.add_parser(name, help=help_text)
        _common(p)

    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)

    if a.dry_run:
        print(json.dumps(dump(cfg), indent=1))
        return 0

    stages = {"select": ("select",), "repair": ("repair",), "all": ("select", "repair")}[a.command]
    report = Pipeline(cfg).run(stages=stages, limit=a.limit)
    print(f"done: {report['n']} trajectories in {report['elapsed_s']}s -> {cfg.output.dir}")
    return 0
