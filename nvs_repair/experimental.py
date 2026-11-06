"""Experimental inference methods, run next to the standard pipeline.

Nothing here changes the standard ``run.py`` workflow or its ``2_repair/``
output. Each experiment arm writes its own ``<out>/2_repair_<name>/`` and reuses
the same ``1_find/`` trajectories and ``caption.h5``::

    python -m nvs_repair.experimental progressive --config <workflow.yaml> --name prog \\
        [--rset progressive.neighbor_intrinsics=photo ...]
    python -m nvs_repair.experimental compare --config <workflow.yaml> \\
        --arms 2_repair 2_repair_prog --labels standard progressive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import layout, stages
from .config import ROOT, load

SUB = ROOT / "subprojects" / "repair_pipeline"


def _ensure_caption(cfg) -> bool:
    if ((Path(cfg.out) / "caption.h5").is_file() or not cfg.workflow.get("auto_caption", True)
            or stages._repair_backend(cfg) == "passthrough"):
        return True
    from .caption_vlm import ensure_caption

    try:
        ensure_caption(cfg, Path(cfg.out), kind=str(cfg.workflow.get("caption_kind", "scene")))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"caption failed: {exc}", file=sys.stderr)
        return False


def progressive(cfg, name: str, rsets: list[str]) -> int:
    if not layout.trajectories(cfg):
        print(f"no trajectories under {layout.repair_root(cfg)}; run the find stage first", file=sys.stderr)
        return 1
    if not _ensure_caption(cfg):
        return 1
    stages.wait_for_vram(cfg, "repair")
    # The arm's own enabled/... overrides come last so `progressive.enabled=false`
    # gives the photos-only baseline through the same code path.
    stage = stages.progressive_stage(cfg, out_dir=Path(cfg.out) / f"2_repair_{name}", extra=rsets)
    stage.name = f"experimental:progressive:{name}"
    return stage.run()


# --------------------------------------------------------------------------- #
# side-by-side comparison
# --------------------------------------------------------------------------- #
def _load(directory: Path) -> np.ndarray | None:
    from PIL import Image

    files = sorted(directory.glob("*.png"))
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]) if files else None


def _font(size: int):
    """``$NVS_REPAIR_FONT`` if set (e.g. a CJK font for Chinese labels), else PIL's default."""
    from PIL import ImageFont

    path = os.environ.get("NVS_REPAIR_FONT")
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            print(f"NVS_REPAIR_FONT={path} could not be loaded; using the default font", file=sys.stderr)
    try:
        return ImageFont.load_default(size)
    except TypeError:          # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def _row(frames: list[np.ndarray], labels: list[str], height: int, font) -> np.ndarray:
    from PIL import Image, ImageDraw

    panels = []
    for f in frames:
        img = Image.fromarray(f)
        w = int(round(img.width * height / img.height))
        panels.append(img.resize((w + w % 2, height), Image.LANCZOS))
    strip = 28
    canvas = Image.new("RGB", (sum(p.width for p in panels), height + strip), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for p, label in zip(panels, labels):
        canvas.paste(p, (x, strip))
        draw.text((x + 8, 5), label, fill=(255, 255, 255), font=font)
        x += p.width
    return np.asarray(canvas)


def compare(cfg, arms: list[str], labels: list[str], height: int, fps: int, name: str = "compare") -> int:
    if str(SUB) not in sys.path:
        sys.path.insert(0, str(SUB))
    from repair_pipeline.frames import write_video
    from PIL import Image

    out = Path(cfg.out) / name
    out.mkdir(parents=True, exist_ok=True)
    labels = labels or arms
    font = _font(18)
    overview = []
    for traj in layout.trajectories(cfg):
        render = _load(traj / "frames")
        clips = [_load(Path(cfg.out) / arm / traj.name / "repaired") for arm in arms]
        if render is None or any(c is None for c in clips):
            print(f"{traj.name}: missing in {[a for a, c in zip(arms, clips) if c is None]}, skipped")
            continue
        n = min(len(render), *(len(c) for c in clips))
        names = ["3DGS render", *labels]
        rows = [_row([render[k], *(c[k] for c in clips)], names, height, font) for k in range(n)]
        write_video(np.stack(rows), out / f"{traj.name}.mp4", fps=fps)
        overview.append(_row([render[n // 2], *(c[n // 2] for c in clips)],
                             [f"{traj.name}  render", *labels], 200, _font(14)))
        print(f"{traj.name}: {out / (traj.name + '.mp4')}")
    if overview:
        Image.fromarray(np.concatenate(overview, axis=0)).save(out / "overview.png")
        print(f"overview: {out / 'overview.png'}")
    summary = {}
    for arm in arms:
        run = Path(cfg.out) / arm / "run.json"
        if run.is_file():
            r = json.loads(run.read_text(encoding="utf-8"))
            summary[arm] = {"method": r.get("method", "standard"), "progressive": r.get("progressive"),
                            "order": r.get("order")}
    (out / "arms.json").write_text(json.dumps({"arms": arms, "labels": labels, "runs": summary}, indent=1), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nvs_repair.experimental", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("progressive", help="repair with earlier repaired frames as extra references")
    p.add_argument("--name", required=True, help="arm name -> <out>/2_repair_<name>")
    p.add_argument("--rset", action="append", default=[], help="repair_pipeline override, e.g. progressive.max_generated=2")
    c = sub.add_parser("compare", help="multi-panel videos + overview.png across arms")
    c.add_argument("--arms", nargs="+", required=True, help="output dirs under <out>, e.g. 2_repair 2_repair_prog")
    c.add_argument("--labels", nargs="*", default=None)
    c.add_argument("--height", type=int, default=300)
    c.add_argument("--fps", type=int, default=15)
    c.add_argument("--name", default="compare", help="output dir under <out>")
    for q in (p, c):
        q.add_argument("--config", required=True)
        q.add_argument("--set", dest="overrides", action="append", default=[])
        q.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load(a.config, a.overrides)
    if a.limit:
        cfg.repair.limit = a.limit
    if a.command == "progressive":
        return progressive(cfg, a.name, a.rset)
    return compare(cfg, a.arms, a.labels, a.height, a.fps, a.name)


if __name__ == "__main__":
    raise SystemExit(main())
