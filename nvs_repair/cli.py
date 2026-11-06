"""Command line: doctor | find | select | repair | all."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import stages
from .config import ROOT, dump, load
from .envs import doctor


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="nvs_repair", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    helps = {
        "doctor": "check both stages' environments and artifacts",
        "find": "stage 1: locate the trajectories worth repairing",
        "select": "stage 2a: pick reference views for each trajectory",
        "repair": "stage 2b: render opacity, run the repair model, extract frames",
        "all": "stage 1 then stage 2",
        "plan": "print the resolved config and the commands that would run",
        "make-config": "generate workflow configs for a scene directory (3DGS ply + COLMAP + images)",
        "caption": "describe the scene with the VLM and write caption.h5",
    }
    for name, text in helps.items():
        p = sub.add_parser(name, help=text)
        if name == "caption":
            p.add_argument("--kind", choices=["scene", "scene_motion"], default="scene")
            p.add_argument("--frames", type=int, default=4, help="images shown to the VLM")
        elif name == "make-config":
            p.add_argument("--scene-root", help="one scene directory")
            p.add_argument("--batch-root", help="a directory whose sub-directories are scenes")
            p.add_argument("--out", default=None, help="default configs/scenes/generated")
            p.add_argument("--variant", default="1.3b")
            paths = p.add_mutually_exclusive_group()
            paths.add_argument("--object-centric", dest="paths", action="store_const", const="object-centric",
                               help="path settings for walking around a subject outdoors / on open ground")
            paths.add_argument("--object-centric-indoor", dest="paths", action="store_const",
                               const="object-centric-indoor",
                               help="path settings for walking around a table or in a small room")
            p.set_defaults(paths="default")
            p.add_argument("--units-per-metre", default="auto",
                           help="scene units per metre for the finder: a number, or 'auto' (default) to "
                                "estimate it from the camera height; use 1 for a metric capture")
            p.add_argument("--mask-invalid", choices=["black", "white", "none"], default=None)
            continue
        p.add_argument("--config", required=True)
        p.add_argument("--set", dest="overrides", action="append", default=[])
        p.add_argument("--limit", type=int, default=None, help="only the first N trajectories")

    a = ap.parse_args(argv)

    if a.command == "caption":
        from .caption_vlm import ensure_caption

        cfg = load(a.config, a.overrides)
        path = ensure_caption(cfg, Path(cfg.out), kind=a.kind, frames=a.frames)
        print(f"caption -> {path}")
        print((path.with_suffix(".txt")).read_text(encoding="utf-8").strip())
        return 0

    if a.command == "make-config":
        from .make_config import discover, generate

        if not (a.scene_root or a.batch_root):
            ap.error("make-config needs --scene-root or --batch-root")
        out = Path(a.out) if a.out else ROOT / "configs" / "scenes" / "generated"
        scenes = [Path(a.scene_root)] if a.scene_root else discover(Path(a.batch_root))
        if not scenes:
            print("no scene directories found", file=sys.stderr)
            return 1
        for scene in scenes:
            path = generate(scene, out, variant=a.variant, mask_invalid=a.mask_invalid,
                            units_per_metre=a.units_per_metre, paths=a.paths)
            print(f"{scene.name:16} -> {path}")
        return 0
    cfg = load(a.config, a.overrides)
    if a.limit:
        cfg.repair.limit = a.limit

    if a.command == "doctor":
        return doctor(cfg)

    if a.command == "plan":
        print(json.dumps(dump(cfg), indent=1))
        second = ([stages.progressive_stage(cfg)] if stages.reference_mode(cfg) == "progressive"
                  else [stages.pipeline_stage(cfg, "select"), stages.pipeline_stage(cfg, "repair")])
        for st in (stages.find_stage(cfg), *second):
            print(f"\n{st.name}: cd {st.cwd} && {' '.join(st.command())}")
        return 0

    if a.command == "find":
        return stages.find_stage(cfg).run()
    progressive = stages.reference_mode(cfg) == "progressive"
    if a.command == "select":
        if progressive:
            print("reference_mode=progressive picks references during repair; run `repair` or `all`")
            return 0
        return stages.pipeline_stage(cfg, "select").run()
    if a.command == "repair":
        stages.wait_for_vram(cfg, "repair")         # may also switch on model.low_vram
        return (stages.progressive_stage(cfg) if progressive else stages.pipeline_stage(cfg, "repair")).run()
    return stages.run_all(cfg)
