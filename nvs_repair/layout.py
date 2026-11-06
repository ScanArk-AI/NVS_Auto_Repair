"""Where each stage writes, and how the two hand off to each other.

One output root per scene run; inside it, one directory per stage, numbered so
the order is obvious in a file listing:

    <out>/
      1_find/                       repair_view_finder
        repair/trajectories/traj_*/ frames/, poses.json, transforms.json, meta.json
      2_repair/                     repair_pipeline
        traj_*/ references/, opacity/, repaired/, repaired.mp4, before_after.mp4
      workflow.json

The hand-off is ``<out>/1_find/repair`` -> ``input.repair_root``; nothing else
crosses the boundary, which is why the stages can stay separate programs.
"""

from __future__ import annotations

from pathlib import Path

FIND = "1_find"
REPAIR = "2_repair"


def find_out(cfg) -> Path:
    return Path(cfg.out) / FIND


def repair_out(cfg) -> Path:
    return Path(cfg.out) / REPAIR


def repair_root(cfg) -> Path:
    """The directory repair_pipeline expects as ``input.repair_root``."""
    return find_out(cfg) / "repair"


def trajectories(cfg) -> list[Path]:
    root = repair_root(cfg) / "trajectories"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "poses.json").is_file())


def repaired_trajectories(cfg) -> list[Path]:
    """Trajectories that already have a full set of repaired frames."""
    done = []
    for traj in trajectories(cfg):
        frames = list((repair_out(cfg) / traj.name / "repaired").glob("*.png"))
        if frames:
            done.append(traj)
    return done


def selected_trajectories(cfg) -> list[Path]:
    """Trajectories that already have their reference views chosen."""
    return [t for t in trajectories(cfg)
            if (repair_out(cfg) / t.name / "references" / "selected.json").is_file()]


def is_complete(cfg) -> bool:
    trajs = trajectories(cfg)
    return bool(trajs) and len(repaired_trajectories(cfg)) == len(trajs)
