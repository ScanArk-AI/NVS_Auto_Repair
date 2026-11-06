"""Render the alpha channel ArtiFixer needs as an extra input.

``repair_view_finder`` writes RGB only, but ArtiFixer conditions on rendered
opacity: low-alpha pixels tell the model where geometry is missing, so a flat
255 would hide exactly the artifacts being repaired. Re-render alpha from the
same 3DGS at the same poses and throw the RGB away.

The trajectories are explicit per-frame poses and are rendered as they are,
never re-sampled along the path.

The render runs in a child process, so whatever gsplat holds on the GPU is
released before the repair model loads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def render_opacity(traj_dirs, ply: Path, device: str = "cuda") -> int:
    """Render ``opacity/00000.png…`` for one or more trajectories.

    Takes a list on purpose: loading the cloud costs ~20 s for a 4.6M-Gaussian
    scene, and a per-trajectory process would pay that once per clip (plus the
    gsplat JIT on the first one). One process for the whole batch.
    """
    from PIL import Image

    from .render_backend import GaussianRenderer  # gsplat-backed, imported lazily

    if isinstance(traj_dirs, (str, Path)):
        traj_dirs = [traj_dirs]

    renderer = GaussianRenderer.from_ply(Path(ply), device=device)
    done = 0
    for traj_dir in traj_dirs:
        traj_dir = Path(traj_dir)
        poses = json.loads((traj_dir / "poses.json").read_text(encoding="utf-8"))
        c2w = np.array(poses["c2w"], dtype=np.float64)
        cam = poses["camera"]
        K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float64)

        _, alpha, _ = renderer.render(
            np.linalg.inv(c2w), np.repeat(K[None], len(c2w), axis=0),
            int(cam["width"]), int(cam["height"]),
        )
        out = traj_dir / "opacity"
        out.mkdir(parents=True, exist_ok=True)
        for i in range(len(c2w)):
            Image.fromarray((alpha[i, :, :, 0] * 255.0 + 0.5).astype(np.uint8), mode="L").save(
                out / f"{i:05d}.png"
            )
        done += 1
    return done


def ensure_opacity(traj_dirs: list[Path], ply: Path, python: str | None, device: str,
                   expected: int) -> None:
    """Fill in any missing ``opacity/`` by running this module in a child process.

    ``python=None`` means the interpreter running this process; set it only when
    ``gsplat`` is installed in a different environment.
    """
    python = python or sys.executable
    missing = [d for d in traj_dirs
               if len(list((d / "opacity").glob("*.png"))) != expected]
    if not missing:
        return
    cmd = [python, "-m", "repair_pipeline.opacity", *[str(d) for d in missing],
           str(ply), "--device", device]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8",
                               "PYTHONPATH": str(Path(__file__).resolve().parent.parent)})
    if proc.returncode != 0:
        raise RuntimeError(f"opacity render failed: {proc.stderr.strip()[-800:]}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Render ArtiFixer opacity for one trajectory")
    ap.add_argument("paths", nargs="+", help="one or more trajectory dirs, then the ply")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    *trajs, ply = a.paths
    n = render_opacity([Path(t) for t in trajs], Path(ply), a.device)
    print(f"opacity: {n} trajectories rendered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
