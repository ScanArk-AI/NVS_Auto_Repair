"""Minimal gsplat renderer used only to produce the alpha channel.

``gsplat`` JIT-compiles its CUDA kernels on first use; that needs ``ninja``
(``_ensure_ninja`` finds the one installed next to the interpreter) and a CUDA
toolkit matching the torch build -- both set up by ``bootstrap.sh``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from . import ply_io


def _ensure_ninja() -> None:
    if shutil.which("ninja"):
        return
    # gsplat JIT-compiles with ninja; it is often installed next to the
    # interpreter (bin/ on Linux, Scripts\ on Windows) but not on PATH when the
    # environment is not activated.
    import sys

    here = Path(sys.executable).parent
    for cand in (here, here / "Scripts"):
        if shutil.which("ninja", path=str(cand)):
            os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
            return


class GaussianRenderer:
    """One gaussian cloud on the GPU, rendering explicit camera poses."""

    def __init__(self, gaussians: ply_io.Gaussians, device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = device
        self.num_gaussians = len(gaussians)
        t = torch.from_numpy
        self.means = t(np.ascontiguousarray(gaussians.centers)).float().to(device)
        self.quats = t(np.ascontiguousarray(gaussians.quats_wxyz)).float().to(device)
        self.scales = t(np.ascontiguousarray(gaussians.scales)).float().to(device)
        self.opacities = t(np.ascontiguousarray(gaussians.opacities[:, 0])).float().to(device)
        self.sh = t(np.ascontiguousarray(gaussians.sh)).float().to(device)
        self.sh_degree = gaussians.sh_degree
        self.colors = t(np.ascontiguousarray(gaussians.rgbs)).float().to(device)

    @classmethod
    def from_ply(cls, ply_path: Path, device: str = "cuda") -> "GaussianRenderer":
        _ensure_ninja()
        return cls(ply_io.read_gaussians(Path(ply_path)), device=device)

    def render(self, w2cs: np.ndarray, Ks: np.ndarray, width: int, height: int):
        """``w2cs`` are OpenCV world-to-camera. Returns (rgb, alpha, depth)."""
        from gsplat import rasterization

        torch = self.torch
        viewmats = torch.from_numpy(np.ascontiguousarray(w2cs)).float().to(self.device)
        ks = torch.from_numpy(np.ascontiguousarray(Ks)).float().to(self.device)

        with torch.no_grad():
            colors, alphas, _ = rasterization(
                self.means, self.quats, self.scales, self.opacities,
                self.sh if self.sh is not None else self.colors,
                viewmats, ks, int(width), int(height),
                sh_degree=self.sh_degree,
                render_mode="RGB+ED",
                rasterize_mode="classic",   # matches how the target frames were rendered
                absgrad=False,
            )
        rgb = colors[..., :3].detach().cpu().numpy()
        depth = colors[..., 3:4].detach().cpu().numpy()
        alpha = alphas.detach().cpu().numpy()
        return np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0), depth
