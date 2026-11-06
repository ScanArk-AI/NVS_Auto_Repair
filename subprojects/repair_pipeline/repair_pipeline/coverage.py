"""Coverage of a trajectory's view by a set of cameras.

Shared by reference selection (``references.py``) and the progressive mode
(``progressive.py``). Masks are bit-packed per camera, one bit per Gaussian, as
``frustum.gaussian_frustum_masks`` returns them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frustum import gaussian_frustum_masks


@dataclass
class TrajMasks:
    c2w_cv: np.ndarray
    camera: tuple
    target_gl: np.ndarray
    per_frame: np.ndarray      # (T, bytes) packed in-frustum bits
    union: np.ndarray


def popcount(bits: np.ndarray) -> np.ndarray:
    return np.bitwise_count(bits).sum(axis=-1).astype(np.int64)


def union_bits(arrays: list[np.ndarray], like: np.ndarray) -> np.ndarray:
    return np.bitwise_or.reduce(arrays, axis=0) if arrays else np.zeros_like(like)


class DirectionDemand:
    """(target frame, Gaussian) pairs, and which of them a reference camera covers.

    A pair counts as covered when the reference has the Gaussian in its frustum
    *and* looks at it within ``angle_deg`` of the target frame's ray. Occlusion
    is still ignored, as in the frustum test. Gaussians are subsampled (fixed
    seed) so a candidate costs one (T, S) dot product.
    """

    def __init__(self, scene, tm: TrajMasks, n_sample: int, angle_deg: float, seed: int):
        n = len(scene.positions)
        union_idx = np.flatnonzero(np.unpackbits(tm.union)[:n])
        rng = np.random.default_rng(seed)
        self.sel = (np.sort(rng.choice(union_idx, min(int(n_sample), len(union_idx)), replace=False))
                    if len(union_idx) else union_idx)
        self.pts = scene.positions[self.sel].astype(np.float32)
        vis = np.unpackbits(tm.per_frame, axis=1)[:, :n][:, self.sel].astype(bool)    # (T, S)
        d = self.pts[None] - tm.target_gl[:, None, :3, 3].astype(np.float32)          # (T, S, 3)
        d /= np.linalg.norm(d, axis=2, keepdims=True) + 1e-9
        self.vis, self.dirs = vis, d
        self.bits = np.packbits(vis.ravel())
        self.cos_thr = float(np.cos(np.deg2rad(angle_deg)))

    def covered(self, c2w_gl: np.ndarray, intr_row) -> np.ndarray:
        if not len(self.pts):
            return self.bits.copy()
        in_f = np.unpackbits(gaussian_frustum_masks(self.pts, np.asarray(c2w_gl)[None],
                                                    np.asarray([intr_row], dtype=np.float64))[0])
        in_f = in_f[: len(self.pts)].astype(bool)
        dr = self.pts - np.asarray(c2w_gl, dtype=np.float32)[:3, 3]
        dr /= np.linalg.norm(dr, axis=1, keepdims=True) + 1e-9
        cos = np.einsum("tsk,sk->ts", self.dirs, dr)
        return np.packbits((self.vis & in_f[None] & (cos >= self.cos_thr)).ravel())

    def fraction(self, covered: np.ndarray) -> float:
        total = int(popcount(self.bits))
        return float(popcount(covered & self.bits)) / total if total else 0.0
