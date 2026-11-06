# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for nvs_repair (adapted from ArtiFixer's model_training/data/utils.py).
"""Gaussian-frustum reference-view selection.

The two functions are copied from ArtiFixer commit ``abe7db7`` ("Add
gaussian_frustum neighbor selection strategy", ``model_training/data/utils.py``)
so the selection is comparable with upstream rather than a private variant.

Shape of the idea: project every Gaussian centre into every candidate camera,
pack the in-frame test as a bit per Gaussian, then rank candidates by how many
Gaussians they share with the union over the trajectory's frames.

Occlusion and lens distortion are ignored, as upstream does. That over-estimates
the frustum of wide/fisheye cameras, so those rigs win more often -- which is
geometrically defensible (they really do see more) but worth knowing when
reading the picks.
"""

from __future__ import annotations

import numpy as np

# Upstream constants.
GAUSSIAN_FRUSTUM_NEAR = 0.01
GAUSSIAN_FRUSTUM_FAR = 1000.0


def gaussian_frustum_masks(
    positions: np.ndarray,
    extrinsics_c2w: np.ndarray,
    intrinsics: np.ndarray,
    near: float = GAUSSIAN_FRUSTUM_NEAR,
    far: float = GAUSSIAN_FRUSTUM_FAR,
) -> np.ndarray:
    """Bit-packed ``(num_cameras, ceil(num_points/8))`` in-frustum masks.

    ``extrinsics_c2w`` are OpenGL camera-to-world (the transforms.json
    convention); ``intrinsics`` rows are ``(w, h, fl_x, fl_y, cx, cy)``.
    """
    positions = np.asarray(positions, dtype=np.float32)
    masks = []
    for c2w, (w, h, fl_x, fl_y, cx, cy) in zip(np.asarray(extrinsics_c2w), np.asarray(intrinsics)):
        w2c = np.linalg.inv(c2w).astype(np.float32)
        local = positions @ w2c[:3, :3].T + w2c[:3, 3]
        depth = -local[:, 2]
        in_depth = (depth > near) & (depth < far)
        safe_depth = np.where(in_depth, depth, 1.0)
        u = fl_x * local[:, 0] / safe_depth + cx
        v = -fl_y * local[:, 1] / safe_depth + cy
        masks.append(np.packbits(in_depth & (u >= 0) & (u < w) & (v >= 0) & (v < h)))
    return np.stack(masks)


def select_neighbor_indices(target_mask: np.ndarray, photo_masks: np.ndarray, num_views: int = 12):
    """Rank candidates by shared Gaussians with the trajectory's union mask.

    Returns ``(indices, overlap)`` with indices ascending and ``overlap``
    parallel to them, so callers can log how good the picks actually are.
    """
    if len(photo_masks) < num_views:
        return [], []
    overlap = np.bitwise_count(photo_masks & target_mask).sum(axis=1).astype(np.int64)
    order = np.argsort(-overlap, kind="stable")[:num_views]
    picked = np.sort(order)
    return picked.tolist(), overlap[picked].tolist()
