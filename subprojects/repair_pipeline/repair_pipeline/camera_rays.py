# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for nvs_repair (adapted from ArtiFixer's model_training/data/utils.py).
"""Camera conditioning for the repair model: PRoPE poses, intrinsics, Plücker rays.

Vendored from ArtiFixer ``model_training/data/utils.py`` (functions
``compute_camera_rays`` / ``compute_camera_conditioning`` / ``load_encoded_prompt``
and their helpers). Only the two entry points this pipeline calls are kept.

Why vendor instead of import: that module pulls in ``threedgrut`` (a CUDA
extension package) and the whole DL3DV dataset stack at import time, i.e. ~1000
lines of code that has nothing to do with inference -- and one more thing to
compile on a fresh machine. The maths below is copied, not reinterpreted; when
vendored it matched the original tensor for tensor (max |diff| 0).

Conventions that matter:

* ``transform_matrix`` is OpenGL camera-to-world (NeRFStudio).
* intrinsics are normalised by image size; distortion defaults to zero.
* frames are grouped in fours for the VAE's temporal compression, so the
  outputs are in *latent* frames, not pixel frames.
* the first frame defines the reference frame -- everything is expressed
  relative to it, and its translation is scaled by ``scale``.
"""

from __future__ import annotations

import numpy as np
import torch

MAX_SEQUENCE_LENGTH = 512
TEXT_EMBED_DIM = 4096

_INTRINSIC_KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy")
_DISTORTION_KEYS = ("k1", "k2", "p1", "p2")


# --------------------------------------------------------------------------- #
# SE3 helper
# --------------------------------------------------------------------------- #
def invert_se3(m: torch.Tensor) -> torch.Tensor:
    """Inverse of a batch of 4x4 rigid transforms."""
    rot = m[:, :3, :3]
    trans = m[:, :3, 3]
    out = torch.zeros_like(m)
    out[:, :3, :3] = rot.transpose(1, 2)
    out[:, :3, 3] = -torch.bmm(rot.transpose(1, 2), trans.unsqueeze(-1)).squeeze(-1)
    out[:, 3, 3] = 1.0
    return out


# --------------------------------------------------------------------------- #
# intrinsics
# --------------------------------------------------------------------------- #
def _intrinsics_vector(intrinsics: dict) -> torch.Tensor:
    for key in _INTRINSIC_KEYS:
        assert key in intrinsics, f"Camera intrinsics missing {key!r}"
    w, h = float(intrinsics["w"]), float(intrinsics["h"])
    assert w > 0 and h > 0, f"Camera image size must be positive, got {(w, h)}"
    return torch.FloatTensor([
        float(intrinsics["fl_x"]) / w, float(intrinsics["fl_y"]) / h,
        float(intrinsics["cx"]) / w, float(intrinsics["cy"]) / h,
        float(intrinsics.get("k1", 0.0)), float(intrinsics.get("k2", 0.0)),
        float(intrinsics.get("p1", 0.0)), float(intrinsics.get("p2", 0.0)),
    ])


def _frame_intrinsics(transforms: dict, frame: dict) -> torch.Tensor:
    intrinsics = dict(transforms)
    for key in (*_INTRINSIC_KEYS, *_DISTORTION_KEYS):
        if key in frame:
            intrinsics[key] = frame[key]
    return _intrinsics_vector(intrinsics)


def _average_temporal_intrinsics(intrinsics: torch.Tensor, skip_vae_check: bool) -> torch.Tensor:
    averaged = [intrinsics[0]]
    num_frames = intrinsics.shape[0]
    for i in range(1, num_frames, 4):
        group = intrinsics[i: min(i + 4, num_frames)]
        if skip_vae_check and group.shape[0] < 4:
            group = torch.cat([group, group[-1:].expand(4 - group.shape[0], -1)], dim=0)
        averaged.append(group.mean(dim=0))
    return torch.stack(averaged)


def _intrinsics_to_K(intrinsics: torch.Tensor) -> torch.Tensor:
    """Normalised intrinsics -> 3x3 with the 0.5 pixel-centre shift PRoPE expects."""
    K = torch.zeros(intrinsics.shape[0], 3, 3, dtype=intrinsics.dtype, device=intrinsics.device)
    K[:, 0, 0] = intrinsics[:, 0]
    K[:, 1, 1] = intrinsics[:, 1]
    K[:, 0, 2] = intrinsics[:, 2] - 0.5
    K[:, 1, 2] = intrinsics[:, 3] - 0.5
    K[:, 2, 2] = 1.0
    return K


# --------------------------------------------------------------------------- #
# rays
# --------------------------------------------------------------------------- #
def _undistort(image_points: torch.Tensor, principal_point: torch.Tensor,
               focal_length: torch.Tensor, radial: torch.Tensor,
               tangential: torch.Tensor, max_iterations: int = 10,
               tol: float = 1e-12) -> torch.Tensor:
    """Iteratively invert the OpenCV distortion model (rational radial + tangential).

    Thin-prism terms are zero in this pipeline, matching how ArtiFixer builds the
    parameters (``[k1, k2, 0, 0, 0, 0]``).
    """
    rays0 = (image_points - principal_point) / focal_length
    rays = rays0
    k1, k2, k3, k4, k5, k6 = radial
    p1, p2 = tangential
    for _ in range(max_iterations):
        sq = rays * rays
        r2 = sq.sum(dim=1)
        icd_num = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        icd_den = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
        icd = icd_num / icd_den
        xy_prod = rays[:, 0] * rays[:, 1]
        a1 = 2.0 * xy_prod
        a2 = r2 + 2.0 * sq[:, 0]
        a3 = r2 + 2.0 * sq[:, 1]
        delta = torch.stack([p1 * a1 + p2 * a2, p1 * a3 + p2 * a1], dim=1)
        new = (rays0 - delta) / icd[:, None]
        residual = rays - new
        rays = new
        if torch.mean(torch.square(residual)).item() <= tol:
            break
    return rays


def _camera_rays_for_intrinsics(intrinsics: torch.Tensor, averaged_c2ws_ref: torch.Tensor,
                                image_height: int, image_width: int) -> torch.Tensor:
    u = np.tile(np.arange(image_width), image_height)
    v = np.arange(image_height).repeat(image_width)
    image_points = torch.tensor(np.stack([u, v], axis=1), dtype=torch.float32) + 0.5

    rays_d_cam = []
    for fx_n, fy_n, cx_n, cy_n, k1, k2, p1, p2 in intrinsics.detach().cpu().tolist():
        rays_d_cam.append(_undistort(
            image_points,
            principal_point=torch.tensor([cx_n * image_width, cy_n * image_height], dtype=torch.float32),
            focal_length=torch.tensor([fx_n * image_width, fy_n * image_height], dtype=torch.float32),
            radial=torch.tensor([k1, k2, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            tangential=torch.tensor([p1, p2], dtype=torch.float32),
        ))
    rays = torch.stack([torch.cat([r, torch.ones_like(r[:, :1])], dim=1) for r in rays_d_cam])
    rays = rays / torch.linalg.norm(rays, dim=2, keepdim=True)

    rays = rays.to(device=averaged_c2ws_ref.device, dtype=averaged_c2ws_ref.dtype)
    rays_d_ref = torch.bmm(rays, averaged_c2ws_ref[:, :3, :3].transpose(1, 2))
    rays_o_ref = averaged_c2ws_ref[:, :3, 3].unsqueeze(1).expand(-1, rays_d_ref.shape[1], -1)
    rays_dxo_ref = torch.linalg.cross(rays_o_ref, rays_d_ref)
    return torch.cat([rays_dxo_ref, rays_d_ref], dim=-1).view(
        rays_d_ref.shape[0], image_height, image_width, 6)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def compute_camera_rays(transforms: dict, frame_indices: list[int], neighbor_indices: list[int],
                        scale: float, image_shape: tuple[int, int],
                        skip_vae_check: bool = False) -> dict[str, torch.Tensor]:
    """``(w2cs, Ks, neighbor_w2cs, neighbor_Ks, camera_rays)`` for one clip."""
    from scipy.spatial.transform import Rotation

    H, W = image_shape
    frames = transforms["frames"]
    c2ws = torch.stack([torch.FloatTensor(frames[i]["transform_matrix"]) for i in frame_indices]).clone()
    nbr_c2ws = torch.stack([torch.FloatTensor(frames[i]["transform_matrix"]) for i in neighbor_indices]).clone()
    target_intr = torch.stack([_frame_intrinsics(transforms, frames[i]) for i in frame_indices])
    nbr_intr = torch.stack([_frame_intrinsics(transforms, frames[i]) for i in neighbor_indices])

    c2ws[:, :3, 3] *= scale
    nbr_c2ws[:, :3, 3] *= scale
    target_intr = target_intr.to(dtype=c2ws.dtype)
    nbr_intr = nbr_intr.to(dtype=c2ws.dtype)

    ref_w2c = invert_se3(c2ws[0:1])
    c2ws_ref = torch.bmm(ref_w2c.expand(c2ws.shape[0], -1, -1), c2ws)

    # Poses are averaged in groups of four: the VAE compresses time 4x, and the
    # model conditions each latent frame on one averaged pose.
    if not skip_vae_check:
        assert (c2ws_ref.shape[0] - 1) % 4 == 0, f"frames must be 1 + 4n, got {c2ws_ref.shape[0]}"
    averaged = [c2ws_ref[0]]
    for i in range(1, c2ws_ref.shape[0], 4):
        group = c2ws_ref[i: min(i + 4, c2ws_ref.shape[0])]
        if skip_vae_check and group.shape[0] < 4:
            group = torch.cat([group, group[-1:].expand(4 - group.shape[0], -1, -1)], dim=0)
        pose = torch.eye(4, dtype=c2ws_ref.dtype)
        rot = Rotation.from_matrix(group[:, :3, :3].detach().cpu().numpy()).mean().as_matrix()
        pose[:3, :3] = torch.as_tensor(rot, dtype=c2ws_ref.dtype)
        pose[:3, 3] = group[:, :3, 3].mean(dim=0)
        averaged.append(pose)
    averaged = torch.stack(averaged)
    averaged_intr = _average_temporal_intrinsics(target_intr, skip_vae_check)

    nbr_c2ws_ref = torch.bmm(ref_w2c.expand(nbr_c2ws.shape[0], -1, -1), nbr_c2ws)
    return {
        "w2cs": invert_se3(averaged),
        "Ks": _intrinsics_to_K(averaged_intr),
        "neighbor_w2cs": invert_se3(nbr_c2ws_ref),
        "neighbor_Ks": _intrinsics_to_K(nbr_intr),
        "camera_rays": _camera_rays_for_intrinsics(averaged_intr, averaged, H, W),
    }


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
def load_encoded_prompt(prompt_paths, generator=None) -> tuple[torch.Tensor, str]:
    """Read ``caption.h5`` the way ArtiFixer does (uint16 bits reinterpreted as bf16)."""
    import h5py

    if not prompt_paths:
        return torch.zeros(MAX_SEQUENCE_LENGTH, TEXT_EMBED_DIM, dtype=torch.bfloat16), ""

    idx = torch.randint(0, len(prompt_paths), size=(1,), generator=generator).item()
    with h5py.File(prompt_paths[idx], "r") as f:
        dataset = f[list(f.keys())[0]]
        encoded = torch.tensor(dataset[:], dtype=torch.uint16).view(torch.bfloat16)
        encoded = torch.cat([encoded, torch.zeros(MAX_SEQUENCE_LENGTH - encoded.shape[0],
                                                  encoded.shape[1], dtype=torch.bfloat16)], dim=0)
        caption = dataset.attrs.get("caption", "")
    return encoded, caption
