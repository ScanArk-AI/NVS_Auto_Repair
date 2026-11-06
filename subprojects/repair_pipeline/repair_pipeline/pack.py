"""Assemble the NeRFStudio-style ``transforms`` dict the repair model expects.

Layout (the same one ArtiFixer's ``prepare_colmap_artifixer_inputs`` produces):

* frames ``0..T-1``    the trajectory's own frames, in order
* frames ``T..T+M-1``  the selected reference photos

Targets are read from ``render_dir`` by index, neighbours from
``frames[i]["file_path"]`` resolved against ``image_root`` -- so only the
neighbour entries need real paths.

Everything is returned as a plain dict that the model consumes in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .colmap_io import Camera
from .poses import CAMERA_CONVENTION_FLIP, c2w_opencv_to_opengl
from .prepare import photo_intrinsics, prepared_intrinsics


def build_transforms(
    traj_dir: Path,
    c2w_opencv: np.ndarray,
    traj_camera: tuple[int, int, float, float, float, float],
    refs: list[dict],
    cams: dict,
    images_by_name: dict,
    target_fov_deg: float | None = None,
    neighbor_intrinsics: str = "prepared",
) -> tuple[dict, list[int], list[int]]:
    """Return ``(transforms, target_indices, neighbor_indices)``.

    ``neighbor_intrinsics`` = ``prepared`` describes each reference by the
    camera of the prepared image it is fed as (needs ``target_fov_deg``, the
    FOV it was prepared at); ``photo`` by the source photo's camera, the
    pre-fix behaviour.
    """
    if neighbor_intrinsics not in ("prepared", "photo"):
        raise ValueError(f"refer.neighbor_intrinsics must be prepared or photo, got {neighbor_intrinsics!r}")
    if neighbor_intrinsics == "prepared" and target_fov_deg is None:
        raise ValueError("neighbor_intrinsics=prepared needs the FOV the references were prepared at")
    tw, th, tfx, tfy, tcx, tcy = traj_camera
    frames: list[dict] = []

    for i, m in enumerate(c2w_opencv):
        frames.append({
            "file_path": f"frames/{i:05d}.png",
            "transform_matrix": c2w_opencv_to_opengl(m).tolist(),
            "w": int(tw), "h": int(th), "fl_x": float(tfx), "fl_y": float(tfy),
            "cx": float(tcx), "cy": float(tcy),
        })

    neighbor_indices: list[int] = []
    for ref in refs:
        im = images_by_name[ref["name"]]
        cam: Camera = cams[im.camera_id]
        intr = (prepared_intrinsics(cam, int(tw), int(th), float(target_fov_deg))
                if neighbor_intrinsics == "prepared" else photo_intrinsics(cam))
        idx = len(frames)
        frames.append({
            "file_path": ref["prepared"],
            "transform_matrix": (np.linalg.inv(im.w2c) @ CAMERA_CONVENTION_FLIP).tolist(),
            **intr,
        })
        neighbor_indices.append(idx)

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(tfx), "fl_y": float(tfy), "cx": float(tcx), "cy": float(tcy),
        "w": int(tw), "h": int(th),
        "frames": frames,
    }
    return transforms, list(range(len(c2w_opencv))), neighbor_indices

