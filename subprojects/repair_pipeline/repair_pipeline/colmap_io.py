"""COLMAP sparse-model reader (binary and text), dependency-free.

Binary parsing uses :mod:`struct`/numpy rather than pycolmap: no heavy dependency, no
version coupling, and the 2-D observations -- which dominate ``images.bin`` --
are seeked over instead of read.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .poses import quat_wxyz_to_matrix

# COLMAP camera model id -> (name, parameter count)
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

FISHEYE_MODELS = {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"}
# One focal length (f, cx, cy, ...): COLMAP's default SIMPLE_RADIAL is one of them.
_SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


@dataclass
class Camera:
    id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray

    @property
    def model_name(self) -> str:
        return CAMERA_MODELS[self.model_id][0]

    @property
    def is_fisheye(self) -> bool:
        return self.model_name in FISHEYE_MODELS

    @property
    def fx_fy_cx_cy(self) -> tuple[float, float, float, float]:
        """Pixel intrinsics, distortion dropped."""
        p = self.params
        if self.model_name in _SINGLE_FOCAL:
            f, cx, cy = p[0], p[1], p[2]
            return float(f), float(f), float(cx), float(cy)
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    @property
    def fisheye_k(self) -> tuple[float, float, float, float]:
        """``(k1, k2, k3, k4)`` of OpenCV's equidistant fisheye model.

        SIMPLE_RADIAL_FISHEYE / RADIAL_FISHEYE are that model with the higher
        terms zero; THIN_PRISM_FISHEYE's tangential and prism terms are dropped.
        """
        p = [float(x) for x in self.params]
        return {
            "OPENCV_FISHEYE": (p[4], p[5], p[6], p[7]) if len(p) >= 8 else (0.0,) * 4,
            "SIMPLE_RADIAL_FISHEYE": (p[3], 0.0, 0.0, 0.0),
            "RADIAL_FISHEYE": (p[3], p[4], 0.0, 0.0),
            "THIN_PRISM_FISHEYE": (p[4], p[5], p[8], p[9]) if len(p) >= 10 else (0.0,) * 4,
        }.get(self.model_name, (0.0, 0.0, 0.0, 0.0))

    @property
    def radial_tangential(self) -> tuple[float, float, float, float]:
        """``(k1, k2, p1, p2)`` of a rectilinear camera (FULL_OPENCV's higher
        radial terms are dropped; models without distortion give zeros)."""
        p = [float(x) for x in self.params]
        return {
            "SIMPLE_RADIAL": (p[3], 0.0, 0.0, 0.0),
            "RADIAL": (p[3], p[4], 0.0, 0.0),
            "OPENCV": tuple(p[4:8]),
            "FULL_OPENCV": tuple(p[4:8]),
        }.get(self.model_name, (0.0, 0.0, 0.0, 0.0))


@dataclass
class RegisteredImage:
    id: int
    name: str
    qvec: np.ndarray  # (4,) w, x, y, z
    tvec: np.ndarray  # (3,)
    camera_id: int

    @property
    def w2c(self) -> np.ndarray:
        """OpenCV world-to-camera 4x4."""
        out = np.eye(4)
        out[:3, :3] = quat_wxyz_to_matrix(self.qvec)
        out[:3, 3] = self.tvec
        return out


def _read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            n_params = CAMERA_MODELS[model_id][1]
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)), dtype=np.float64)
            cameras[cam_id] = Camera(cam_id, model_id, int(width), int(height), params)
    return cameras


def _read_images_binary(path: Path) -> list[RegisteredImage]:
    images: list[RegisteredImage] = []
    with open(path, "rb") as f:
        (num,) = struct.unpack("<Q", f.read(8))
        for _ in range(num):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00" or not ch:
                    break
                name += ch
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * num_points2d, 1)  # 0-D observations: 24 bytes each
            images.append(RegisteredImage(image_id, name.decode("utf-8", "replace"), qvec, tvec, camera_id))
    return images


def _read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model_name = int(parts[0]), parts[1]
        model_id = next((k for k, (n, _) in CAMERA_MODELS.items() if n == model_name), None)
        if model_id is None:
            raise ValueError(f"{path}: unsupported camera model {model_name!r}")
        cameras[cam_id] = Camera(cam_id, model_id, int(parts[2]), int(parts[3]),
                                 np.array([float(x) for x in parts[4:]], dtype=np.float64))
    return cameras


def _read_images_text(path: Path) -> list[RegisteredImage]:
    images: list[RegisteredImage] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        p = line.split()
        images.append(RegisteredImage(
            int(p[0]),
            p[9],
            np.array([float(x) for x in p[1:5]], dtype=np.float64),
            np.array([float(x) for x in p[5:8]], dtype=np.float64),
            int(p[8]),
        ))
        i += 2  # every image line is followed by exactly one observation line
    return images


def read_model(sparse_dir: Path) -> tuple[dict[int, Camera], list[RegisteredImage]]:
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "cameras.bin").is_file():
        return _read_cameras_binary(sparse_dir / "cameras.bin"), _read_images_binary(sparse_dir / "images.bin")
    return _read_cameras_text(sparse_dir / "cameras.txt"), _read_images_text(sparse_dir / "images.txt")
