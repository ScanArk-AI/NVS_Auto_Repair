"""Minimal COLMAP reader (binary and text) and model discovery."""

from __future__ import annotations

import os
import struct

import numpy as np

# COLMAP's camera model ids (src/colmap/sensor/models.h) -> parameter count.
MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
               3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV",
               7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE",
               10: "THIN_PRISM_FISHEYE"}
# Models whose first parameter is a single focal length shared by x and y.
SINGLE_FOCAL = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}


def find_sparse_model(scene: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(scene, explicit)
    patterns = ["", "sparse/0", "sparse", "sfm_results", "sfm_result/final",
                "sfm_result", "sfm_results/final", "colmap/sparse/0"]
    for p in patterns:
        cand = os.path.join(scene, p)
        if os.path.isdir(cand) and any(
                os.path.isfile(os.path.join(cand, f))
                for f in ("images.bin", "images.txt")):
            return cand
    for dirpath, _dirs, _files in os.walk(scene):
        if any(os.path.isfile(os.path.join(dirpath, f))
               for f in ("images.bin", "images.txt")):
            return dirpath
    raise SystemExit(f"no COLMAP model under {scene}")


def _bin_cameras(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    off, cams = 8, {}
    for _ in range(n):
        cid, model, w, h = struct.unpack("<iiQQ", b[off:off + 24]); off += 24
        npar = MODEL_PARAMS[model]
        par = struct.unpack("<" + "d" * npar, b[off:off + 8 * npar]); off += 8 * npar
        cams[cid] = {"model": MODEL_NAMES.get(model, str(model)),
                     "W": int(w), "H": int(h), "params": list(par)}
    return cams


def _txt_cameras(path):
    cams = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cams[int(p[0])] = {"model": p[1], "W": int(p[2]), "H": int(p[3]),
                           "params": list(map(float, p[4:]))}
    return cams


def _bin_images(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    off, out = 8, []
    for _ in range(n):
        off += 4
        qw, qx, qy, qz = struct.unpack("<4d", b[off:off + 32]); off += 32
        t = np.array(struct.unpack("<3d", b[off:off + 24])); off += 24
        cid = struct.unpack("<i", b[off:off + 4])[0]; off += 4
        end = b.index(b"\x00", off)
        name = b[off:end].decode("utf-8", "ignore"); off = end + 1
        num = struct.unpack("<Q", b[off:off + 8])[0]
        off += 8 + num * 24
        out.append({"cam_id": cid, "name": name,
                    "q": np.array([qw, qx, qy, qz]), "t": t})
    return out


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _is_image_line(p: list[str]) -> bool:
    """``IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME``; a POINTS2D line is all numbers."""
    return len(p) >= 10 and p[0].isdigit() and p[8].isdigit() and not _is_number(p[9])


def _txt_images(path):
    # Each image is a header line plus a POINTS2D line. That second line is
    # empty for pose-only models and some writers drop it entirely, so pair by
    # content rather than by counting lines.
    lines = [l.split() for l in open(path, encoding="utf-8") if not l.startswith("#")]
    out = []
    for p in lines:
        if not _is_image_line(p):
            continue
        out.append({"cam_id": int(p[8]), "name": " ".join(p[9:]),  # names may contain spaces
                    "q": np.array(list(map(float, p[1:5]))),
                    "t": np.array(list(map(float, p[5:8])))})
    return out


def _quat_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_colmap(sparse_dir: str):
    if os.path.isfile(os.path.join(sparse_dir, "cameras.bin")):
        cams = _bin_cameras(os.path.join(sparse_dir, "cameras.bin"))
        raw = _bin_images(os.path.join(sparse_dir, "images.bin"))
    else:
        cams = _txt_cameras(os.path.join(sparse_dir, "cameras.txt"))
        raw = _txt_images(os.path.join(sparse_dir, "images.txt"))
    imgs = []
    for r in raw:
        R = _quat_R(r["q"])
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ r["t"]
        imgs.append({"cam_id": r["cam_id"], "name": r["name"], "c2w": c2w})
    return cams, imgs


def fx_fy(cam):
    p = cam["params"]
    if cam["model"] in SINGLE_FOCAL:
        return p[0], p[0]
    return p[0], p[1]


def fov_from_cam(cam, max_deg: float | None = None):
    fx, fy = fx_fy(cam)
    fovx = 2 * np.arctan(cam["W"] / (2 * fx))
    fovy = 2 * np.arctan(cam["H"] / (2 * fy))
    if max_deg is not None:
        cap = np.deg2rad(max_deg)
        if fovx > cap:
            fovx = cap
            fovy = 2 * np.arctan(np.tan(cap / 2) * cam["H"] / cam["W"])
    return float(fovx), float(fovy)
