"""Convert a nerfstudio ``transforms.json`` into a COLMAP text model.

    python tools/nerfstudio_to_colmap.py <transforms.json> <out_dir>
        [--dataparser-transforms <dataparser_transforms.json>]
        [--keep-applied-transform]

Both stages read COLMAP models, so this is all a nerfstudio-format capture
needs. ``out_dir`` gets ``cameras.txt``, ``images.txt`` and an empty
``points3D.txt``; image names are ``file_path`` relative to the folder holding
transforms.json, so point ``scene.images`` (and the finder's image root) there.

**The cameras must end up in the frame of the 3DGS you repair.**

* A splat trained by nerfstudio (splatfacto) lives in the dataparser frame:
  pass the run's ``dataparser_transforms.json`` and the cameras are moved into
  it (rotation/translation, then ``scale``).
* Otherwise the poses are returned to the frame the capture was in: the
  ``applied_transform`` that ``ns-process-data`` adds when converting from
  COLMAP is undone (``--keep-applied-transform`` keeps it).

Note that a dataparser ``scale`` makes the scene units non-metric.

Camera models: PINHOLE, OPENCV (k1 k2 p1 p2) and OPENCV_FISHEYE (k1..k4);
intrinsics may be global or per frame. Frames sharing intrinsics share a
COLMAP camera.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_DIST = {"PINHOLE": (), "OPENCV": ("k1", "k2", "p1", "p2"), "OPENCV_FISHEYE": ("k1", "k2", "k3", "k4")}
# Flip of the camera's y and z axes: OpenGL (nerfstudio) <-> OpenCV (COLMAP).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _as4(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    if m.shape == (3, 4):
        m = np.vstack([m, [0.0, 0.0, 0.0, 1.0]])
    if m.shape != (4, 4):
        raise ValueError(f"expected a 3x4 or 4x4 matrix, got {m.shape}")
    return m


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP's (w, x, y, z), w >= 0 -- same as COLMAP's ``rotmat2qvec``."""
    # COLMAP unpacks R.flat as Rxx, Ryx, Rzx, Rxy, ... i.e. its names are the
    # transpose of the indices; spelled out with indices here.
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -q if q[0] < 0 else q


def _camera(meta: dict, frame: dict) -> tuple:
    """``(model, w, h, params...)`` with per-frame keys overriding global ones."""
    get = lambda k, d=None: frame.get(k, meta.get(k, d))  # noqa: E731
    model = str(get("camera_model", "OPENCV")).upper()
    if model == "SIMPLE_PINHOLE":
        model = "PINHOLE"
    if model not in _DIST:
        raise ValueError(f"camera_model {model!r} is not supported (PINHOLE, OPENCV, OPENCV_FISHEYE)")
    w, h = get("w"), get("h")
    fx, fy, cx, cy = get("fl_x"), get("fl_y", get("fl_x")), get("cx"), get("cy")
    if None in (w, h, fx, cx, cy):
        raise ValueError(f"{frame.get('file_path')}: missing intrinsics (w, h, fl_x, cx, cy)")
    dist = [float(get(k, 0.0)) for k in _DIST[model]]
    if model == "OPENCV" and not any(dist):
        model, dist = "PINHOLE", []
    return (model, int(w), int(h), float(fx), float(fy), float(cx), float(cy), *dist)


def convert(transforms_json: Path, out_dir: Path, dataparser_json: Path | None = None,
            keep_applied: bool = False) -> dict:
    meta = json.loads(Path(transforms_json).read_text())
    frames = meta.get("frames") or []
    if not frames:
        raise ValueError(f"{transforms_json}: no frames")

    to_frame = np.eye(4)
    if dataparser_json is not None:
        dp = json.loads(Path(dataparser_json).read_text())
        T, scale = _as4(dp["transform"]), float(dp.get("scale", 1.0))
        to_frame = np.diag([scale, scale, scale, 1.0]) @ T
    elif "applied_transform" in meta and not keep_applied:
        to_frame = np.linalg.inv(_as4(meta["applied_transform"]))

    cameras: dict[tuple, int] = {}
    lines = []
    for image_id, fr in enumerate(sorted(frames, key=lambda f: f["file_path"]), start=1):
        cam = _camera(meta, fr)
        cam_id = cameras.setdefault(cam, len(cameras) + 1)
        c2w = to_frame @ _as4(fr["transform_matrix"])
        # Rotation part may carry the dataparser scale; strip it so R stays orthonormal.
        R = c2w[:3, :3]
        s = np.cbrt(np.linalg.det(R))
        c2w_cv = np.eye(4)
        c2w_cv[:3, :3] = (R / s) @ _GL_TO_CV[:3, :3]
        c2w_cv[:3, 3] = c2w[:3, 3]
        w2c = np.linalg.inv(c2w_cv)
        q, t = _rotmat_to_qvec(w2c[:3, :3]), w2c[:3, 3]
        name = str(fr["file_path"]).removeprefix("./")
        lines.append(f"{image_id} {' '.join(f'{v:.12g}' for v in (*q, *t))} {cam_id} {name}\n\n")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for cam, cid in cameras.items():
            f.write(f"{cid} {cam[0]} {cam[1]} {cam[2]} {' '.join(f'{v:.12g}' for v in cam[3:])}\n")
    with open(out_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# (no 2-D observations)\n")
        f.writelines(lines)
    (out_dir / "points3D.txt").write_text("# no 3-D points\n")
    return {"images": len(lines), "cameras": len(cameras),
            "frame": "dataparser" if dataparser_json else ("as-is" if keep_applied else "capture")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("transforms_json", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--dataparser-transforms", type=Path, default=None,
                    help="nerfstudio run's dataparser_transforms.json; puts the cameras in a splatfacto splat's frame")
    ap.add_argument("--keep-applied-transform", action="store_true",
                    help="leave transforms.json's applied_transform in place")
    a = ap.parse_args(argv)
    info = convert(a.transforms_json, a.out_dir, a.dataparser_transforms, a.keep_applied_transform)
    print(f"{info['images']} images, {info['cameras']} camera(s), {info['frame']} frame -> {a.out_dir}")
    print(f"scene config:  sparse: {a.out_dir}   images: {a.transforms_json.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
