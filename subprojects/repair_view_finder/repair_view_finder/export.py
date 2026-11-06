"""Standardised machine-readable outputs for the downstream repair pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def camera_dict(W, H, fovx, fovy):
    fx = W / (2 * np.tan(fovx / 2))
    fy = H / (2 * np.tan(fovy / 2))
    return {"model": "PINHOLE", "width": int(W), "height": int(H),
            "fx": float(fx), "fy": float(fy),
            "cx": float(W) / 2.0, "cy": float(H) / 2.0}


def pose_payload(name, c2ws, W, H, fovx, fovy, frame_names=None):
    c2ws = [np.asarray(m, dtype=np.float64) for m in c2ws]
    w2cs = [np.linalg.inv(m) for m in c2ws]
    names = frame_names or [f"{i:05d}.png" for i in range(len(c2ws))]
    return {
        "trajectory": name,
        "frames": len(c2ws),
        "camera": camera_dict(W, H, fovx, fovy),
        "convention": "w2c/c2w are 4x4 OpenCV (x right, y down, z forward)",
        "w2c": [m.tolist() for m in w2cs],
        "c2w": [m.tolist() for m in c2ws],
        "frame_names": names,
    }


def transforms_payload(c2ws, W, H, fovx, fovy, file_paths):
    c = camera_dict(W, H, fovx, fovy)
    return {
        "camera_model": "OPENCV",
        "fl_x": c["fx"], "fl_y": c["fy"], "cx": c["cx"], "cy": c["cy"],
        "w": c["width"], "h": c["height"],
        "frames": [{"file_path": fp, "transform_matrix": np.asarray(m).tolist()}
                   for fp, m in zip(file_paths, c2ws)],
    }


def write_selected_views(root, entries):
    """Index of the selected single-frame views (poses + scores only)."""
    payload = {"convention": "c2w/w2c are 4x4 OpenCV (x right, y down, z forward)",
               "views": entries}
    write_json(Path(root) / "views.json", payload)
    write_json(Path(root) / "poses.json", payload)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_trajectory(root, name, c2ws, rgb_frames, W, H, fovx, fovy, meta):
    """Repair-ready clip: frames/ + poses.json + transforms.json + meta.json."""
    from PIL import Image
    d = Path(root) / "trajectories" / name
    fdir = d / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, fr in enumerate(rgb_frames):
        fn = f"{i:05d}.png"
        Image.fromarray(fr).save(fdir / fn)
        names.append(fn)
    write_json(d / "poses.json",
               pose_payload(name, c2ws, W, H, fovx, fovy, names))
    write_json(d / "transforms.json",
               transforms_payload(c2ws, W, H, fovx, fovy,
                                  [f"frames/{n}" for n in names]))
    write_json(d / "meta.json", meta)
    return d
