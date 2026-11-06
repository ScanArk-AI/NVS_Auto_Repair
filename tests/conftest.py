"""CPU-only tests. GPU paths (GAVIS rendering, ArtiFixer inference) are covered
by the smoke runs described in the README, not here."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "subprojects" / "repair_pipeline", ROOT / "subprojects" / "repair_view_finder",
          ROOT / "windows"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _qvec(R):
    """Rotation matrix -> COLMAP (qw, qx, qy, qz)."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return w, x, y, z


_CAMERAS = [  # id, model name, model id, w, h, params
    (1, "PINHOLE", 1, 960, 544, [500, 510, 480, 272]),
    (2, "OPENCV", 4, 4000, 3000, [1922.5, 1923.7, 1925.4, 1563.4, -0.042, -0.019, 0, 0]),
]
_IMAGES = [  # name, camera id, euler xyz (deg), camera centre
    ("a.jpg", 1, (0, 0, 0), (0, 0, 0)),
    ("b.jpg", 1, (0, 30, 0), (1.0, 0, 0.5)),
    ("c/d.jpg", 2, (10, -20, 5), (-0.3, 0.2, 2.0)),
]


def _write_text(d, rows):
    (d / "cameras.txt").write_text("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + "".join(
        f"{cid} {name} {w} {h} {' '.join(str(v) for v in params)}\n" for cid, name, _, w, h, params in _CAMERAS))
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for i, (name, cam, q, t) in enumerate(rows, start=1):
        lines += [f"{i} {' '.join(str(v) for v in q)} {' '.join(str(v) for v in t)} {cam} {name}", ""]
    (d / "images.txt").write_text("\n".join(lines) + "\n")


def _write_binary(d, rows):
    import struct

    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(_CAMERAS)))
        for cid, _, model_id, w, h, params in _CAMERAS:
            f.write(struct.pack("<iiQQ", cid, model_id, w, h) + struct.pack(f"<{len(params)}d", *params))
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(rows)))
        for i, (name, cam, q, t) in enumerate(rows, start=1):
            f.write(struct.pack("<i7di", i, *q, *t, cam) + name.encode() + b"\0" + struct.pack("<Q", 0))


@pytest.fixture(params=["txt", "bin"])
def colmap_model(request, tmp_path):
    """A two-camera, three-image COLMAP model in text or binary form; returns (dir, c2w per image name)."""
    from scipy.spatial.transform import Rotation

    d = tmp_path / "sparse" / "0"
    d.mkdir(parents=True)
    c2ws, rows = {}, []
    for name, cam, euler, centre in _IMAGES:
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        c2w[:3, 3] = centre
        w2c = np.linalg.inv(c2w)
        rows.append((name, cam, _qvec(w2c[:3, :3]), tuple(w2c[:3, 3])))
        c2ws[name] = c2w
    (_write_text if request.param == "txt" else _write_binary)(d, rows)
    return d, c2ws
