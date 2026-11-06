"""Camera pose conventions.

Getting this wrong is the single most damaging mistake available: a wrong
convention still produces a plausible-looking room, just mirrored and
depth-reversed.

**OpenCV** -- x right, y down, z forward. This is COLMAP's convention, what
``poses.json`` carries, and the tool's internal convention.

**OpenGL** -- x right, y up, z backward. NeRFStudio and therefore ArtiFixer's
``transforms.json`` use it. The two differ by a 180 degree rotation about the
camera x axis::

    c2w_opengl = c2w_opencv @ FLIP
    w2c_opencv = FLIP @ c2w_opengl^-1

which is ``CAMERA_CONVENTION_FLIP`` in ArtiFixer's
``data_processing/camera_trajectories.py``.
"""

from __future__ import annotations

import numpy as np

CAMERA_CONVENTION_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def as_4x4(matrix) -> np.ndarray:
    """Accept a 3x4 or 4x4 matrix and return a 4x4 float64."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (3, 4):
        m = np.vstack([m, [0.0, 0.0, 0.0, 1.0]])
    if m.shape != (4, 4):
        raise ValueError(f"expected a 3x4 or 4x4 matrix, got {m.shape}")
    return m


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """(w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ]
    )


def c2w_opencv_to_opengl(c2w_opencv) -> np.ndarray:
    return as_4x4(c2w_opencv) @ CAMERA_CONVENTION_FLIP
