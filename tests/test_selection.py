"""Reference-view selection: frustum test, ranking, prepared intrinsics, progressive picks."""

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf


def _bits(mask_row):
    return np.unpackbits(mask_row)


def test_frustum_in_front_behind_outside():
    from repair_pipeline.frustum import gaussian_frustum_masks

    pts = np.array([[0, 0, -2.0], [0, 0, 2.0], [100.0, 0, -1.0]])   # OpenGL camera looks down -z
    masks = gaussian_frustum_masks(pts, np.eye(4)[None], np.array([[960, 544, 500, 500, 480, 272]]))
    assert _bits(masks[0])[:3].tolist() == [1, 0, 0]


def test_select_neighbor_indices_ranks_by_shared_points():
    from repair_pipeline.frustum import select_neighbor_indices

    target = np.packbits(np.array([1] * 8 + [0] * 8, bool))
    photos = np.stack([np.packbits(np.array(row, bool)) for row in (
        [1] * 2 + [0] * 14,          # shares 2
        [1] * 8 + [1] * 8,           # shares 8
        [0] * 16,                    # shares 0
        [1] * 5 + [0] * 11,          # shares 5
    )])
    picked, overlap = select_neighbor_indices(target, photos, 2)
    assert picked == [1, 3] and overlap == [8, 5]           # ascending indices, overlaps parallel
    assert select_neighbor_indices(target, photos, 5) == ([], [])   # not enough candidates


def _prepared_intrinsics():
    from repair_pipeline import prepare, progressive

    fn = getattr(prepare, "prepared_intrinsics", None) or getattr(progressive, "prepared_intrinsics", None)
    if fn is None:
        pytest.skip("prepared_intrinsics not available")
    return fn


@pytest.mark.parametrize("model_id, params, expected_fy_over_h", [
    # A real capture rig: a 4000x3000 OPENCV_FISHEYE and a 4000x3000 OPENCV camera,
    # prepared for a 960x544 target at 75 deg vertical FOV. Values were checked
    # against photo-consistency of projected Gaussians on the real images.
    (5, [786.614, 786.509, 2001.821, 1498.162, 0.079, -0.008, -0.006, 0.0], 0.407),
    (4, [1922.491, 1923.704, 1925.361, 1563.439, -0.042, -0.019, 0.0, 0.0], 0.849),
])
def test_prepared_intrinsics_follow_the_prepared_image(model_id, params, expected_fy_over_h):
    from repair_pipeline.colmap_io import Camera

    fn = _prepared_intrinsics()
    cam = Camera(id=1, model_id=model_id, width=4000, height=3000, params=np.array(params))
    intr = fn(cam, 960, 544, 75.0)
    assert (intr["w"], intr["h"]) == (960, 544)
    assert intr["fl_y"] / intr["h"] == pytest.approx(expected_fy_over_h, abs=0.002)
    assert 0 < intr["cx"] < 960 and 0 < intr["cy"] < 544


def test_progressive_greedy_prefers_residual_and_stops_on_no_gain():
    from repair_pipeline.progressive import _greedy

    def bits(on):
        row = np.zeros(32, bool)
        row[list(on)] = True
        return np.packbits(row)

    demand = bits(range(32))
    covered = bits(range(16))                         # the photos already cover half
    frame = lambda traj: SimpleNamespace(traj=traj)
    a, b, c, d = frame("t1"), frame("t1"), frame("t2"), frame("t3")
    cands = [
        (a, bits(range(16, 24)), 0),                  # +8 new
        (b, bits(range(16, 32)), 0),                  # +16 new
        (c, bits(range(0, 16)), 0),                   # nothing new
        (d, bits(range(28, 32)), 0),                  # +4, but b already covers it
    ]
    prog = OmegaConf.create({"max_generated": 4, "per_traj_max": 2, "min_gain_frac": 0.01})
    picks = _greedy(demand, covered, cands, prog)
    assert [p[0] for p in picks] == [b]               # b first; afterwards nobody adds anything
    assert picks[0][2] == 16

    prog.per_traj_max = 1                             # one frame per trajectory
    picks = _greedy(demand, covered, [cands[0], cands[1], cands[3]], prog)
    assert [p[0] for p in picks] == [b]               # a is from t1 too, d adds nothing
