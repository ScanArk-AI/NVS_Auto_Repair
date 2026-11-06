"""Scene units (repair_view_finder/units.py): config conversion and the
scale-free floor estimate that ``units_per_metre: auto`` is built on."""

import numpy as np
import pytest
from omegaconf import OmegaConf


def test_scene_units_scales_every_metric_length():
    from repair_view_finder.units import METRIC_KEYS, scene_units

    cfg = OmegaConf.create({
        "views": {"width": 640},
        "trajectories": {"height_m": 0.4, "dist_m": [0.2, 1.5], "frames": 25, "iou": [0.08, 0.7]},
        "geometry": {"height_band_m": [0.0, 1.5], "floor_m": -0.12},
        "scoring": {"voxel_m": 0.1, "region_voxel_m": None, "uq_excess": 1.0},
    })
    out = scene_units(cfg, 3.7)
    assert out.trajectories.height_m == pytest.approx(1.48)
    assert list(out.trajectories.dist_m) == pytest.approx([0.74, 5.55])
    assert list(out.geometry.height_band_m) == pytest.approx([0.0, 5.55])
    assert out.scoring.voxel_m == pytest.approx(0.37)
    assert out.scoring.region_voxel_m is None           # unset stays unset
    # not lengths: untouched
    assert out.trajectories.frames == 25 and list(out.trajectories.iou) == [0.08, 0.7]
    assert out.views.width == 640 and out.scoring.uq_excess == 1.0
    assert out.geometry.floor_m == -0.12                # a coordinate, already in scene units
    assert cfg.trajectories.height_m == 0.4             # input not mutated
    assert "height_m" in METRIC_KEYS["trajectories"]


def _room(scale, rng):
    """Floor at z=0, walls, a camera path at ~1.3 m; everything times ``scale``."""
    n = 20000
    floor = np.c_[rng.uniform(0, 6, n), rng.uniform(0, 4, n), rng.normal(0, 0.005, n)]
    wall = np.c_[rng.uniform(0, 6, n // 2), np.full(n // 2, 4.0), rng.uniform(0, 2.6, n // 2)]
    clutter = np.c_[rng.uniform(0, 6, n // 4), rng.uniform(0, 4, n // 4), rng.uniform(0.3, 2.5, n // 4)]
    t = np.linspace(0, 1, 200)
    cams = np.c_[1 + 4 * t, 1 + 2 * np.sin(3 * t), 1.3 + 0.1 * np.sin(9 * t)]
    return np.vstack([floor, wall, clutter]) * scale, cams * scale


@pytest.mark.parametrize("scale", [1.0, 3.7, 0.05])
def test_floor_scale_free_is_scale_equivariant(scale):
    from repair_view_finder.units import floor_scale_free

    xyz, cams = _room(scale, np.random.default_rng(0))
    up = np.array([0.0, 0.0, 1.0])
    floor = floor_scale_free(xyz, up, cams)
    assert floor == pytest.approx(0.0, abs=0.03 * scale)
    # what `auto` divides by capture_height_m: the median camera height
    assert np.median(cams @ up) - floor == pytest.approx(1.3 * scale, rel=0.05)
