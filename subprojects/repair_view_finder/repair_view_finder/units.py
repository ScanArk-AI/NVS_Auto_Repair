"""Scene units.

Every ``*_m`` parameter in the configs is in metres, but a COLMAP
reconstruction has an arbitrary scale. ``geometry.units_per_metre`` says how
many scene units make one metre: 1.0 for metric captures (e.g. LiDAR-
registered), a measured number when the scale is known, or ``auto`` to
estimate it from the camera height above the floor (see ``scene.load_scene``).

``scene_units`` returns a copy of the config with every metric length
converted, so the algorithm code can work in scene units throughout. Anything
written to the outputs under a ``*_m`` name is converted back to metres.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, OmegaConf

# section -> keys holding a length (scalar or [lo, hi]) in metres.
# geometry.floor_m is a coordinate in scene units, not a length, and is left alone.
METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "trajectories": ("height_m", "dist_m", "clearance_m", "forward_band_m", "forward_far_m",
                     "forward_step_m", "forward_radius_m", "forward_height_band_m", "voxel_m"),
    "geometry": ("height_band_m",),
    "scoring": ("voxel_m", "region_voxel_m", "projection_margin_m"),
}


def scene_units(cfg: DictConfig, units_per_metre: float) -> DictConfig:
    """Copy of ``cfg`` with every metric length multiplied by ``units_per_metre``."""
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for section, keys in METRIC_KEYS.items():
        if section not in out:
            continue
        for key in keys:
            value = out[section].get(key)
            if value is None:
                continue
            if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
                out[section][key] = [float(v) * units_per_metre for v in value]
            else:
                out[section][key] = float(value) * units_per_metre
    return out


def floor_scale_free(xyz, up, cam_centers) -> float:
    """Densest layer below every camera, inside the path footprint (+20 %).

    Needs no notion of scale, which is what makes it usable for estimating
    one: everything is relative to the camera path's own extent.
    """
    e1 = np.array([1.0, 0.0, 0.0]); e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    h_all, h_cam = xyz @ up, cam_centers @ up
    u_all, v_all = xyz @ e1, xyz @ e2
    uc, vc = cam_centers @ e1, cam_centers @ e2
    mu, mv = 0.2 * np.ptp(uc), 0.2 * np.ptp(vc)
    sel = ((u_all > uc.min() - mu) & (u_all < uc.max() + mu)
           & (v_all > vc.min() - mv) & (v_all < vc.max() + mv)
           & (h_all < h_cam.min()))
    if int(sel.sum()) < 100:
        raise SystemExit("geometry.units_per_metre=auto found no geometry below the cameras; "
                         "set geometry.units_per_metre to a number")
    hs = h_all[sel]
    hist, edges = np.histogram(hs, bins=200, range=(np.percentile(hs, 1), h_cam.min()))
    return float(0.5 * (edges[:-1] + edges[1:])[int(np.argmax(hist))])
