"""Scene loading: 3DGS model, gravity, optional floor, plane basis."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .gavis_compat import GaussianModel

from .colmap_io import find_sparse_model, read_colmap
from .units import floor_scale_free


def find_ply(scene: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(scene, explicit)
    for p in ("aligned.ply", "source.ply",
              "output/point_cloud/point_cloud.ply",
              "point_cloud/iteration_30000/point_cloud.ply",
              "point_cloud/point_cloud.ply"):
        cand = os.path.join(scene, p)
        if os.path.isfile(cand):
            return cand
    for dirpath, _dirs, files in os.walk(scene):
        for f in files:
            if f.endswith(".ply") and "point_cloud" in f:
                return os.path.join(dirpath, f)
    raise SystemExit(f"no 3DGS ply under {scene}")


def detect_sh_degree(ply: str) -> int:
    from plyfile import PlyData
    names = set(PlyData.read(ply)["vertex"].data.dtype.names or ())
    n_rest = len([n for n in names if n.startswith("f_rest_")])
    if n_rest == 0:
        return 0
    return max(0, int(round(((n_rest / 3) + 1) ** 0.5)) - 1)


def load_model(ply: str, max_gaussians: int = 0, seed: int = 0):
    degree = detect_sh_degree(ply)
    model = GaussianModel(sh_degree=degree)
    model.load_ply(ply)
    n = model.get_xyz.shape[0]
    if max_gaussians and n > max_gaussians:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, max_gaussians, replace=False))
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling",
                     "_rotation", "_opacity"):
            v = getattr(model, name)
            setattr(model, name, torch.nn.Parameter(v[idx], requires_grad=True))
        model.max_radii2D = torch.zeros(model.get_xyz.shape[0], device="cuda")
    return model, degree


def estimate_up(imgs) -> np.ndarray:
    Rs = np.stack([np.linalg.inv(im["c2w"])[:3, :3] for im in imgs])
    up = -Rs[:, 1, :].mean(0)
    return up / np.linalg.norm(up)


def estimate_floor(xyz, up, cam_centers, units_per_metre: float = 1.0):
    """Dense layer near the camera path, not a global peak.

    The search window (4 m around the path, 2 m below to 0.5 m above the
    lowest camera, 3 cm layer) is in metres and scaled to scene units.
    """
    s = float(units_per_metre)
    e1 = np.array([1.0, 0.0, 0.0]); e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    h_all = xyz @ up
    h_cam = cam_centers @ up
    u_all, v_all = xyz @ e1, xyz @ e2
    uc, vc = cam_centers @ e1, cam_centers @ e2
    band = ((u_all > uc.min() - 4 * s) & (u_all < uc.max() + 4 * s)
            & (v_all > vc.min() - 4 * s) & (v_all < vc.max() + 4 * s)
            & (h_all > h_cam.min() - 2.0 * s) & (h_all < h_cam.min() + 0.5 * s))
    hist, edges = np.histogram(h_all[band], bins=300)
    peak = 0.5 * (edges[:-1] + edges[1:])[int(np.argmax(hist))]
    return float(h_all[band & (np.abs(h_all - peak) < 0.03 * s)].mean())



def level_pose(c2w, up):
    fwd = c2w[:3, 2]
    fwd = fwd - (fwd @ up) * up
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return c2w.copy()
    fwd /= n
    x = np.cross(fwd, up); x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    p = c2w.copy()
    p[:3, 0], p[:3, 1], p[:3, 2] = x, y, fwd
    return p


def pitch_pose(c2w_level, up, pitch_rad):
    fwd = c2w_level[:3, 2]
    fwd = fwd - (fwd @ up) * up
    n = np.linalg.norm(fwd)
    if n < 1e-6:
        return c2w_level.copy()
    fwd /= n
    fwd = np.cos(pitch_rad) * fwd + np.sin(pitch_rad) * up
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up); x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    p = c2w_level.copy()
    p[:3, 0], p[:3, 1], p[:3, 2] = x, y, fwd
    return p


@dataclass
class Scene:
    root: str
    ply: str
    sparse: str
    model: object
    sh_degree: int
    cams: dict
    imgs: list
    xyz: np.ndarray
    up: np.ndarray
    floor: float | None
    units_per_metre: float     # scene units per metre (1.0 = metric)
    units_source: str          # "config" or "auto"
    camera_height: float | None  # median camera height above the floor, scene units
    e1: np.ndarray
    e2: np.ndarray
    aabb: np.ndarray


def load_scene(cfg) -> Scene:
    root = os.path.abspath(cfg.scene.root)
    ply = find_ply(root, cfg.scene.get("ply"))
    sparse = find_sparse_model(root, cfg.scene.get("sparse"))
    cams, imgs = read_colmap(sparse)
    model, degree = load_model(ply, int(cfg.model.max_gaussians),
                               int(cfg.model.seed))
    xyz = model.get_xyz.detach().cpu().numpy()

    up = cfg.geometry.get("up")
    up = (np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
          if up is not None else estimate_up(imgs))
    cs = np.array([im["c2w"][:3, 3] for im in imgs])

    # Scale first: the floor search window below is in metres.
    upm = cfg.geometry.units_per_metre
    if str(upm).lower() == "auto":
        cam_h = float(np.median(cs @ up) - floor_scale_free(xyz, up, cs))
        if not cam_h > 0:
            raise SystemExit("geometry.units_per_metre=auto: cameras are not above the floor "
                             "it found; set geometry.units_per_metre to a number")
        upm, units_source = cam_h / float(cfg.geometry.capture_height_m), "auto"
    else:
        upm, units_source = float(upm), "config"
        if not upm > 0:
            raise SystemExit(f"geometry.units_per_metre must be > 0 or auto, got {upm}")

    floor = cfg.geometry.get("floor_m")
    need_floor = (floor is None and (
        cfg.geometry.get("height_band_m") is not None
        or cfg.trajectories.get("height_m") is not None))
    if floor is not None:
        floor = float(floor)
    elif need_floor:
        floor = estimate_floor(xyz, up, cs, upm)
    else:
        floor = None

    e1 = np.array([1.0, 0.0, 0.0]); e1 = e1 - (e1 @ up) * up
    e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    lo, hi = np.percentile(xyz, [0.5, 99.5], axis=0)
    ext = hi - lo
    aabb = np.stack([lo - 0.05 * ext, hi + 0.05 * ext], axis=1)
    camera_height = float(np.median(cs @ up) - floor) if floor is not None else None
    return Scene(root, ply, sparse, model, degree, cams, imgs, xyz, up,
                 floor, upm, units_source, camera_height, e1, e2, aabb)
