"""GAVIS visibility field + UQ/RGB rendering for candidate views."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from .gavis_compat import GAVIS, pose_to_camera, render

from .colmap_io import fov_from_cam


def to_img(t):
    return (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


class UQEngine:
    def __init__(self, cfg, scene, cam_id=None, render_model=None):
        self.cfg = cfg
        self.scene = scene
        by_cam = {}
        for im in scene.imgs:
            by_cam.setdefault(im["cam_id"], []).append(im)
        fid = cam_id if cam_id is not None else cfg.field.get("cam_id")
        if fid is None:
            fid = max(by_cam, key=lambda c: len(by_cam[c]))
        self.field_cid = int(fid)
        lst = sorted(by_cam[self.field_cid], key=lambda x: x["name"])
        step = max(1, len(lst) // max(1, int(cfg.field.views)))
        self.field = lst[::step][: int(cfg.field.views)]
        self.field_cam = scene.cams[self.field_cid]

        self.max_fov = float(cfg.views.max_fov_deg)
        fovx, fovy = fov_from_cam(self.field_cam, max_deg=self.max_fov)
        self.base_cfg = OmegaConf.create({
            "H": int(cfg.views.height), "W": int(cfg.views.width),
            "fovx": fovx, "fovy": fovy,
            "aabb": np.asarray(scene.aabb).tolist(),
            "vis": {"fixed_gaussian": False,
                    "fixed_gaussian_use_angular_vis": False},
        })
        self.gavis = GAVIS(self.base_cfg)
        self.gavis.load_model(scene.model, [im["c2w"] for im in self.field])
        # UQ/visibility uses the subsampled model; RGB uses ``render_model``
        # (the full PLY, loaded once by the caller and shared by every
        # per-camera engine) or, without one, the same subsampled model.
        self.render_model = render_model if render_model is not None else scene.model
        self.device = str(cfg.device)
        self.pipe = SimpleNamespace(convert_SHs_python=False,
                                    compute_cov3D_python=False, debug=False)
        bg = cfg.output.bg_color
        self.white = torch.tensor(bg, dtype=torch.float32, device=self.device)

    def render_pose(self, cam_info, pose, name="view"):
        W, H = int(self.cfg.views.width), int(self.cfg.views.height)
        fovx, fovy = fov_from_cam(cam_info, max_deg=self.max_fov)
        self.gavis.H, self.gavis.W = H, W
        self.gavis.fovx, self.gavis.fovy = fovx, fovy
        with torch.no_grad():
            uq = self.gavis.render_uncertainty_map(pose).numpy()
            rgb = self.render_rgb(cam_info, pose, name)
        return uq, rgb

    def render_uq(self, cam_info, pose, size=None):
        """UQ only — used when scoring synthesised trajectory anchors."""
        W, H = size or (int(self.cfg.views.width), int(self.cfg.views.height))
        fovx, fovy = fov_from_cam(cam_info, max_deg=self.max_fov)
        self.gavis.H, self.gavis.W = int(H), int(W)
        self.gavis.fovx, self.gavis.fovy = fovx, fovy
        with torch.no_grad():
            return self.gavis.render_uncertainty_map(pose).numpy()

    def render_rgb(self, cam_info, pose, name="view", size=None, fovy_deg=None):
        """RGB only (no UQ) — used for the repair-video frames."""
        W, H = size or (int(self.cfg.views.width), int(self.cfg.views.height))
        W, H = int(W), int(H)
        if fovy_deg is None:
            fovx, fovy = fov_from_cam(cam_info, max_deg=self.max_fov)
        else:
            fovy = float(np.deg2rad(fovy_deg))
            fovx = 2.0 * np.arctan(np.tan(fovy / 2.0) * W / H)
        with torch.no_grad():
            cam = pose_to_camera(pose, H, W, fovx, fovy,
                                 name=name, device=self.device)
            return to_img(render(cam, self.render_model, self.pipe,
                                 self.white)["render"])

    def render(self, cand):
        return self.render_pose(cand.cam, cand.pose, cand.name)
