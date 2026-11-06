"""Per-view UQ -> scene voxels -> defect regions."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as nd_label

from .colmap_io import fov_from_cam


def project_points(pose, xyz, fx, fy, W, H, near=0.05):
    R, c = pose[:3, :3], pose[:3, 3]
    pc = (xyz - c) @ R
    z = pc[:, 2]
    ok = z > near
    safe = np.where(ok, z, 1.0)
    u = fx * pc[:, 0] / safe + W / 2.0
    v = fy * pc[:, 1] / safe + H / 2.0
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return ok, np.stack([u, v], axis=1)


class DefectVolume:
    def __init__(self, scene, cfg):
        self.cfg = cfg
        self.units_per_metre = float(scene.units_per_metre)
        self.max_fov = float(cfg.views.max_fov_deg)
        self.voxel = float(cfg.scoring.voxel_m)
        self.thr = float(cfg.scoring.uq_excess)
        self.W = int(cfg.views.width)
        self.H = int(cfg.views.height)
        max_pts = int(cfg.scoring.max_projection_points)
        self.xyz = scene.xyz
        # Keep only points in a box around the camera footprint. The PLY can
        # contain far floaters; projecting them would create defect voxels
        # hundreds of metres away.
        cs = np.array([im["c2w"][:3, 3] for im in scene.imgs])
        margin = float(cfg.scoring.projection_margin_m)
        keep = np.all((self.xyz >= cs.min(0) - margin)
                      & (self.xyz <= cs.max(0) + margin), axis=1)
        pts = self.xyz[keep]
        self.sub = pts[:: max(1, len(pts) // max_pts)]
        k = np.floor(self.sub / self.voxel).astype(np.int64)
        # stable 1-D ids
        span = 1 << 20
        self.vk = (k[:, 0] + span) * (2 * span) ** 2 + (k[:, 1] + span) * (2 * span) + (k[:, 2] + span)
        self.uniq, self.inv = np.unique(self.vk, return_inverse=True)
        self.n_vox = len(self.uniq)
        self.vox_xyz = np.zeros((self.n_vox, 3))
        np.add.at(self.vox_xyz, self.inv, self.sub)
        self.vox_xyz /= np.maximum(np.bincount(self.inv)[:, None], 1)

        h = self.vox_xyz @ scene.up
        self.allowed = np.ones(self.n_vox, dtype=bool)
        band = cfg.geometry.get("height_band_m")
        if band is not None and scene.floor is not None:
            rel = h - scene.floor
            self.allowed = (rel >= float(band[0])) & (rel <= float(band[1]))
        self.vox_max = np.full(self.n_vox, -np.inf)
        self.view_bad = {}
        self.view_val = {}

    def add_view(self, cand_idx, pose, cam, excess):
        fovx, fovy = fov_from_cam(cam, max_deg=self.max_fov)
        fx = self.W / (2 * np.tan(fovx / 2))
        fy = self.H / (2 * np.tan(fovy / 2))
        inside, uv = project_points(pose, self.sub, fx, fy, self.W, self.H,
                                    near=0.05 * float(self.units_per_metre))
        vv = np.full(self.n_vox, -np.inf)
        if inside.any():
            pix = uv[inside]
            px = np.clip(pix[:, 0].astype(int), 0, self.W - 1)
            py = np.clip(pix[:, 1].astype(int), 0, self.H - 1)
            vals = excess[py, px]
            np.maximum.at(vv, self.inv[inside], vals)
        vv[~self.allowed] = -np.inf
        bad = np.where(vv > self.thr)[0]
        self.view_bad[cand_idx] = bad
        self.view_val[cand_idx] = vv[bad]
        np.maximum.at(self.vox_max, bad, vv[bad])
        return bad

    def finalize(self):
        defect = np.where((self.vox_max > self.thr) & self.allowed)[0]
        self.defect_vox = defect
        self.regions = []
        self.region_of_vox = np.full(self.n_vox, -1, dtype=np.int32)
        if defect.size == 0:
            return
        # cluster on a coarser region grid (default 0.6 m) so the output is
        # a handful of reviewable regions instead of thousands of fragments
        rv = float(self.cfg.scoring.region_voxel_m)
        coords = self.vox_xyz[defect]
        lo = coords.min(axis=0) - rv
        idx = np.floor((coords - lo) / rv).astype(np.int64)
        shape = tuple(idx.max(axis=0) + 2)
        grid = np.zeros(shape, dtype=bool)
        grid[tuple(idx.T)] = True
        lab_grid, ncomp = nd_label(grid, structure=np.ones((3, 3, 3), dtype=int))
        lab = lab_grid[tuple(idx.T)]
        for rid in range(1, ncomp + 1):
            members = np.where(lab == rid)[0]
            if members.size == 0:
                continue
            gidx = defect[members]
            ridx = int(rid - 1)
            self.region_of_vox[gidx] = ridx
            self.regions.append({
                "id": ridx,
                "voxels": int(len(gidx)),
                "region_voxel_m": rv,
                "centroid": self.vox_xyz[gidx].mean(axis=0).tolist(),
                "bbox_min": self.vox_xyz[gidx].min(axis=0).tolist(),
                "bbox_max": self.vox_xyz[gidx].max(axis=0).tolist(),
                "max_uq_excess": float(self.vox_max[gidx].max()),
                "mean_uq_excess": float(self.vox_max[gidx].mean()),
            })

    def regions_seen(self, cand_idx):
        out = {}
        for v, val in zip(self.view_bad.get(cand_idx, []),
                          self.view_val.get(cand_idx, [])):
            rid = int(self.region_of_vox[v])
            if rid < 0:
                continue
            out[rid] = max(out.get(rid, -np.inf), float(val))
        return out
