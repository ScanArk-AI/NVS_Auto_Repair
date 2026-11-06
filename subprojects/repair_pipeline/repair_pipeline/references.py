"""Pick reference views for a repair trajectory and export them ready to use.

Pipeline per trajectory:

1. read the trajectory's 25 poses (``poses.json``, OpenCV c2w),
2. compute the union "which Gaussians does this trajectory see" mask,
3. rank every registered photo by how many of those Gaussians it shares,
4. drop photos whose dynamic-object mask exceeds ``max_mask_frac``,
5. keep the top ``num_views`` and export each at the target size.

The photo-side masks depend only on the scene, not on the trajectory, so they
are cached -- recomputing them per trajectory is what made 166 trajectories take
70 minutes instead of 10.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import colmap_io, ply_io
from .frustum import gaussian_frustum_masks, select_neighbor_indices
from .masks import mask_fractions
from .poses import CAMERA_CONVENTION_FLIP, c2w_opencv_to_opengl
from .prepare import prepare_reference


@dataclass
class SceneRefs:
    cams: dict
    images: list
    frames_gl: np.ndarray          # (num_photos, 4, 4) OpenGL c2w
    intrinsics: np.ndarray         # (num_photos, 6) w, h, fl_x, fl_y, cx, cy
    mask_frac: list[float | None]
    positions: np.ndarray          # (num_gaussians, 3)


def load_scene(scene_cfg) -> SceneRefs:
    cams, images = colmap_io.read_model(Path(scene_cfg.sparse))
    images = sorted(images, key=lambda im: im.name)
    pts = ply_io.read_points(Path(scene_cfg.ply))
    keep = pts.opacity >= float(scene_cfg.min_opacity)
    positions = pts.centers[keep]

    frames_gl, intrinsics = [], []
    for im in images:
        cam = cams[im.camera_id]
        fx, fy, cx, cy = cam.fx_fy_cx_cy
        frames_gl.append(np.linalg.inv(im.w2c) @ CAMERA_CONVENTION_FLIP)
        intrinsics.append((cam.width, cam.height, fx, fy, cx, cy))
    mask_frac = mask_fractions(Path(scene_cfg.images), [im.name for im in images], scene_cfg.mask_invalid)

    return SceneRefs(
        cams=cams,
        images=images,
        frames_gl=np.stack(frames_gl),
        intrinsics=np.array(intrinsics, dtype=np.float64),
        mask_frac=mask_frac,
        positions=positions,
    )


def _photo_masks(scene: SceneRefs, scene_cfg, cache_dir: Path) -> np.ndarray:
    key = hashlib.sha1(
        f"{Path(scene_cfg.ply).resolve()}|{len(scene.positions)}|{len(scene.images)}|"
        f"{scene_cfg.min_opacity}".encode()
    ).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"photo_masks_{key}.npy"
    if path.is_file():
        masks = np.load(path)
        if masks.shape[0] == len(scene.images):
            return masks
    masks = gaussian_frustum_masks(scene.positions, scene.frames_gl, scene.intrinsics)
    try:
        np.save(path, masks)
    except OSError:
        pass
    return masks


def load_trajectory(traj_dir: Path) -> tuple[np.ndarray, tuple[int, int, float, float, float, float]]:
    """``(c2w_opencv (T,4,4), intrinsics (w,h,fx,fy,cx,cy))`` from poses.json."""
    poses = json.loads((traj_dir / "poses.json").read_text(encoding="utf-8"))
    c2w = np.array(poses["c2w"], dtype=np.float64)
    cam = poses["camera"]
    return c2w, (int(cam["width"]), int(cam["height"]), float(cam["fx"]), float(cam["fy"]),
                 float(cam["cx"]), float(cam["cy"]))


def select_by_coverage(scene: SceneRefs, photo_masks: np.ndarray, idx: np.ndarray, tm, refer_cfg):
    """Greedy direction-aware coverage over the best-overlapping candidates.

    Taking the top ``num_views`` by shared Gaussians picks near-identical
    neighbours on video captures (12 consecutive frames). Here each pick is the
    photo that sees the most (target frame, Gaussian) pairs not yet covered from
    within ``refer.angle_deg`` of the target's ray; when nothing adds coverage
    any more, the photo farthest from those already picked. Candidates are the
    ``refer.candidates`` best by shared Gaussians, which keeps it a few seconds.

    Returns ``(picked photo indices ascending, their overlap, log dict)``.
    """
    from .coverage import DirectionDemand, popcount, union_bits

    num_views = int(refer_cfg.num_views)
    overlap_all = popcount(photo_masks[idx] & tm.union)
    order = np.argsort(-overlap_all, kind="stable")[: max(num_views, int(refer_cfg.get("candidates", 64)))]
    cand = [int(idx[o]) for o in order]
    cand_overlap = {int(idx[o]): int(overlap_all[o]) for o in order}
    dem = DirectionDemand(scene, tm, int(refer_cfg.get("sample_gaussians", 100000)),
                          float(refer_cfg.get("angle_deg", 30.0)), 0)
    bits = {i: dem.covered(scene.frames_gl[i], scene.intrinsics[i]) for i in cand}
    centres = {i: scene.frames_gl[i][:3, 3] for i in cand}
    covered = np.zeros_like(dem.bits)
    picked, gains = [], []
    while len(picked) < num_views and len(picked) < len(cand):
        rest = [i for i in cand if i not in picked]
        g = np.array([popcount(bits[i] & ~covered & dem.bits) for i in rest])
        if g.max() > 0:
            best = rest[int(np.argmax(g))]
        else:  # coverage saturated: spread out instead of taking a neighbour
            best = max(rest, key=lambda i: min(np.linalg.norm(centres[i] - centres[j]) for j in picked))
        picked.append(best); gains.append(int(g.max()))
        covered |= bits[best]
    order_out = np.argsort(picked)
    picked_arr = np.asarray(picked)[order_out]
    log = {"strategy": "coverage", "candidates": len(cand), "angle_deg": float(refer_cfg.get("angle_deg", 30.0)),
           "direction_coverage": dem.fraction(union_bits([bits[i] for i in picked], dem.bits)),
           "direction_coverage_top_overlap": dem.fraction(union_bits([bits[i] for i in cand[:num_views]], dem.bits)),
           "pick_order": [scene.images[i].name for i in picked], "gains": gains}
    return picked_arr, [cand_overlap[int(i)] for i in picked_arr], log


def select_for_trajectory(
    scene: SceneRefs,
    photo_masks: np.ndarray,
    traj_dir: Path,
    refer_cfg,
    scene_cfg,
    out_dir: Path,
) -> dict:
    """Rank, filter, export. Returns the ``selected.json`` payload."""
    c2w_cv, (tw, th, tfx, tfy, tcx, tcy) = load_trajectory(traj_dir)
    target_gl = np.stack([c2w_opencv_to_opengl(m) for m in c2w_cv])
    target_intr = np.tile([tw, th, tfx, tfy, tcx, tcy], (len(target_gl), 1))

    target_masks = gaussian_frustum_masks(scene.positions, target_gl, target_intr)
    union = np.bitwise_or.reduce(target_masks, axis=0)

    # Candidate filter: skip photos whose dynamic-object mask is too large.
    # `None` means the photo has no mask at all and counts as clean.
    eligible = np.array(
        [f is None or f <= float(refer_cfg.max_mask_frac) for f in scene.mask_frac], dtype=bool
    )
    idx = np.flatnonzero(eligible)
    strategy = str(refer_cfg.get("strategy", "coverage"))
    coverage_log = None
    if strategy == "coverage":
        from .coverage import TrajMasks

        tm = TrajMasks(c2w_cv, (tw, th, tfx, tfy, tcx, tcy), target_gl, target_masks, union)
        picked, overlap, coverage_log = select_by_coverage(scene, photo_masks, idx, tm, refer_cfg)
    elif strategy == "overlap":
        picked_local, overlap = select_neighbor_indices(union, photo_masks[idx], int(refer_cfg.num_views))
        picked = idx[picked_local] if picked_local else np.array([], dtype=int)
    else:
        raise ValueError(f"refer.strategy={strategy!r}; expected coverage or overlap")
    if len(picked) < int(refer_cfg.num_views):
        raise RuntimeError(f"{traj_dir.name}: only {len(idx)} eligible photos, cannot pick {refer_cfg.num_views}")

    # The target camera's own vertical FOV defines the perspective the reference
    # must be warped into; config can override it for debugging.
    target_fov = float(refer_cfg.target_fov_deg) if refer_cfg.target_fov_deg else np.degrees(2.0 * np.arctan2(th / 2.0, tfy))
    prepared_dir = out_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for rank, (photo_idx, ov) in enumerate(zip(picked, overlap), start=1):
        im = scene.images[photo_idx]
        cam = scene.cams[im.camera_id]
        arr, post_frac = prepare_reference(
            Path(scene_cfg.images) / im.name, cam, tw, th, float(target_fov),
            images_root=Path(scene_cfg.images), image_name=im.name,
            mask_invalid=scene_cfg.mask_invalid, fill_masked=bool(refer_cfg.fill_masked),
        )
        rel = Path(im.name).with_suffix(".png")
        dst = prepared_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(arr).save(dst)
        selected.append({
            "rank": rank,
            "photo_index": int(photo_idx),
            "name": im.name,
            "overlap": int(ov),
            "overlap_frac": float(ov) / float(len(scene.positions)),
            "mask_frac": scene.mask_frac[photo_idx],
            "prepared": dst.relative_to(out_dir).as_posix(),
            "prepared_mask_frac": post_frac,
        })

    payload = {
        "trajectory": traj_dir.name,
        "frames": int(len(c2w_cv)),
        "num_gaussians": int(len(scene.positions)),
        "num_views": len(selected),
        "eligible_photos": int(len(idx)),
        "selection": coverage_log or {"strategy": "overlap"},
        "config": {
            "num_views": int(refer_cfg.num_views),
            "strategy": strategy,
            "max_mask_frac": float(refer_cfg.max_mask_frac),
            "mask_invalid": str(scene_cfg.mask_invalid),
            "fill_masked": bool(refer_cfg.fill_masked),
            "target_fov_deg": float(target_fov),
        },
        "selected": selected,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload
