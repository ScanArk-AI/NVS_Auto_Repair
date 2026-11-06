"""Repair-trajectory selection.

Every SfM frame of a camera is lowered to a ground-robot height, levelled
(optionally pitched), and frame pairs are turned into 25-frame trajectories
when they are close in time, far enough apart in space, overlap in view, keep
clearance from the scene, and pass through uncertain space.  This is the only
trajectory-finding path.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .colmap_io import fov_from_cam
from .export import write_trajectory
from .scene import level_pose, pitch_pose
from . import viz


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(dot, -1, 1))
    th = th0 * t
    return (np.sin(th0 - th) * q0 + np.sin(th) * q1) / np.sin(th0)


def view_mask(points, pose, fx, fy, W, H, near=0.05):
    R, t = pose[:3, :3], pose[:3, 3]
    pc = (points - t) @ R
    z = pc[:, 2]
    ok = z > near
    u = fx * pc[:, 0] / np.maximum(z, 1e-6) + W / 2.0
    v = fy * pc[:, 1] / np.maximum(z, 1e-6) + H / 2.0
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return ok


def forward_distances(tree, poses, far=8.0, step=0.05, radius=0.08, start=0.1):
    """Distance along each camera's forward axis to the nearest Gaussian.

    Only non-floor geometry is in ``tree``; this measures what the camera is
    actually looking at, not the floor 0.4 m below it.
    """
    poses = np.asarray(poses)
    centers = poses[:, :3, 3]
    dirs = poses[:, :3, 2]
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    vals = np.arange(start, far + 1e-9, step)
    q = centers[:, None, :] + vals[None, :, None] * dirs[:, None, :]
    dist, _ = tree.query(q.reshape(-1, 3))
    dist = dist.reshape(len(poses), len(vals))
    hit = dist <= radius
    out = np.full(len(poses), np.inf)
    has = hit.any(axis=1)
    out[has] = vals[np.argmax(hit[has], axis=1)]
    return out


def cells_of(path_uv, res=0.15):
    out = set()
    for k in range(len(path_uv) - 1):
        a, b = path_uv[k], path_uv[k + 1]
        n = max(2, int(np.linalg.norm(b - a) / (res * 0.5)) + 1)
        for t in np.linspace(0, 1, n):
            p = a + t * (b - a)
            out.add((int(np.floor(p[0] / res)), int(np.floor(p[1] / res))))
    out2 = set()
    for a, b in out:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                out2.add((a + da, b + db))
    return out2


def _lower_level(pose, up, floor, height):
    p = pose.copy()
    pos = p[:3, 3]
    p[:3, 3] = pos + (floor + height - pos @ up) * up
    return level_pose(p, up)


def build_trajectories(cfg, scene, engine_for, repair_root, cam_ids):
    """Select and render repair trajectories. Returns manifest entries."""
    # Every key comes from configs/method.yaml (merged under the scene config),
    # so there are no fallback defaults here to drift out of step with it.
    tr = cfg.trajectories
    floor, up = scene.floor, scene.up
    # cfg is already in scene units (units.scene_units); upm converts the
    # constants below and the *_m values written back out.
    upm = float(scene.units_per_metre)
    if floor is None:
        raise SystemExit("trajectory selection needs a floor "
                         "(set geometry.floor_m or enable estimation)")
    # One height or several; each yields its own candidates and the greedy
    # pick below chooses among all of them.
    heights = ([float(h) for h in tr.height_m] if OmegaConf.is_list(tr.height_m)
               or isinstance(tr.height_m, (list, tuple)) else [float(tr.height_m)])
    frames = int(tr.frames)
    fps = int(tr.fps)
    W, H = int(tr.uq_width), int(tr.uq_height)
    out_w, out_h = int(tr.width), int(tr.height)
    orientation = str(tr.orientation)
    if orientation not in ("auto", "landscape", "portrait"):
        raise SystemExit(f"trajectories.orientation must be auto, landscape or portrait, got {orientation}")
    fovy_deg = tr.fov_y_deg
    max_fov = float(cfg.views.max_fov_deg)
    min_frames = int(tr.min_frames)
    query_target = int(tr.query_target)
    max_gap = int(tr.max_gap)
    dist_lo, dist_hi = [float(x) for x in tr.dist_m]
    iou_lo, iou_hi = [float(x) for x in tr.iou]
    clearance = float(tr.clearance_m)
    fwd_lo, fwd_hi = [float(x) for x in tr.forward_band_m]
    fwd_far = float(tr.forward_far_m)
    fwd_step = float(tr.forward_step_m)
    fwd_radius = float(tr.forward_radius_m)
    uq_pct = float(tr.uq_percentile)
    min_sep = int(tr.min_sep)
    w_scene = float(tr.scene_weight)
    w_path = float(tr.path_weight)
    w_uq = float(tr.uq_weight)
    voxel = float(tr.voxel_m)
    pitches = [float(x) for x in tr.pitches]
    num = int(tr.per_camera)

    by_cam = {}
    for im in scene.imgs:
        by_cam.setdefault(im["cam_id"], []).append(im)

    # scene coverage / clearance structures
    pts = scene.xyz[::20]
    vmin = pts.min(0) - voxel
    vcode = np.floor((pts - vmin) / voxel).astype(np.int64) + 1
    code = (vcode[:, 0] * 10_000 + vcode[:, 1]) * 10_000 + vcode[:, 2]
    occupied = set(code.tolist())
    tree = cKDTree(scene.xyz[::5])
    # forward-direction geometry: ignore floor and ceiling
    h_rel = scene.xyz @ up - floor
    fwd_lo_h, fwd_hi_h = [float(x) for x in tr.forward_height_band_m]
    keep = (h_rel >= fwd_lo_h) & (h_rel <= fwd_hi_h)
    # Handheld captures leave translucent floaters along the camera path; they
    # would count as "something straight ahead" at 10 cm. Optionally ignore them.
    min_op = float(tr.forward_min_opacity)
    if min_op > 0:
        keep &= scene.model.get_opacity.detach().cpu().numpy().ravel() >= min_op
    fwd_pts = scene.xyz[keep]
    fwd_tree = cKDTree(fwd_pts)
    mj64 = min_jerk(np.linspace(0, 1, 64))
    ss = np.linspace(0, 1, frames)

    def forward_ok(poses):
        fd = forward_distances(fwd_tree, poses, fwd_far, fwd_step, fwd_radius, start=0.1 * upm)
        finite = fd[np.isfinite(fd)]
        if finite.size < 0.8 * len(fd):
            return False, fd
        med = float(np.median(finite))
        ok = (med >= fwd_lo and med <= fwd_hi and finite.min() >= fwd_lo)
        return ok, fd

    entries = []
    for cam_id in sorted(cam_ids):
        cimgs = sorted(by_cam.get(cam_id, []), key=lambda x: x["name"])
        if len(cimgs) < min_frames:
            print(f"[rvf] camera {cam_id}: only {len(cimgs)} frames, skip")
            continue
        qstride = max(1, len(cimgs) // max(query_target, 1))
        query = cimgs[::qstride][:query_target]
        cam = scene.cams[cam_id]
        # Clips follow the camera's orientation: a portrait phone video gets
        # portrait clips, so its photos need no cropping to serve as references.
        portrait = orientation == "portrait" or (orientation == "auto" and cam["H"] > cam["W"])
        cw, ch = (H, W) if portrait != (H > W) else (W, H)
        ow, oh = (out_h, out_w) if portrait != (out_h > out_w) else (out_w, out_h)
        fovx, fovy = fov_from_cam(cam, max_deg=max_fov)
        fx = cw / (2 * np.tan(fovx / 2))
        fy = ch / (2 * np.tan(fovy / 2))
        engine = engine_for(cam_id)
        cands = []
        for height, pd in ((h, pd) for h in heights for pd in pitches):
            base_level = [_lower_level(im["c2w"], up, floor, height) for im in query]
            ps = ([p.copy() for p in base_level] if abs(pd) < 1e-9 else
                  [pitch_pose(p, up, np.deg2rad(pd)) for p in base_level])
            uv = np.stack([[p[:3, 3] @ scene.e1, p[:3, 3] @ scene.e2]
                           for p in ps])
            uq = np.array([float(np.nanmean(engine.render_uq(
                cam, p, size=(cw, ch)))) for p in ps])
            uq_thr = float(np.percentile(uq, uq_pct))
            masks = [view_mask(pts, p, fx, fy, cw, ch, near=0.05 * upm) for p in ps]
            src_cells = cells_of(uv, res=0.15 * upm)
            gap = min(max_gap, len(ps) - 1)
            for i in range(len(ps)):
                for j in range(i + 2, min(i + gap + 1, len(ps))):
                    seg = uq[i:j + 1]
                    if float(seg.max()) < uq_thr:
                        continue
                    d = float(np.linalg.norm(uv[i] - uv[j]))
                    if not (dist_lo <= d <= dist_hi):
                        continue
                    inter = np.logical_and(masks[i], masks[j]).sum()
                    union = np.logical_or(masks[i], masks[j]).sum()
                    iou = float(inter) / max(int(union), 1)
                    if not (iou_lo <= iou <= iou_hi):
                        continue
                    p0 = (uv[i, 0] * scene.e1 + uv[i, 1] * scene.e2
                          + (floor + height) * up)
                    p1 = (uv[j, 0] * scene.e1 + uv[j, 1] * scene.e2
                          + (floor + height) * up)
                    clr = float(tree.query(p0 + mj64[:, None] * (p1 - p0))[0].min())
                    if clr < clearance:
                        continue
                    q0 = Rotation.from_matrix(ps[i][:3, :3]).as_quat()
                    q1 = Rotation.from_matrix(ps[j][:3, :3]).as_quat()
                    pp = p0 + min_jerk(ss)[:, None] * (p1 - p0)
                    ip = []
                    for m in range(frames):
                        q = np.eye(4)
                        q[:3, :3] = Rotation.from_quat(
                            slerp(q0, q1, ss[m])).as_matrix()
                        q[:3, 3] = pp[m]
                        ip.append(q)
                    seen = np.zeros(len(pts), dtype=bool)
                    path = []
                    for q in ip:
                        seen |= view_mask(pts, q, fx, fy, cw, ch, near=0.05 * upm)
                        path.append([q[:3, 3] @ scene.e1,
                                     q[:3, 3] @ scene.e2])
                    cands.append({
                        "i": i, "j": j, "pitch": float(pd), "height": height, "poses": ip,
                        "d": d, "iou": iou, "clr": clr,
                        "cov": set(code[seen].tolist()),
                        "path": cells_of(np.asarray(path), res=0.15 * upm),
                        "uq": 0.5 * float(seg.mean()) + 0.5 * float(seg.max()),
                        "uq_seg_max": float(seg.max()),
                        "i_name": query[i]["name"], "j_name": query[j]["name"],
                    })

        if not cands:
            print(f"[rvf] camera {cam_id}: no trajectory candidates")
            continue
        uqv = np.array([c["uq"] for c in cands])
        un = ((uqv - uqv.min()) / (uqv.max() - uqv.min())
              if uqv.max() > uqv.min() else np.full(len(uqv), 0.5))
        for c, u in zip(cands, un):
            c["uq_norm"] = float(u)

        def pick(sep):
            best, best_gain = None, -1.0
            for c in cands:
                if (c["i"], c["j"], c["height"]) in used:
                    continue
                # temporal separation only among clips at the same height:
                # the same frames seen from another height are a different view
                if any(s["height"] == c["height"]
                       and not (max(c["i"], c["j"]) < min(s["i"], s["j"]) - sep
                                or min(c["i"], c["j"]) > max(s["i"], s["j"]) + sep)
                       for s in selected):
                    continue
                gs = len(c["cov"] - covered) / max(len(occupied), 1)
                gp = len(c["path"] - covered_path) / max(len(src_cells), 1)
                gain = w_scene * gs + w_path * gp + w_uq * c["uq_norm"]
                if gain > best_gain:
                    best, best_gain = c, gain
            return best, best_gain

        selected, covered, covered_path, used = [], set(), set(), set()
        fwd_rejected = 0
        for _ in range(num):
            # Keep the temporal separation first; if the pool is exhausted
            # before reaching the per-camera quota, relax it rather than
            # returning fewer trajectories.
            # The best remaining candidate that passes the forward check. Each
            # candidate is checked at most once; there is no attempt cap, since
            # in cluttered scenes the highest-gain clips are often the ones
            # hugging the furniture and all fail before a valid one comes up.
            while True:
                best, best_gain = pick(min_sep)
                if best is None:
                    best, best_gain = pick(1)
                if best is None:
                    best, best_gain = pick(0)
                if best is None:
                    break
                valid, fd = forward_ok(best["poses"])
                if valid:
                    break
                fwd_rejected += 1
                used.add((best["i"], best["j"], best["height"]))
            if best is None:
                break
            best["fwd"] = fd
            selected.append(best)
            used.add((best["i"], best["j"], best["height"]))
            covered |= best["cov"]
            covered_path |= best["path"]

        for k, c in enumerate(selected):
            rgbs = [engine.render_rgb(cam, p, size=(ow, oh),
                                      fovy_deg=fovy_deg)
                    for p in c["poses"]]
            name = f"traj_c{cam_id}_{k:02d}"
            meta = {
                "schema": "repair_view_finder/trajectory/v1",
                "candidate": c["i_name"], "camera": f"camera_{cam_id}",
                "rank": k + 1,
                "pitch_deg": c["pitch"],
                "coordinate_frame": cfg.scene.get("frame_label", "unknown"),
                "pair": {"i": c["i_name"], "j": c["j_name"],
                         "baseline_m": c["d"] / upm, "view_iou": c["iou"],
                         "clearance_m": c["clr"] / upm},
                "forward": {
                    "median_m": (float(np.median(c["fwd"][np.isfinite(c["fwd"])])) / upm
                                 if np.isfinite(c["fwd"]).any() else None),
                    "min_m": (float(c["fwd"][np.isfinite(c["fwd"])].min()) / upm
                              if np.isfinite(c["fwd"]).any() else None),
                    "max_m": (float(c["fwd"][np.isfinite(c["fwd"])].max()) / upm
                              if np.isfinite(c["fwd"]).any() else None),
                },
                "clip": {"frames": len(rgbs), "video": "trajectory.mp4",
                         "height_m": c["height"] / upm, "width": ow,
                         "height": oh, "fps": fps},
                "score": {"uq_excess_max": c["uq_seg_max"],
                          "coverage": len(c["cov"]) / max(len(occupied), 1)},
            }
            if fovy_deg is None:
                ofx, ofy = fovx, fovy
            else:
                ofy = float(np.deg2rad(fovy_deg))
                ofx = 2.0 * np.arctan(np.tan(ofy / 2.0) * ow / oh)
            tdir = write_trajectory(repair_root, name, c["poses"], rgbs,
                                    ow, oh, ofx, ofy, meta)
            viz.write_video(rgbs, tdir / "trajectory.mp4", fps=fps)
            entries.append({
                "rank": k + 1, "camera": f"camera_{cam_id}",
                "video": f"trajectories/{name}/trajectory.mp4",
                "trajectory": name, "candidate": c["i_name"],
                "pitch_deg": c["pitch"]})
        print(f"[rvf] camera {cam_id}: {len(selected)} trajectories "
              f"(from {len(cands)} candidates)")
        if len(selected) < num and fwd_rejected:
            # The forward check is tuned for indoor, low, level views; say so
            # instead of quietly returning fewer clips.
            print(f"[rvf] camera {cam_id}: {fwd_rejected} candidates failed the forward-distance "
                  f"check (nothing within trajectories.forward_band_m along the view axis). "
                  f"For object-centric or outdoor captures look down (negative "
                  f"trajectories.pitches), raise trajectories.height_m, and let the ground count "
                  f"(trajectories.forward_height_band_m: [0.0, 3.0]); see "
                  f"docs/configuration.md#choosing-the-path-settings.")

    return entries
