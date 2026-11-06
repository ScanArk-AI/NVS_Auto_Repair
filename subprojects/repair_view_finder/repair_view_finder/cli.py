"""Command-line entry point for the repair-view finder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf

from .candidates import build_candidates
from .colmap_io import fov_from_cam
from .config import load_config
from .defects import DefectVolume
from .export import camera_dict, write_json, write_selected_views
from .ranking import rank_views
from .trajectories import build_trajectories
from .scene import load_model, load_scene
from .units import scene_units
from .uq_engine import UQEngine
from . import viz


def main():
    ap = argparse.ArgumentParser(description="Find viewpoints that need repair")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], action="extend",
                    metavar="key=value")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-views", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.out:
        cfg.output.out_dir = args.out
    if args.limit_views:
        cfg.views.max_views = args.limit_views
    out = Path(cfg.output.out_dir)
    (out / "review").mkdir(parents=True, exist_ok=True)
    (out / "repair").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scene = load_scene(cfg)
    print(f"[rvf] scene {scene.root}")
    print(f"[rvf] ply {scene.ply}  ({scene.xyz.shape[0]:,} gaussians, "
          f"SH {scene.sh_degree})")
    print(f"[rvf] sparse {scene.sparse}  ({len(scene.imgs)} images)")
    print(f"[rvf] floor {scene.floor}  up {scene.up.round(3)}")
    print(f"[rvf] {scene.units_per_metre:.4g} scene units per metre ({scene.units_source})"
          + (f", median camera height {scene.camera_height / scene.units_per_metre:.2f} m"
             if scene.camera_height is not None else ""))
    # From here on the algorithm works in scene units; ``metric`` keeps the
    # config as written for everything a person reads (run.json, review page).
    metric = cfg
    cfg = scene_units(metric, scene.units_per_metre)

    cands = build_candidates(cfg, scene)
    print(f"[rvf] {len(cands)} candidate views")

    # One visibility field per camera (default) keeps each camera's intrinsics
    # consistent; ``field.cam_id`` pins a single field for every candidate.
    shared_render = None
    if (bool(cfg.output.full_render)
            and int(cfg.model.max_gaussians) > 0):
        shared_render, _ = load_model(scene.ply, 0,
                                      int(cfg.model.seed))
    per_camera = (cfg.field.get("cam_id") is None
                  and bool(cfg.field.per_camera))
    engines = {}

    def engine_for(cam_id):
        if not per_camera:
            if "default" not in engines:
                engines["default"] = UQEngine(cfg, scene,
                                              render_model=shared_render)
                eng = engines["default"]
                print(f"[rvf] field camera {eng.field_cid}, "
                      f"{len(eng.field)} views")
            return engines["default"]
        cam_id = int(cam_id)
        if cam_id not in engines:
            engines[cam_id] = UQEngine(cfg, scene, cam_id=cam_id,
                                       render_model=shared_render)
            print(f"[rvf] field camera {cam_id}, "
                  f"{len(engines[cam_id].field)} views")
        return engines[cam_id]

    dv = DefectVolume(scene, cfg)
    records = []
    for k, cand in enumerate(cands):
        uq, rgb = engine_for(cand.cam_id).render(cand)
        ex = uq - np.nanmedian(uq)
        dv.add_view(cand.idx, cand.pose, cand.cam, ex)
        rec = {
            "cand_idx": cand.idx, "name": cand.name, "pitch": cand.pitch_deg,
            "uq": uq, "excess": ex, "rgb": rgb,
            "mean_uq": float(np.nanmean(uq)),
            "mean_excess": float(np.nanmean(ex)),
            "frac_high": float(np.nanmean(ex > float(cfg.scoring.uq_excess))),
            "bad_vox": dv.view_bad[cand.idx],
            "bad_val": dv.view_val[cand.idx],
        }
        records.append(rec)
        if (k + 1) % 20 == 0:
            print(f"[rvf] scored {k + 1}/{len(cands)}")

    dv.finalize()
    print(f"[rvf] {len(dv.regions)} defect regions, "
          f"{0 if dv.defect_vox is None else len(dv.defect_vox)} high-UQ voxels")

    for rec in records:
        rec["regions_seen"] = dv.regions_seen(rec["cand_idx"])
    ranked = rank_views(records, dv.n_vox, int(cfg.views.count))
    rec_by_cand = {r["cand_idx"]: r for r in records}
    cand_by_idx = {c.idx: c for c in cands}
    print(f"[rvf] ranked {len(ranked)} views")

    # -------------------------------------------------- machine-readable
    repair = out / "repair"
    entries = []
    for rank, (cid, gain, new) in enumerate(ranked):
        cand = cand_by_idx[cid]
        rec = rec_by_cand[cid]
        fovx, fovy = fov_from_cam(cand.cam,
                                  max_deg=float(cfg.views.max_fov_deg))
        score = {"rank": rank + 1, "gain": gain, "new_voxels": new,
                 "mean_uq": rec["mean_uq"], "mean_excess": rec["mean_excess"],
                 "frac_high": rec["frac_high"], "pitch_deg": cand.pitch_deg,
                 "bad_voxels": int(len(dv.view_bad.get(cid, []))),
                 "regions_seen": rec["regions_seen"]}
        entries.append({
            "name": cand.name, "pitch_deg": cand.pitch_deg,
            "camera": camera_dict(int(cfg.views.width), int(cfg.views.height), fovx, fovy),
            "c2w": np.asarray(cand.pose).tolist(),
            "w2c": np.linalg.inv(cand.pose).tolist(),
            "score": score,
        })
    write_selected_views(repair, entries)
    write_json(repair / "defects.json", {
        "convention": "world coordinates of the loaded scene",
        "coordinate_frame": cfg.scene.get("frame_label", "unknown"),
        "voxel_m": float(metric.scoring.voxel_m),
        "uq_excess_threshold": float(cfg.scoring.uq_excess),
        "regions": dv.regions,
    })

    # ------------------------------------------------------- pose overview
    top = ranked[: int(cfg.views.count)]
    def short(name: str) -> str:
        # "map_0/camera_0/123.jpg" -> "camera_0"; a flat COLMAP name -> its stem
        parts = name.split("/")
        return parts[-2] if len(parts) >= 2 else os.path.splitext(name)[0]

    items = [(f"#{rank + 1}  {short(cand_by_idx[cid].name)}",
              rec_by_cand[cid]["rgb"])
             for rank, (cid, _gain, _new) in enumerate(top)]
    viz.overview_image(items, out / "review" / "top40_views.png",
                       title=f"最该修的{len(items)}个视角",
                       subtitle=f"{os.path.basename(scene.root)} · GAVIS UQ ranking")
    viz.topdown_map(scene, cands, ranked, dv,
                    out / "review" / "map_topdown.png",
                    top=int(cfg.views.count))

    # ------------------------------------------------------- trajectories
    # The one trajectory-finding algorithm, run independently per camera.
    clip_entries = build_trajectories(
        cfg, scene, engine_for, repair, sorted({c.cam_id for c in cands}))

    # ----------------------------------------------------------- tables
    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "candidate", "pitch_deg", "gain", "new_voxels",
                    "mean_uq", "mean_excess", "frac_high", "bad_voxels",
                    "regions"])
        for rank, (cid, gain, new) in enumerate(ranked):
            rec = rec_by_cand[cid]
            cand = cand_by_idx[cid]
            w.writerow([rank + 1, cand.name, cand.pitch_deg, f"{gain:.4f}",
                        new, f"{rec['mean_uq']:.4f}", f"{rec['mean_excess']:.4f}",
                        f"{rec['frac_high']:.4f}",
                        len(dv.view_bad.get(cid, [])),
                        sorted(rec["regions_seen"])])
    write_json(out / "candidates.json", [
        {"idx": c.idx, "name": c.name, "pitch_deg": c.pitch_deg,
         "cam_id": c.cam_id, "c2w": np.asarray(c.pose).tolist()}
        for c in cands])
    write_json(out / "run.json", {
        "schema": "repair_view_finder/run/v1",
        "config": OmegaConf.to_container(metric, resolve=True),
        "units_per_metre": scene.units_per_metre,
        "inputs": {"scene": scene.root, "ply": scene.ply,
                   "sparse": scene.sparse,
                   "scene_config": str(args.config),
                   "frame_label": cfg.scene.get("frame_label", "unknown")},
        "counts": {"candidates": len(cands), "selected": len(ranked),
                   "defect_regions": len(dv.regions),
                   "defect_voxels": 0 if dv.defect_vox is None else len(dv.defect_vox)},
        "elapsed_s": time.time() - t0,
    })
    # Read by the repair stage: camera_scale = 0.01 / units_per_metre.
    write_json(repair / "scene.json", {
        "schema": "repair_view_finder/scene/v1",
        "units_per_metre": float(scene.units_per_metre),
        "units_source": scene.units_source,
        "up": np.asarray(scene.up).tolist(),
        "floor": scene.floor,
        "camera_height_median": scene.camera_height,
    })
    write_json(repair / "manifest.json", {
        "schema": "repair_view_finder/manifest/v1",
        "scene": scene.root, "ply": scene.ply, "sparse": scene.sparse,
        "coordinate_frame": cfg.scene.get("frame_label", "unknown"),
        "selected_views": entries,
        "trajectories": clip_entries,
        "defects": "defects.json",
        "review": "../review/index.html",
    })

    # ------------------------------------------------------------- html
    groups = {}
    for e in clip_entries:
        groups.setdefault(e["camera"], []).append(e)
    sections = []
    for cam, es in sorted(groups.items()):
        cards = []
        for e in es:
            video = e["video"]
            poster = f"../repair/{video.rsplit('/', 1)[0]}/frames/00000.png"
            cards.append(
                f"<figure><video controls muted loop playsinline "
                f"preload='metadata' poster='{poster}'>"
                f"<source src='../repair/{video}' type='video/mp4'></video>"
                f"<figcaption>#{e['rank']} · {e['candidate']}"
                f"</figcaption></figure>")
        sections.append(
            f"<h3>{cam}（{len(es)} 条）</h3>"
            f"<div class='grid'>{''.join(cards)}</div>")
    html_doc = f"""<!doctype html><meta charset="utf-8">
<title>修复轨迹视频 - {os.path.basename(scene.root)}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:24px}}
h3{{margin:22px 0 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));
gap:16px;margin-top:10px}}
figure{{margin:0;background:#1b1b1b;border-radius:6px;padding:8px}}
video{{width:100%;display:block;border-radius:4px}}
figcaption{{font-size:12px;color:#bbb;margin-top:6px;word-break:break-all}}
</style>
<h2>修复轨迹视频 — {os.path.basename(scene.root)}</h2>
<p>{len(clip_entries)} 条 · 每条 {cfg.trajectories.frames} 帧 ·
离地 {metric.trajectories.height_m} m · 俯仰
{list(cfg.trajectories.pitches)}° · {cfg.trajectories.fps} fps</p>
{''.join(sections)}
"""
    (out / "review" / "index.html").write_text(html_doc, encoding="utf-8")
    print(f"[rvf] review  {out / 'review' / 'index.html'}")
    print(f"[rvf] repair  {repair}")
    print(f"[rvf] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
