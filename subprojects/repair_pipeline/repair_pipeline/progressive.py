"""Progressive-reference repair -- an experimental inference method.

The standard pipeline (``pipeline.py``) treats trajectories independently: each
gets the ``num_views`` real photos sharing the most Gaussians with it. This
module repairs them one after another instead, and lets a later trajectory use
frames already repaired for an earlier one as references.

What it changes, all opt-in via the ``progressive`` config block:

* **Order.** ``order: coverage`` repairs the trajectories the photos cover best
  first, so the frames handed on are the most trustworthy ones; ``name`` keeps
  the finder's order.
* **Candidate pool.** Real photos plus every frame of every trajectory repaired
  so far. Repaired frames are not ranked on raw overlap -- one always overlaps a
  neighbouring trajectory heavily -- but on *residual* coverage: what this
  trajectory sees that the chosen photos do not. ``criterion: direction``
  (default) counts a (target frame, Gaussian) pair as covered only if a
  reference sees that Gaussian within ``angle_deg`` of the target's viewing
  direction; ``frustum`` only asks whether it is in the reference's frustum,
  which on fisheye rigs says the photos see everything (measured on an indoor office capture: 18 of
  22 trajectories at >= 99.95 %). The total stays at
  ``num_views`` (the model was trained with 12); at most ``max_generated`` slots
  go to repaired frames, at most ``per_traj_max`` from any one trajectory, and
  only if each adds ``min_gain_frac`` of the trajectory's Gaussians, and only as
  many as make the final set cover *more* than the photos they displace (a tie
  keeps the photos), so a trajectory never ends up worse covered than with the
  standard references.
* **Drift guard.** A trajectory repaired from photos only is generation 1; one
  that used generation-g frames is g+1. ``max_generation`` caps the chain.
* **Seeded noise.** The sampler draws fresh noise at every step from torch's
  global RNG and the standard pipeline never seeds it, so two runs on identical
  inputs differ. Each trajectory here is repaired after
  ``torch.manual_seed(seed + crc32(name))``: arms that differ only in their
  references then differ only because of their references. Leave
  ``model.seed`` at null: when set, ``RepairModel.repair`` reseeds with it and
  every trajectory gets that same seed instead.
* **Reference intrinsics** follow ``refer.neighbor_intrinsics``, the same switch
  the standard pipeline reads, so both modes describe photos identically.

Nothing in the standard pipeline is modified. Run it with::

    python -m repair_pipeline.progressive --config <scene.yaml> [--set progressive.enabled=false ...]

With ``enabled=false`` it selects exactly the references the standard pipeline
would and feeds them identically -- the baseline arm for a comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from . import caption as caption_mod
from . import frames as frames_mod
from .config import load_config
from .coverage import DirectionDemand, TrajMasks
from .coverage import popcount as _popcount
from .coverage import union_bits as _union_bits
from .frustum import gaussian_frustum_masks, select_neighbor_indices
from .pipeline import Pipeline
from .poses import CAMERA_CONVENTION_FLIP, c2w_opencv_to_opengl
from .prepare import photo_intrinsics, prepare_reference, prepared_intrinsics
from .references import load_trajectory

DEFAULTS = OmegaConf.create({"progressive": {
    "enabled": True,
    "order": "coverage",          # coverage | name
    "max_generated": 4,
    "per_traj_max": 2,
    "max_generation": 2,
    "min_gain_frac": 0.005,
    "seed": 0,
    "criterion": "direction",     # direction | frustum
    "angle_deg": 30.0,            # direction: max angle between viewing rays
    "sample_gaussians": 200000,   # direction: Gaussians sampled per trajectory
    "min_pool_overlap": 0.01,     # skip repaired frames sharing < 1 % of the target's Gaussians
    "pool_stride": 2,             # offer every n-th repaired frame (neighbours are near-duplicates)
}})


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
@dataclass
class RepairedFrame:
    traj: str
    frame: int
    path: Path
    c2w_gl: np.ndarray
    intrinsics: dict
    mask: np.ndarray
    generation: int


def trajectory_masks(scene, traj_dir: Path) -> TrajMasks:
    c2w_cv, cam = load_trajectory(traj_dir)
    target_gl = np.stack([c2w_opencv_to_opengl(m) for m in c2w_cv])
    per_frame = gaussian_frustum_masks(scene.positions, target_gl, np.tile(list(cam), (len(target_gl), 1)))
    return TrajMasks(c2w_cv, cam, target_gl, per_frame, np.bitwise_or.reduce(per_frame, axis=0))


def _eligible(scene, refer_cfg) -> np.ndarray:
    return np.flatnonzero(np.array(
        [f is None or f <= float(refer_cfg.max_mask_frac) for f in scene.mask_frac], dtype=bool))


def photo_coverage(scene, photo_masks, tm: TrajMasks, refer_cfg) -> float:
    idx = _eligible(scene, refer_cfg)
    picked, _ = select_neighbor_indices(tm.union, photo_masks[idx], int(refer_cfg.num_views))
    total = int(_popcount(tm.union))
    if not picked or not total:
        return 0.0
    covered = np.bitwise_or.reduce(photo_masks[idx[picked]], axis=0) & tm.union
    return float(_popcount(covered)) / total


def _greedy(demand, covered, cands: list[tuple], prog) -> list[tuple]:
    """Greedy residual coverage over packed bit sets.

    ``cands`` is ``[(frame, bits, overlap), ...]``; returns
    ``[(frame, overlap, gain), ...]``.
    """
    total = int(_popcount(demand))
    residual = demand & ~covered
    cands = list(cands)
    per_traj: dict[str, int] = {}
    picks = []
    while len(picks) < int(prog.max_generated) and cands and total:
        gains = np.array([_popcount(bits & residual) for _, bits, _ in cands])
        best = int(np.argmax(gains))
        if gains[best] / total < float(prog.min_gain_frac):
            break
        f, bits, ov = cands.pop(best)
        picks.append((f, ov, int(gains[best])))
        residual &= ~bits
        per_traj[f.traj] = per_traj.get(f.traj, 0) + 1
        if per_traj[f.traj] >= int(prog.per_traj_max):
            cands = [c for c in cands if c[0].traj != f.traj]
    return picks


def _intr_row(intr: dict) -> tuple:
    return (intr["w"], intr["h"], intr["fl_x"], intr["fl_y"], intr["cx"], intr["cy"])


def direction_coverage_top(scene, photo_masks, tm: TrajMasks, refer_cfg, prog) -> float:
    """Direction coverage of the photos the standard pipeline would pick."""
    idx = _eligible(scene, refer_cfg)
    picked, _ = select_neighbor_indices(tm.union, photo_masks[idx], int(refer_cfg.num_views))
    dem = DirectionDemand(scene, tm, prog.sample_gaussians, prog.angle_deg, int(prog.seed))
    cov = _union_bits([dem.covered(scene.frames_gl[i], scene.intrinsics[i]) for i in idx[picked]], dem.bits)
    return dem.fraction(cov)


def select(scene, photo_masks, traj_dir: Path, tm: TrajMasks, pool: list[RepairedFrame],
           refer_cfg, scene_cfg, prog, out_dir: Path) -> dict:
    """Photos (as the standard pipeline picks them), then repaired frames for the gaps."""
    from PIL import Image

    tw, th, _tfx, tfy, _tcx, _tcy = tm.camera
    num_views = int(refer_cfg.num_views)
    idx = _eligible(scene, refer_cfg)
    if len(idx) < num_views:
        raise RuntimeError(f"{traj_dir.name}: only {len(idx)} eligible photos, cannot pick {num_views}")

    overlap_all = _popcount(photo_masks[idx] & tm.union)
    order = np.argsort(-overlap_all, kind="stable")
    picks, dir_stats = [], None
    if prog.enabled:
        base = order[: num_views - min(int(prog.max_generated), num_views)]
        n_union = float(max(1, _popcount(tm.union)))
        cands = [f for f in pool if f.generation + 1 <= int(prog.max_generation)
                 and _popcount(f.mask & tm.union) / n_union >= float(prog.min_pool_overlap)]
        if prog.criterion == "direction":
            dem = DirectionDemand(scene, tm, prog.sample_gaussians, prog.angle_deg, int(prog.seed))
            photo_cov = [dem.covered(scene.frames_gl[i], scene.intrinsics[i]) for i in idx[order[:num_views]]]
            covered = _union_bits(photo_cov[: len(base)], dem.bits)
            ranked = _greedy(dem.bits, covered,
                             [(f, dem.covered(f.c2w_gl, _intr_row(f.intrinsics)), int(_popcount(f.mask & tm.union)))
                              for f in cands], prog) if cands else []
            pick_cov = [dem.covered(f.c2w_gl, _intr_row(f.intrinsics)) for f, _, _ in ranked]
            # A repaired frame displaces a photo. Keep the first k greedy picks
            # only for the k that beats the photos it pushes out; ties keep photos.
            curve = [dem.fraction(_union_bits(photo_cov[: num_views - k] + pick_cov[:k], dem.bits))
                     for k in range(len(ranked) + 1)]
            best_k = max(range(len(curve)), key=lambda k: (round(curve[k], 6), -k))
            picks = ranked[:best_k]
            dir_stats = {
                "angle_deg": float(prog.angle_deg), "sampled_gaussians": int(len(dem.sel)),
                "photos_top12": curve[0],
                "base_photos": dem.fraction(covered),
                "final_references": curve[best_k],
                "coverage_by_num_repaired": [round(c, 4) for c in curve],
            }
        else:
            covered = _union_bits(list(photo_masks[idx[base]]), tm.union)
            ranked = _greedy(tm.union, covered,
                             [(f, f.mask, int(_popcount(f.mask & tm.union))) for f in cands], prog) if cands else []
            n_u = float(max(1, _popcount(tm.union)))
            curve = [float(_popcount(_union_bits(list(photo_masks[idx[order[: num_views - k]]])
                                                 + [f.mask for f, _, _ in ranked[:k]], tm.union) & tm.union)) / n_u
                     for k in range(len(ranked) + 1)]
            best_k = max(range(len(curve)), key=lambda k: (round(curve[k], 6), -k))
            picks = ranked[:best_k]
    chosen = np.sort(order[: num_views - len(picks)])   # same picks as select_neighbor_indices

    target_fov = (float(refer_cfg.target_fov_deg) if refer_cfg.target_fov_deg
                  else float(np.degrees(2.0 * np.arctan2(th / 2.0, tfy))))
    prepared_dir = out_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    total = float(max(1, _popcount(tm.union)))

    selected = []
    for c in chosen:
        photo_idx = int(idx[c])
        im = scene.images[photo_idx]
        cam = scene.cams[im.camera_id]
        arr, post_frac = prepare_reference(
            Path(scene_cfg.images) / im.name, cam, tw, th, target_fov,
            images_root=Path(scene_cfg.images), image_name=im.name,
            mask_invalid=scene_cfg.mask_invalid, fill_masked=bool(refer_cfg.fill_masked))
        dst = prepared_dir / Path(im.name).with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(dst)
        intr = (prepared_intrinsics(cam, tw, th, target_fov)
                if refer_cfg.get("neighbor_intrinsics", "prepared") == "prepared" else photo_intrinsics(cam))
        selected.append({
            "rank": len(selected) + 1, "source": "photo", "photo_index": photo_idx, "name": im.name,
            "overlap": int(overlap_all[c]), "overlap_frac": float(overlap_all[c]) / len(scene.positions),
            "mask_frac": scene.mask_frac[photo_idx], "prepared": dst.relative_to(out_dir).as_posix(),
            "prepared_mask_frac": post_frac, "intrinsics": intr,
            "c2w_gl": (np.linalg.inv(im.w2c) @ CAMERA_CONVENTION_FLIP).tolist(),
        })

    for f, ov, gain in picks:
        dst = prepared_dir / "repaired" / f.traj / f.path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f.path, dst)
        selected.append({
            "rank": len(selected) + 1, "source": "repaired", "name": f"{f.traj}/{f.path.name}",
            "traj": f.traj, "frame": int(f.frame), "generation": int(f.generation),
            "overlap": ov, "overlap_frac": ov / len(scene.positions),
            "residual_gain": gain, "residual_gain_frac": gain / total,
            "mask_frac": 0.0, "prepared": dst.relative_to(out_dir).as_posix(), "prepared_mask_frac": 0.0,
            "intrinsics": f.intrinsics, "c2w_gl": np.asarray(f.c2w_gl).tolist(),
        })

    photos_or = np.bitwise_or.reduce(photo_masks[idx[chosen]], axis=0)
    refs_or = np.bitwise_or.reduce([photos_or] + [f.mask for f, _, _ in picks], axis=0)
    photo_cov = float(_popcount(photos_or & tm.union)) / total
    ref_cov = float(_popcount(refs_or & tm.union)) / total
    payload = {
        "trajectory": traj_dir.name, "frames": int(len(tm.c2w_cv)),
        "num_gaussians": int(len(scene.positions)), "num_views": len(selected),
        "eligible_photos": int(len(idx)),
        "generation": 1 + max((f.generation for f, _, _ in picks), default=0),
        "num_repaired_refs": len(picks),
        "photo_coverage": photo_cov, "reference_coverage": ref_cov,
        "direction_coverage": dir_stats,
        "config": {
            "num_views": num_views, "max_mask_frac": float(refer_cfg.max_mask_frac),
            "mask_invalid": str(scene_cfg.mask_invalid), "fill_masked": bool(refer_cfg.fill_masked),
            "target_fov_deg": target_fov, "progressive": OmegaConf.to_container(prog),
        },
        "selected": selected,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return payload


def build_transforms(tm: TrajMasks, refs: list[dict]) -> tuple[dict, list[int], list[int]]:
    """Same layout as ``pack.build_transforms``; each neighbour carries its own camera."""
    tw, th, tfx, tfy, tcx, tcy = tm.camera
    frames = [{
        "file_path": f"frames/{i:05d}.png", "transform_matrix": c2w_opencv_to_opengl(m).tolist(),
        "w": int(tw), "h": int(th), "fl_x": float(tfx), "fl_y": float(tfy), "cx": float(tcx), "cy": float(tcy),
    } for i, m in enumerate(tm.c2w_cv)]
    neighbors = []
    for ref in refs:
        neighbors.append(len(frames))
        frames.append({"file_path": ref["prepared"], "transform_matrix": ref["c2w_gl"], **ref["intrinsics"]})
    transforms = {"camera_model": "OPENCV", "fl_x": float(tfx), "fl_y": float(tfy),
                  "cx": float(tcx), "cy": float(tcy), "w": int(tw), "h": int(th), "frames": frames}
    return transforms, list(range(len(tm.c2w_cv))), neighbors


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
class ProgressivePipeline(Pipeline):
    def _frames_for_pool(self, traj: Path, tm: TrajMasks, generation: int) -> list[RepairedFrame]:
        tw, th, tfx, tfy, tcx, tcy = tm.camera
        intr = {"w": int(tw), "h": int(th), "fl_x": float(tfx), "fl_y": float(tfy),
                "cx": float(tcx), "cy": float(tcy)}
        files = sorted((self.out_dir(traj) / "repaired").glob("*.png"))
        stride = max(1, int(self.cfg.progressive.pool_stride))
        return [RepairedFrame(traj.name, i, f, tm.target_gl[i], intr, tm.per_frame[i], generation)
                for i, f in enumerate(files[: len(tm.target_gl)]) if i % stride == 0]

    def _repair_one(self, traj: Path, tm: TrajMasks, payload: dict, prompt_h5: Path) -> np.ndarray:
        from .backends import make_backend
        from .model import Clip

        out = self.out_dir(traj)
        transforms, targets, neighbors = build_transforms(tm, payload["selected"])
        clip = Clip(
            rgb=frames_mod.load_frames(traj / "frames"),
            opacity=frames_mod.load_gray_frames(traj / "opacity"),
            neighbors=frames_mod.load_neighbors(out / "references", payload["selected"]),
            transforms=transforms, targets=targets, neighbor_indices=neighbors, prompt_h5=prompt_h5)
        if self._model is None:
            self._model = make_backend(self.cfg)
        import torch

        seed = int(self.cfg.progressive.seed) + zlib.crc32(traj.name.encode())
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self._model.repair(clip)

    def run(self, limit: int | None = None) -> dict:
        prog = self.cfg.progressive
        if not self.cfg.output.save_png:
            raise ValueError("progressive repair reads earlier trajectories' PNGs; keep output.save_png=true")
        traj_dirs = self.trajectories(limit)
        self.out_root.mkdir(parents=True, exist_ok=True)
        prompt_h5, prompt_text = caption_mod.resolve_prompt(self.cfg, self.out_root)
        self.ensure_opacity(traj_dirs)

        scene, photo_masks = self._scene_refs()
        masks = {t.name: trajectory_masks(scene, t) for t in traj_dirs}
        if prog.criterion == "direction":
            coverage = {t.name: direction_coverage_top(scene, photo_masks, masks[t.name], self.cfg.refer, prog)
                        for t in traj_dirs}
        else:
            coverage = {t.name: photo_coverage(scene, photo_masks, masks[t.name], self.cfg.refer)
                        for t in traj_dirs}
        order = (sorted(traj_dirs, key=lambda t: (-coverage[t.name], t.name)) if prog.order == "coverage"
                 else list(traj_dirs))

        report = {"method": "progressive" if prog.enabled else "progressive-baseline", "progressive": OmegaConf.to_container(prog),
                  "prompt_text": prompt_text, "order": [t.name for t in order], "trajectories": {}}
        pool: list[RepairedFrame] = []
        t_start = time.time()
        for i, traj in enumerate(order, start=1):
            t0 = time.time()
            out = self.out_dir(traj)
            tm = masks[traj.name]
            sel_json = out / "references" / "selected.json"
            entry = {f"{prog.criterion}_coverage_top12": round(coverage[traj.name], 4)}

            prior = json.loads(sel_json.read_text(encoding="utf-8")) if sel_json.is_file() else None
            # Only resume work this module did: a selected.json from the photos
            # pipeline has no progressive block and must be redone, not reused.
            if (self.cfg.output.get("skip_existing") and prior and "progressive" in prior.get("config", {})
                    and list((out / "repaired").glob("*.png"))):
                payload = prior
                entry["skipped"] = True
            else:
                payload = select(scene, photo_masks, traj, tm, pool if prog.enabled else [],
                                 self.cfg.refer, self.cfg.scene, prog, out / "references")
                t1 = time.time()
                frames = self._repair_one(traj, tm, payload, prompt_h5)
                entry["repair_s"] = round(time.time() - t1, 1)
                frames_mod.save_frames(frames, out / "repaired")
                if self.cfg.output.save_mp4:
                    frames_mod.write_video((np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8),
                                           out / "repaired.mp4", fps=int(self.cfg.output.fps))
                if self.cfg.output.before_after:
                    before = frames_mod.load_frames(traj / "frames")[: len(frames)]
                    frames_mod.before_after(before, frames, out / "before_after.mp4",
                                            before_label="before (3DGS render)",
                                            after_label=(f"after ({self.cfg.model.variant}, "
                                                         f"{'progressive' if prog.enabled else 'baseline'})"),
                                            fps=int(self.cfg.output.fps))

            if prog.enabled:
                pool.extend(self._frames_for_pool(traj, tm, int(payload.get("generation", 1))))
            entry.update({
                "generation": payload.get("generation", 1),
                "num_repaired_refs": payload.get("num_repaired_refs", 0),
                "repaired_refs": [r["name"] for r in payload["selected"] if r.get("source") == "repaired"],
                "reference_coverage": round(payload.get("reference_coverage", 0.0), 4),
                "direction_coverage": payload.get("direction_coverage"),
                "total_s": round(time.time() - t0, 1),
            })
            report["trajectories"][traj.name] = entry
            dc = entry.get("direction_coverage") or {}
            cov_txt = (f"direction coverage {dc['photos_top12']:.1%} (12 photos) -> {dc['final_references']:.1%}"
                       if dc else f"frustum coverage {entry['reference_coverage']:.1%}")
            print(f"[{i}/{len(order)}] {traj.name}: gen {entry['generation']}, "
                  f"{entry['num_repaired_refs']} repaired refs, {cov_txt}"
                  + (", skipped" if entry.get("skipped") else f", repair {entry.get('repair_s')}s"), flush=True)

        report["elapsed_s"] = round(time.time() - t_start, 1)
        (self.out_root / "run.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m repair_pipeline.progressive", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    # Defaults underneath whatever the overrides already put there.
    cfg = OmegaConf.merge(DEFAULTS, cfg)
    report = ProgressivePipeline(cfg).run(limit=a.limit)
    print(f"done: {len(report['trajectories'])} trajectories in {report['elapsed_s']}s -> {cfg.output.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
