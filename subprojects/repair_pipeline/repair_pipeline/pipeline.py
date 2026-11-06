"""Stage orchestration: select references -> render opacity -> repair -> frames."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import caption as caption_mod
from . import frames as frames_mod
from . import pack, references
from .opacity import ensure_opacity

STAGES = ("select", "repair")


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.repair_root = Path(cfg.input.repair_root)
        self.out_root = Path(cfg.output.dir)
        self._scene = None
        self._photo_masks = None
        self._model = None

    # -- discovery ---------------------------------------------------------
    def trajectories(self, limit: int | None = None) -> list[Path]:
        root = self.repair_root / "trajectories"
        if not root.is_dir():
            raise FileNotFoundError(f"{root} not found -- is input.repair_root the `repair/` dir?")
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        missing = [d.name for d in dirs if not (d / "poses.json").is_file()]
        if missing:
            raise FileNotFoundError(f"trajectories without poses.json: {missing[:5]}")
        return dirs[:limit] if limit else dirs

    def out_dir(self, traj_dir: Path) -> Path:
        return self.out_root / traj_dir.name

    # -- stages ------------------------------------------------------------
    def _scene_refs(self):
        if self._scene is None:
            self._scene = references.load_scene(self.cfg.scene)
            cache = Path(self.cfg.cache.dir)
            self._photo_masks = references._photo_masks(self._scene, self.cfg.scene, cache)
        return self._scene, self._photo_masks

    def select(self, traj_dir: Path) -> dict:
        scene, photo_masks = self._scene_refs()
        return references.select_for_trajectory(
            scene, photo_masks, traj_dir, self.cfg.refer, self.cfg.scene,
            self.out_dir(traj_dir) / "references"
        )

    def ensure_opacity(self, traj_dirs: list[Path]) -> None:
        if not self.cfg.opacity.enabled:
            return
        expected = len(sorted((traj_dirs[0] / "frames").glob("*.png")))
        ensure_opacity(traj_dirs, Path(self.cfg.scene.ply), self.cfg.opacity.python,
                       str(self.cfg.model.device), expected)

    def repair(self, traj_dir: Path, payload: dict, prompt_h5: Path) -> np.ndarray:
        scene, _ = self._scene_refs()
        out = self.out_dir(traj_dir)
        c2w, cam = references.load_trajectory(traj_dir)
        transforms, targets, neighbors = pack.build_transforms(
            traj_dir, c2w, cam, payload["selected"], scene.cams,
            {im.name: im for im in scene.images},
            target_fov_deg=payload["config"]["target_fov_deg"],
            neighbor_intrinsics=str(self.cfg.refer.get("neighbor_intrinsics", "prepared")),
        )

        from .model import Clip

        clip = Clip(
            rgb=frames_mod.load_frames(traj_dir / "frames"),
            opacity=frames_mod.load_gray_frames(traj_dir / "opacity"),
            neighbors=frames_mod.load_neighbors(out / "references", payload["selected"]),
            transforms=transforms,
            targets=targets,
            neighbor_indices=neighbors,
            prompt_h5=prompt_h5,
        )
        if self._model is None:
            from .backends import make_backend

            self._model = make_backend(self.cfg)
        return self._model.repair(clip)

    # -- driver ------------------------------------------------------------
    def run(self, stages=STAGES, limit: int | None = None) -> dict:
        traj_dirs = self.trajectories(limit)
        self.out_root.mkdir(parents=True, exist_ok=True)
        # Only the repair stage needs a prompt; `select` must not pay to load a
        # 4 GB text encoder.
        prompt_h5, prompt_text = (None, "")
        if "repair" in stages:
            prompt_h5, prompt_text = caption_mod.resolve_prompt(self.cfg, self.out_root)

        report = {"config": None, "prompt_text": prompt_text, "trajectories": {}}
        t_start = time.time()

        if "repair" in stages:
            self.ensure_opacity(traj_dirs)

        for i, traj in enumerate(traj_dirs, start=1):
            t0 = time.time()
            out = self.out_dir(traj)
            entry = {}

            # Re-running a workflow should not redo finished work.
            if (self.cfg.output.get("skip_existing")
                    and list((out / "repaired").glob("*.png"))):
                entry["skipped"] = True
                report["trajectories"][traj.name] = entry
                print(f"[{i}/{len(traj_dirs)}] {traj.name}: already repaired, skipped", flush=True)
                continue

            payload = None
            sel_json = out / "references" / "selected.json"
            if "select" in stages or not sel_json.is_file():
                payload = self.select(traj)
                entry["select_s"] = round(time.time() - t0, 1)
            else:
                payload = json.loads(sel_json.read_text(encoding="utf-8"))

            if "repair" in stages:
                t1 = time.time()
                frames = self.repair(traj, payload, prompt_h5)
                entry["repair_s"] = round(time.time() - t1, 1)

                if self.cfg.output.save_png:
                    frames_mod.save_frames(frames, out / "repaired")
                if self.cfg.output.save_mp4:
                    frames_mod.write_video(
                        (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8),
                        out / "repaired.mp4", fps=int(self.cfg.output.fps))
                if self.cfg.output.before_after:
                    before = frames_mod.load_frames(traj / "frames")[: len(frames)]
                    frames_mod.before_after(
                        before, frames, out / "before_after.mp4",
                        before_label="before (3DGS render)",
                        after_label=f"after ({self.cfg.model.variant})",
                        fps=int(self.cfg.output.fps))

            entry["overlap_median_frac"] = float(np.median([s["overlap_frac"] for s in payload["selected"]]))
            entry["total_s"] = round(time.time() - t0, 1)
            report["trajectories"][traj.name] = entry
            print(f"[{i}/{len(traj_dirs)}] {traj.name}: "
                  f"overlap {entry['overlap_median_frac']:.1%}"
                  + (f", repair {entry['repair_s']}s" if "repair_s" in entry else "")
                  + f", {entry['total_s']}s", flush=True)

        report["elapsed_s"] = round(time.time() - t_start, 1)
        report["n"] = len(traj_dirs)
        (self.out_root / "run.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return report
