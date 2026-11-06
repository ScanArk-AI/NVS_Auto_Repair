"""Build and run the two stages as subprocesses.

Both stages are ordinary CLIs in their subproject, so the orchestrator only has
to assemble arguments and pick the interpreter. Note the two use different
``--set`` conventions (the finder takes a list, the pipeline repeats the flag);
that is handled here rather than pushed onto the user.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import envs, layout
from .config import ROOT, variant


@dataclass
class Stage:
    name: str
    python: str
    cwd: Path
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)

    def command(self) -> list[str]:
        return [self.python, *self.argv]

    def run(self) -> int:
        started = time.time()
        print(f"\n=== {self.name}  ({time.strftime('%H:%M:%S')})")
        print(f"    {self.python} {' '.join(self.argv)}")
        # UTF-8 mode: on Windows, open()/print default to the locale code page,
        # which breaks on the non-ASCII names and captions the stages write.
        proc = subprocess.run(self.command(), cwd=str(self.cwd),
                              env={**os.environ, "PYTHONUTF8": "1", **self.env})
        print(f"=== {self.name} finished in {time.time() - started:.0f}s (exit {proc.returncode})")
        return proc.returncode


def find_stage(cfg) -> Stage:
    python = envs.resolve(cfg, "finder")
    argv = ["run.py", "--config", str(cfg.find.config), "--out", str(layout.find_out(cfg))]
    if cfg.find.get("limit_views"):
        argv += ["--limit-views", str(int(cfg.find.limit_views))]
    if cfg.find.set:
        argv += ["--set", *cfg.find.set]
    return Stage("find", python, ROOT / "subprojects" / "repair_view_finder", argv)


def _repair_sets(cfg) -> list[str]:
    """Injected paths first, then the user's own overrides (so theirs win)."""
    cp = Path(cfg.checkpoints_dir)
    v = variant(cfg)
    spec = cfg.checkpoints.artifixer[v]
    resolve = lambda p: str(Path(p) if Path(p).is_absolute() else cp / p)

    caption = Path(cfg.out) / "caption.h5"
    injected = [
        f"input.repair_root={layout.repair_root(cfg)}",
        f"output.dir={layout.repair_out(cfg)}",
        f"model.repo={cfg.paths.artifixer_repo}",
        f"model.variant={v}",
        f"model.variants.{v}.checkpoint_pt={resolve(spec.checkpoint_pt)}",
        f"model.variants.{v}.model_id={resolve(spec.model_id)}",
        f"prompt.text_encoder={resolve(cfg.checkpoints.text_encoder)}",
        "output.skip_existing=true",
        *( [f"prompt.caption_h5={caption}"] if caption.is_file() else [] ),
        "opacity.python=null",   # gsplat is installed in this same env now
        # Frustum/mask caches key on the ply and survive across scenes; keep
        # them at the repo root instead of writing 1+ GB inside a subproject.
        f"cache.dir={ROOT / '.cache'}",
    ]
    return injected + list(cfg.repair.set)


# Free GiB the repair stage waits for (960x544, 25 frames, 12 references). The
# 1.3B with model.low_vram takes ~12.3-13.1 GiB above the idle card (Linux with
# flash SDPA off; an RTX 5060 Ti 16 GB under Windows/WSL), ~28 GB without; below
# _VRAM_FULL_GB free the stage turns low_vram on by itself. On Windows the
# 16 GB card measured 12.1 GiB above idle (native) and ~12.3 (WSL), while the
# desktop holds 1.7-2.9 GiB of it, leaving 12.7-14.0 free -- 13 made it wait
# for the desktop, which never gives memory back on its own. The 14B was only
# measured without low_vram.
_VRAM_NEED_GB = {"1.3b": 12.5, "14b": 92}
_VRAM_FULL_GB = {"1.3b": 24}


def _free_vram_gb(device: str) -> float | None:
    """Free memory on the GPU ``device`` names (``cuda`` -> the first visible one)."""
    import re
    import subprocess

    m = re.search(r"(\d+)$", device)
    idx = int(m.group(1)) if m else 0
    # nvidia-smi numbers physical cards; with CUDA_VISIBLE_DEVICES set, "cuda:0"
    # is the first *visible* one. Map back or the gate watches the wrong card.
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if visible and idx < len(visible):
        idx = int(visible[idx])
    idx = str(idx)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=" + idx],
            capture_output=True, text=True, timeout=20)
        return round(int(out.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:  # noqa: BLE001 - no nvidia-smi, or the id does not exist
        return None


def _repair_setting(cfg, key: str, defaults: bool = True):
    """A repair-stage setting as the stage will resolve it, or None if unset.

    Same precedence as repair_pipeline's loader: the last ``key=`` in
    ``repair.set``, else the scene's repair config, else (with ``defaults``)
    the stage's method.yaml.
    """
    from omegaconf import OmegaConf

    for item in reversed(list(cfg.repair.set)):
        if str(item).startswith(key + "="):
            return str(item).split("=", 1)[1].strip()
    paths = [Path(cfg.repair.config)]
    if defaults:
        paths.append(ROOT / "subprojects" / "repair_pipeline" / "configs" / "method.yaml")
    for path in paths:
        if path.is_file():
            value = OmegaConf.select(OmegaConf.load(path), key)
            if value is not None:
                return value
    return None


def _repair_backend(cfg) -> str:
    """``model.backend`` as the repair stage will resolve it."""
    return str(_repair_setting(cfg, "model.backend") or "artifixer")


def wait_for_vram(cfg, subcommand: str) -> None:
    """Block until the repair GPU has room, or explain why it will not.

    The 14B needs ~89 GiB; on a shared box another tenant's process can take the
    remainder at any moment, and the failure mode is an OOM 4 minutes into the
    model load. Checking first turns that into a wait with a visible reason.

    With less than ``_VRAM_FULL_GB`` free (a 16 GB card) it also adds
    ``model.low_vram=true`` to ``cfg.repair.set``, unless the user set
    ``model.low_vram`` either way; so call it before building the repair stage.
    """
    if subcommand != "repair":
        return
    # The sizes below are ArtiFixer's. passthrough loads no model, and a custom
    # backend's needs are unknown, so neither waits.
    if _repair_backend(cfg) != "artifixer":
        return
    need = _VRAM_NEED_GB.get(variant(cfg))
    device = str(cfg.model.get("device", "cuda"))
    if need is None or not device.startswith("cuda"):
        return
    free = _free_vram_gb(device)
    if free is None:
        return
    limit = float(cfg.workflow.get("vram_wait_min", 30)) * 60
    waited = 0.0
    while free < need:
        if waited > limit:
            raise RuntimeError(
                f"{device} has {free:.1f} GiB free, the {variant(cfg)} model needs ~{need} GiB; "
                f"gave up after {waited/60:.0f} min. Free the card or set "
                f"workflow.vram_wait_min higher.")
        print(f"waiting for {device}: {free:.1f} GiB free, need ~{need} GiB", flush=True)
        time.sleep(30)
        waited += 30
        free = _free_vram_gb(device)
    full = _VRAM_FULL_GB.get(variant(cfg))
    if full is not None and free < full and _repair_setting(cfg, "model.low_vram", defaults=False) is None:
        print(f"{device}: {free:.1f} GiB free (< {full}), repairing with model.low_vram=true "
              f"(same output)", flush=True)
        cfg.repair.set = [*cfg.repair.set, "model.low_vram=true"]


REFERENCE_MODES = ("photos", "progressive")


def reference_mode(cfg) -> str:
    """How stage 2 picks references: ``photos`` (each trajectory on its own) or
    ``progressive`` (later trajectories may also use frames repaired earlier)."""
    mode = str(cfg.workflow.get("reference_mode", "photos"))
    if mode not in REFERENCE_MODES:
        raise ValueError(f"workflow.reference_mode={mode!r}; expected one of {REFERENCE_MODES}")
    return mode


def progressive_stage(cfg, out_dir: Path | None = None, extra: list[str] | None = None) -> Stage:
    """Select-and-repair in one process (``repair_pipeline.progressive``).

    References depend on what has already been repaired, so they cannot be
    chosen up front the way the ``photos`` mode's select stage does.
    """
    python = envs.resolve(cfg, "repair")
    sets = _repair_sets(cfg) + ["progressive.enabled=true", *(extra or [])]
    if out_dir is not None:
        sets = [x for x in sets if not x.startswith("output.dir=")] + [f"output.dir={out_dir}"]
    argv = ["-m", "repair_pipeline.progressive", "--config", str(cfg.repair.config)]
    for item in sets:
        argv += ["--set", item]
    if cfg.repair.get("limit"):
        argv += ["--limit", str(int(cfg.repair.limit))]
    return Stage("repair:progressive", python, ROOT / "subprojects" / "repair_pipeline", argv)


def _previous_mode(cfg) -> str | None:
    import json

    report = Path(cfg.out) / "workflow.json"
    if not report.is_file():
        return None
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("reference_mode", "photos")
    except ValueError:
        return None


def pipeline_stage(cfg, subcommand: str) -> Stage:
    python = envs.resolve(cfg, "repair")
    argv = ["run.py", subcommand, "--config", str(cfg.repair.config)]
    for item in _repair_sets(cfg):
        argv += ["--set", item]
    if cfg.repair.get("limit"):
        argv += ["--limit", str(int(cfg.repair.limit))]
    return Stage(f"repair:{subcommand}", python, ROOT / "subprojects" / "repair_pipeline", argv)


def run_all(cfg, *args, **kwargs) -> int:
    """find -> select -> repair, stopping at the first failure."""
    report = {}
    if not layout.trajectories(cfg) or cfg.workflow.force_find:
        rc = find_stage(cfg).run()
        report["find"] = rc
        if rc != 0:
            return rc
        if not layout.trajectories(cfg):
            print("find produced no trajectories; check the scene config", file=sys.stderr)
            return 1

    mode = reference_mode(cfg)
    previous = _previous_mode(cfg)
    if (previous and previous != mode and layout.repaired_trajectories(cfg)
            and not cfg.workflow.force_repair):
        # Resuming would keep the other mode's finished trajectories and mix
        # two reference strategies in one output.
        print(f"{layout.repair_out(cfg)} was repaired with reference_mode={previous}; "
              f"use a different `out`, or move it aside before switching to {mode}", file=sys.stderr)
        return 1

    trajs = layout.trajectories(cfg)
    have_refs = layout.selected_trajectories(cfg)
    if mode == "progressive":
        print("\nreference_mode=progressive: references are picked during repair; no separate select stage")
        report["select"] = "inline"
    elif len(have_refs) == len(trajs) and not cfg.workflow.force_select:
        print(f"\nall {len(trajs)} trajectories already have reference views; skipping select")
        report["select"] = 0
    elif not cfg.workflow.skip_select:
        rc = pipeline_stage(cfg, "select").run()
        report["select"] = rc
        if rc != 0:
            return rc

    # passthrough ignores the prompt, and encoding one needs the text encoder's
    # weights -- exactly what a wiring check without weights does not have.
    if (cfg.workflow.get("auto_caption", True) and not (Path(cfg.out) / "caption.h5").is_file()
            and _repair_backend(cfg) != "passthrough"):
        from .caption_vlm import ensure_caption

        source = ("workflow.caption_text" if (cfg.workflow.get("caption_text") or "").strip()
                  else "the VLM (--set workflow.auto_caption=false to skip)")
        print(f"\ncaptioning the scene from {source}", flush=True)
        try:
            ensure_caption(cfg, Path(cfg.out), kind=str(cfg.workflow.get("caption_kind", "scene")))
        except Exception as exc:  # noqa: BLE001
            # Repairing without the caption silently falls back to a zero
            # prompt, which is a different experiment -- stop instead.
            traceback.print_exc()
            print(f"caption failed: {exc}\n"
                  "fix it, or pass --set workflow.auto_caption=false to repair "
                  "unconditioned on purpose", file=sys.stderr)
            return 1

    wait_for_vram(cfg, "repair")

    if layout.is_complete(cfg) and not cfg.workflow.force_repair:
        print("\nall trajectories already repaired; nothing to do "
              "(use --set workflow.force_repair=true to redo)")
        report["repair"] = 0
    else:
        stage = progressive_stage(cfg) if mode == "progressive" else pipeline_stage(cfg, "repair")
        rc = stage.run()
        report["repair"] = rc
        if rc != 0:
            return rc

    _write_report(cfg, report)
    return 0


def _write_report(cfg, stages: dict) -> None:
    import json

    out = Path(cfg.out) / "workflow.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "out": cfg.out,
        "variant": variant(cfg),
        "reference_mode": reference_mode(cfg),
        "stages": stages,
        "trajectories": [p.name for p in layout.trajectories(cfg)],
        "repaired": [p.name for p in layout.repaired_trajectories(cfg)],
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
