"""YAML config: method defaults underneath, scene config on top, dotted overrides last.

Mirrors ``repair_view_finder``'s convention so the two projects read the same
way:

    repair_pipeline/configs/method.yaml      algorithm + model defaults
    repair_pipeline/configs/scenes/example.yaml scene paths and per-run choices
    --set model.variant=14b                  anything, by dotted path
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

METHOD_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "method.yaml"

_SCENE_FILES = {
    "ply": ("aligned.ply", "source.ply", "output/point_cloud/point_cloud.ply"),
    "sparse": ("sfm_results", "sparse/0", "sparse"),
}


def _resolve_scene(cfg: DictConfig) -> None:
    root = Path(cfg.scene.root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"scene.root does not exist: {root}")
    cfg.scene.root = str(root)

    if cfg.scene.ply:
        cfg.scene.ply = str((root / cfg.scene.ply) if not Path(cfg.scene.ply).is_absolute() else Path(cfg.scene.ply))
    else:
        found = None
        for rel in _SCENE_FILES["ply"]:
            if (root / rel).is_file():
                found = root / rel
                break
        if found is None:
            raise FileNotFoundError(f"no 3DGS ply under {root}; set scene.ply")
        cfg.scene.ply = str(found)

    if cfg.scene.sparse:
        cfg.scene.sparse = str((root / cfg.scene.sparse) if not Path(cfg.scene.sparse).is_absolute() else Path(cfg.scene.sparse))
    else:
        found = next((root / rel for rel in _SCENE_FILES["sparse"] if (root / rel).is_dir()), None)
        if found is None:
            raise FileNotFoundError(f"no COLMAP model under {root}; set scene.sparse")
        cfg.scene.sparse = str(found)

    cfg.scene.images = str(root / cfg.scene.images) if not Path(cfg.scene.images).is_absolute() else cfg.scene.images
    if not cfg.scene.name:
        cfg.scene.name = root.name


# ArtiFixer was trained on scenes scaled so that this maps metres to its units.
METRIC_CAMERA_SCALE = 0.01


def camera_scale_for(repair_root) -> tuple[float, str]:
    """``(model.camera_scale, where it came from)`` for a finder output.

    The finder writes ``scene.json`` with ``units_per_metre`` (1.0 for metric
    captures; estimated or configured otherwise). Outputs without it predate
    the field and are metric.
    """
    scene = Path(repair_root) / "scene.json"
    if scene.is_file():
        upm = float(json.loads(scene.read_text(encoding="utf-8"))["units_per_metre"])
        if not upm > 0:
            raise ValueError(f"{scene}: units_per_metre must be > 0, got {upm}")
        return METRIC_CAMERA_SCALE / upm, f"{scene.name}: units_per_metre={upm:g}"
    return METRIC_CAMERA_SCALE, "metric (no scene.json)"


def weights_dir() -> Path:
    """Where relative weight paths in a config point: ``$NVS_REPAIR_WEIGHTS``,
    else the ``checkpoints/`` directory of the nvs_repair checkout this
    subproject lives in."""
    env = os.environ.get("NVS_REPAIR_WEIGHTS")
    return Path(env).expanduser() if env else METHOD_CONFIG.parents[3] / "checkpoints"


def _weight(path) -> str | None:
    if path is None:
        return None
    p = Path(str(path)).expanduser()
    return str(p if p.is_absolute() else weights_dir() / p)


def resolve_variant(cfg: DictConfig) -> None:
    """Flatten ``model.variant`` into ``model.checkpoint_pt`` / ``model.model_id``.

    Weight paths may be relative (``ArtiFixer/artifixer-1.3b.pt``); they are
    resolved against :func:`weights_dir`. ``model.repo`` falls back to
    ``$ARTIFIXER_REPO``.
    """
    for spec in (cfg.model.get("variants") or {}).values():
        spec.checkpoint_pt = _weight(spec.get("checkpoint_pt"))
        spec.model_id = _weight(spec.get("model_id"))
    if cfg.get("prompt") is not None and cfg.prompt.get("text_encoder"):
        cfg.prompt.text_encoder = _weight(cfg.prompt.text_encoder)
    if not cfg.model.get("repo") and os.environ.get("ARTIFIXER_REPO"):
        cfg.model.repo = os.environ["ARTIFIXER_REPO"]
    variants = cfg.model.pop("variants", None)
    variant = str(cfg.model.variant)
    if variants is not None:
        if variant not in variants:
            raise ValueError(f"model.variant={variant!r} not in {list(variants)}")
        cfg.model.checkpoint_pt = variants[variant].checkpoint_pt
        cfg.model.model_id = variants[variant].model_id
    if not cfg.model.get("checkpoint_pt") or not cfg.model.get("model_id"):
        raise ValueError("model.checkpoint_pt / model.model_id could not be resolved")


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.merge(OmegaConf.load(METHOD_CONFIG), OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    _resolve_scene(cfg)

    resolve_variant(cfg)

    # Relative paths in a config are relative to the *project* root, not the
    # caller's cwd -- otherwise `python run.py` from elsewhere silently writes to
    # a different outputs/ and cannot find the caption file.
    project_root = METHOD_CONFIG.parent.parent
    if cfg.prompt.text_file and not Path(cfg.prompt.text_file).is_absolute():
        cfg.prompt.text_file = str(project_root / cfg.prompt.text_file)

    out = cfg.output.dir
    if out is None:
        out = f"outputs/{Path(path).stem}"
    if not Path(out).is_absolute():
        out = project_root / out
    cfg.output.dir = str(out)

    if not Path(cfg.cache.dir).is_absolute():
        cfg.cache.dir = str(project_root / cfg.cache.dir)

    # The repair root is the `repair/` directory repair_view_finder writes.
    if cfg.input.repair_root is None:
        raise ValueError("input.repair_root must point at <rvf_out>/repair")
    cfg.input.repair_root = str(Path(cfg.input.repair_root).expanduser())

    # An explicit model.camera_scale wins; otherwise follow the scene's units.
    if cfg.model.get("camera_scale") is None:
        cfg.model.camera_scale, cfg.model.camera_scale_source = camera_scale_for(cfg.input.repair_root)
    else:
        cfg.model.camera_scale_source = "config"
    return cfg


def dump(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)
