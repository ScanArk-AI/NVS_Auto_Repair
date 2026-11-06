"""YAML config loading with dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

METHOD_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "method.yaml"


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    # Scene configs only carry scene-level choices; method defaults live in
    # configs/method.yaml and are merged underneath.
    cfg = OmegaConf.merge(OmegaConf.load(METHOD_CONFIG), OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    out_dir = cfg.get("output_dir")
    if out_dir is None:
        out_dir = f"outputs/{Path(path).stem}"
    cfg.output.out_dir = str(out_dir)
    return cfg
