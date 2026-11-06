"""The repair-model plug-in point.

A backend turns one trajectory's inputs into repaired frames::

    class MyBackend:
        def __init__(self, cfg): ...              # the resolved repair_pipeline config
        def repair(self, clip: Clip) -> np.ndarray:
            ...                                   # (T, H, W, 3) float32 in [0, 1]

``Clip`` (``repair_pipeline.model``) carries everything the pipeline has
prepared: the 3DGS renders and their opacity, the reference views at the
target size, a NeRFStudio-style ``transforms`` dict holding every camera
(targets first, then references), and the encoded caption. A backend may
ignore what it does not need -- an image-to-image model can use ``rgb`` alone.

Choose one with ``model.backend``:

* ``artifixer`` (default) -- ArtiFixer, see ``model.py``.
* ``passthrough`` -- returns the renders unchanged. No GPU model, no weights:
  for checking that a new scene or install is wired up end to end.
* ``package.module:ClassName`` -- your own class, imported by path.
"""

from __future__ import annotations

import importlib
from typing import Protocol

import numpy as np

from .model import Clip


class RepairBackend(Protocol):
    def repair(self, clip: Clip) -> np.ndarray: ...


class Passthrough:
    """Returns the 3DGS renders as they are."""

    def __init__(self, cfg=None):
        pass

    def repair(self, clip: Clip) -> np.ndarray:
        return clip.rgb.astype(np.float32) / 255.0


def make_backend(cfg) -> RepairBackend:
    spec = str(cfg.model.get("backend") or "artifixer")
    if spec == "artifixer":
        from .model import RepairModel

        return RepairModel(cfg)
    if spec == "passthrough":
        return Passthrough(cfg)
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"model.backend={spec!r}: use artifixer, passthrough, or 'package.module:ClassName'")
    return getattr(importlib.import_module(module), name)(cfg)
