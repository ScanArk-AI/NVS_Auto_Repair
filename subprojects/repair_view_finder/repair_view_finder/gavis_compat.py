"""The four GAVIS entry points this project uses, imported from the installed package.

GAVIS (https://github.com/xsj01/gavis) is a dependency, not vendored code; the
commit is pinned in ``configs/workflow.yaml`` and installed by ``bootstrap.sh``.
Two mismatches between that package and its own dependencies are bridged here
rather than by patching it:

* ``gavis.backend.sh.sh_utils`` imports ``scipy.special.sph_harm``, which SciPy
  1.16 removed. GAVIS only ever calls ``sph_harm_y``, so a stub that raises if
  called is enough to let the import succeed.
* GAVIS imports its rasteriser as ``gavis_rasterizer``, but
  ``xsj01/gavis_rasterizer`` builds a package named ``gavis_rasterization``.
  The installed one is aliased to the name GAVIS expects.
"""

from __future__ import annotations

import importlib
import sys

import scipy.special

if not hasattr(scipy.special, "sph_harm"):
    def _sph_harm_removed(*_args, **_kwargs):
        raise NotImplementedError("scipy.special.sph_harm was removed in SciPy 1.16; use sph_harm_y")

    scipy.special.sph_harm = _sph_harm_removed

if "gavis_rasterizer" not in sys.modules:
    try:
        importlib.import_module("gavis_rasterizer")
    except ImportError:
        sys.modules["gavis_rasterizer"] = importlib.import_module("gavis_rasterization")

from gavis import GAVIS  # noqa: E402
from gavis.backend.renderer import render  # noqa: E402
from gavis.backend.scene.gaussian_model import GaussianModel  # noqa: E402
from gavis.pose_utils import pose_to_camera  # noqa: E402

__all__ = ["GAVIS", "GaussianModel", "pose_to_camera", "render"]
