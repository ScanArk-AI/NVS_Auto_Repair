"""Dynamic-object masks shipped next to the training photos.

Two polarities exist across the scenes this pipeline has seen, and mixing them
up silently inverts the filter (it would keep only the frames with a person):

* ``black`` -- rotated captures: black marks the operator **and** the empty corners left
  when a rotated capture was written into an axis-aligned canvas. Every camera
  therefore has a ~12.5% floor that is not a person at all.
* ``white`` -- upright office captures: white marks the operator only, no corner floor.

Decoding a 4000x3000 PNG per photo is the expensive part, and the same scene
reuses the same photos across every trajectory, so fractions are cached to a
JSON next to the images.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_NAME = ".mask_frac_{polarity}.json"


def mask_path_for(images_root: Path, image_name: str) -> Path | None:
    """``mask_<COLMAP path>/<stem>.png`` for either layout.

    ``mask_camera_0_0/x.png`` (single level) and ``mask_map_0/camera_0/x.png``
    (only the first component gets the prefix) both occur upstream.
    """
    parts = image_name.split("/")
    stem = Path(parts[-1]).stem
    for cand in (
        Path(images_root) / ("mask_" + "/".join(parts[:-1])) / f"{stem}.png",
        Path(images_root) / ("mask_" + parts[0]) / f"{stem}.png",
    ):
        if cand.is_file():
            return cand
    return None


def mask_fraction(images_root: Path, image_name: str, invalid: str = "black") -> float | None:
    """Fraction of a photo marked invalid, or None when it has no mask."""
    if invalid == "none":
        return None
    path = mask_path_for(images_root, image_name)
    if path is None:
        return None
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("L"))
    return float((arr >= 128).mean()) if invalid == "white" else float((arr < 128).mean())


def mask_fractions(images_root: Path, names: list[str], invalid: str = "black") -> list[float | None]:
    """Cached per-scene lookup: decodes each mask once, not once per trajectory."""
    cache_path = Path(images_root) / CACHE_NAME.format(polarity=invalid)
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}

    out, dirty = [], False
    for name in names:
        if name in cache:
            out.append(cache[name])
            continue
        frac = mask_fraction(images_root, name, invalid)
        cache[name] = frac
        out.append(frac)
        dirty = True

    if dirty:
        try:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass  # a read-only input tree is fine; we just lose the cache
    return out
