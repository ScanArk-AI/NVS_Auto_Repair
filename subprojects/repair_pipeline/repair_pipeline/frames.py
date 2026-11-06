"""Frame I/O: read PNG sequences, write lossless PNGs / mp4 / a before-after clip.

Frames going into the reconstruction side are written as PNG on purpose. An H.264
round-trip of these renders measured mean-abs 1.9 with excursions to 68 on 8-bit
values -- fine for viewing, not fine as 3DGS training data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def load_frames(directory: Path, count: int | None = None) -> np.ndarray:
    """``(N, H, W, 3)`` uint8 from ``00000.png…``."""
    from PIL import Image

    files = sorted(Path(directory).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no PNGs in {directory}")
    if count is not None and len(files) != count:
        raise ValueError(f"{directory}: expected {count} frames, found {len(files)}")
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])


def load_gray_frames(directory: Path, count: int | None = None) -> np.ndarray:
    """``(N, H, W)`` uint8 from a single-channel PNG sequence (opacity)."""
    from PIL import Image

    files = sorted(Path(directory).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no PNGs in {directory}")
    if count is not None and len(files) != count:
        raise ValueError(f"{directory}: expected {count} frames, found {len(files)}")
    return np.stack([np.asarray(Image.open(f).convert("L")) for f in files])


def load_neighbors(out_dir: Path, refs: list[dict]) -> np.ndarray:
    """``(M, H, W, 3)`` uint8 from the prepared reference images."""
    from PIL import Image

    return np.stack([np.asarray(Image.open(Path(out_dir) / r["prepared"]).convert("RGB")) for r in refs])


def save_frames(frames: np.ndarray, directory: Path) -> int:
    """Write ``(N,H,W,3)`` float in [0,1] or uint8 to ``00000.png…``."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.png"):
        try:
            old.unlink()  # a shorter rerun must not inherit the previous tail
        except PermissionError as exc:
            raise PermissionError(
                f"cannot replace {old}: the file is in use (open in an image viewer?). "
                f"Close it and run again.") from exc
    arr = frames
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    for i, f in enumerate(arr):
        Image.fromarray(f).save(directory / f"{i:05d}.png")
    return len(arr)


def write_video(frames: np.ndarray, path: Path, fps: int = 15) -> Path:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames if frames.dtype == np.uint8 else (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for f in arr:
            writer.append_data(f)
    return path


def before_after(before: np.ndarray, after: np.ndarray, path: Path, *,
                 before_label: str = "before", after_label: str = "after",
                 height: int = 400, fps: int = 12) -> Path:
    """Two-panel comparison clip: same frame index left and right."""
    from PIL import Image, ImageDraw, ImageFont

    n = min(len(before), len(after))
    to_u8 = lambda a: a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

    def panel(frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(to_u8(frame))
        w = max(2, int(round(img.width * height / img.height)))
        return np.asarray(img.resize((w + (w % 2), height), Image.LANCZOS))

    font = label_font(18)

    strip_h = 30
    rows = []
    for k in range(n):
        left, right = panel(before[k]), panel(after[k])
        canvas = Image.new("RGB", (left.shape[1] + right.shape[1], height + strip_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for img, label in ((left, before_label), (right, after_label)):
            canvas.paste(Image.fromarray(img), (x, strip_h))
            draw.text((x + 8, 6), label, fill=(255, 255, 255), font=font)
            x += img.shape[1]
        rows.append(np.asarray(canvas))
    return write_video(np.stack(rows), path, fps=fps)


def label_font(size: int):
    """Font for the labels drawn on comparison videos.

    ``$NVS_REPAIR_FONT`` if set -- point it at a CJK font (e.g. a WenQuanYi
    .ttc) for non-Latin labels -- otherwise Pillow's built-in font. No font
    ships with the repo.
    """
    from PIL import ImageFont

    path = os.environ.get("NVS_REPAIR_FONT")
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            print(f"NVS_REPAIR_FONT={path!r} could not be loaded; using the default font")
    try:
        return ImageFont.load_default(size=size)   # scalable, Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()
