"""Human-review visualisations: pose map and video encoding."""

from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A font that has Chinese glyphs, tried in order as (path, face index):
# $NVS_REPAIR_FONT ("path" or "path#index" for a .ttc collection), Noto CJK
# on Linux (face 2 = Simplified Chinese), then the Windows system fonts.
CJK_FONTS = (
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 2),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
    (r"C:\Windows\Fonts\msyh.ttc", 0),
    (r"C:\Windows\Fonts\simhei.ttf", 0),
)
_font_warned = False


def _font_candidates():
    env = os.environ.get("NVS_REPAIR_FONT", "").strip()
    if env:
        path, _, index = env.partition("#")
        yield path, int(index) if index.isdigit() else 0
    yield from CJK_FONTS


def cjk_font(size):
    """The first usable CJK font at ``size``; Pillow's default font otherwise.

    Never raises: a missing font only costs the Chinese glyphs in the review
    pictures, it must not stop a run.
    """
    global _font_warned
    for path, index in _font_candidates():
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    if not _font_warned:
        _font_warned = True
        print("[rvf] no CJK font found (set NVS_REPAIR_FONT); "
              "review pictures use Pillow's default font")
    try:
        return ImageFont.load_default(size)
    except TypeError:                     # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def _drawable(text, font):
    """Pillow's bitmap fallback font only encodes Latin-1."""
    if isinstance(font, ImageFont.FreeTypeFont):
        return text
    return text.encode("latin-1", "replace").decode("latin-1")


def overview_image(items, path, title, subtitle="", cols=8, tile_w=320):
    """One contact sheet of the ranked views.

    items: list of (label, rgb_array). A Chinese title is drawn at the top.
    """
    if not items:
        return
    f_title = cjk_font(30)
    f_sub = cjk_font(14)
    f_lab = cjk_font(13)
    tile_w = int(tile_w)
    cols = min(int(cols), len(items))
    tile_h = int(round(tile_w * items[0][1].shape[0] / items[0][1].shape[1]))
    bar_h, title_h = 24, 66
    rows = int(np.ceil(len(items) / cols))
    W = cols * tile_w
    H = title_h + rows * (tile_h + bar_h)
    canvas = Image.new("RGB", (W, H), (14, 14, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 10), _drawable(title, f_title), font=f_title, fill=(240, 240, 240))
    if subtitle:
        draw.text((15, 44), _drawable(subtitle, f_sub), font=f_sub, fill=(150, 150, 158))
    for i, (label, rgb) in enumerate(items):
        x = (i % cols) * tile_w
        y = title_h + (i // cols) * (tile_h + bar_h)
        im = Image.fromarray(rgb).resize((tile_w, tile_h))
        canvas.paste(im, (x, y))
        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + bar_h],
                       fill=(0, 0, 0))
        draw.text((x + 5, y + tile_h + 4), _drawable(label, f_lab), font=f_lab,
                  fill=(235, 235, 235))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def topdown_map(scene, candidates, ranked, defects, path, top=40):
    e1, e2 = scene.e1, scene.e2
    cs = np.array([im["c2w"][:3, 3] for im in scene.imgs])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(cs @ e1, cs @ e2, ".", ms=1, color="0.82", label="source cameras")
    for rank, (cid, gain, new) in enumerate(ranked[:top]):
        cand = candidates[cid]
        p = cand.pose[:3, 3]
        u, v = float(p @ e1), float(p @ e2)
        f = cand.pose[:3, 2]
        fh = np.array([f @ e1, f @ e2])
        n = np.linalg.norm(fh)
        if n > 1e-6:
            fh /= n
        else:
            fh = np.array([1.0, 0.0])
        color = plt.cm.viridis_r(rank / max(top - 1, 1))
        ax.arrow(u, v, 0.35 * fh[0], 0.35 * fh[1], width=0.012,
                 color=color, length_includes_head=True, zorder=5)
        ax.scatter([u], [v], s=26, color=color, edgecolors="k",
                   linewidths=0.4, zorder=6)
        ax.text(u, v, str(rank + 1), fontsize=6, zorder=7)
    ax.set_aspect("equal")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    ax.set_title(f"ranked views ({min(len(ranked), top)}), arrows = forward")

    ax2 = axes[1]
    if defects is not None and getattr(defects, "defect_vox", None) is not None:
        dv = defects.defect_vox
        # Keep the map in the camera footprint; the PLY may contain far
        # outliers that would otherwise dominate the axes.
        ucs, vcs = cs @ e1, cs @ e2
        pad = 3.0
        m = ((defects.vox_xyz @ e1 > ucs.min() - pad)
             & (defects.vox_xyz @ e1 < ucs.max() + pad)
             & (defects.vox_xyz @ e2 > vcs.min() - pad)
             & (defects.vox_xyz @ e2 < vcs.max() + pad))
        ax2.scatter((defects.vox_xyz[m] @ e1), (defects.vox_xyz[m] @ e2),
                    s=0.2, color="0.92")
        if dv.size:
            mv = m[dv]
            ax2.scatter(defects.vox_xyz[dv][mv] @ e1,
                        defects.vox_xyz[dv][mv] @ e2,
                        c=defects.vox_max[dv][mv], cmap="inferno", s=3,
                        vmin=float(defects.thr))
        ax2.set_title(f"high-UQ voxels ({0 if dv is None else len(dv)})")
    else:
        ax2.set_title("no defect volume")
    ax2.set_aspect("equal")
    ax2.set_xlabel("u (m)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_video(frames, path, fps=12):
    """Encode RGB frames as an H.264 mp4."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264",
                            quality=8, macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)
