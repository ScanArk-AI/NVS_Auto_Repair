"""Candidate views for the single-view ranking: the SfM poses, subsampled per camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    idx: int
    name: str
    pose: np.ndarray          # c2w, OpenCV
    cam: dict                 # COLMAP intrinsics
    pitch_deg: float
    cam_id: int


def _subsample(seq, n):
    if len(seq) <= n:
        return list(seq)
    step = max(1, len(seq) // n)
    return list(seq[::step])[:n]


def build_candidates(cfg, scene) -> list[Candidate]:
    imgs, cams = scene.imgs, scene.cams
    sel = cfg.views.get("cameras")
    allowed = None
    if isinstance(sel, int):
        # a count: keep the cameras with the most images
        counts = {}
        for im in imgs:
            counts[im["cam_id"]] = counts.get(im["cam_id"], 0) + 1
        allowed = set(sorted(counts, key=lambda c: -counts[c])[:sel])
    elif sel is not None:
        allowed = set(int(x) for x in sel)
    if allowed:
        imgs = [im for im in imgs if im["cam_id"] in allowed]
    max_views = int(cfg.views.max_views)

    # The SfM poses themselves, spread evenly over the cameras.
    by_cam = {}
    for im in imgs:
        by_cam.setdefault(im["cam_id"], []).append(im)
    per = max(1, max_views // max(len(by_cam), 1))
    chosen = []
    for cid, lst in by_cam.items():
        lst = sorted(lst, key=lambda x: x["name"])
        chosen += [(im, cid) for im in _subsample(lst, per)]
    chosen = chosen[:max_views]
    return [Candidate(idx=k, name=im["name"], pose=im["c2w"].copy(), cam=cams[cid],
                      pitch_deg=0.0, cam_id=cid)
            for k, (im, cid) in enumerate(chosen)]
