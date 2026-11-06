"""Turn a captured photo into an image the repair model can actually condition on.

Three corrections, each of which fails silently if skipped:

* **Fisheye cameras** (a ~180 deg ``OPENCV_FISHEYE`` rig) cannot be resized
  directly -- squeezing a circular image into 16:9 destroys the geometry. They
  are undistorted to a pinhole with the *target trajectory's* vertical FOV,
  which also makes them comparable with the rendered targets.
* **Rectilinear cameras with a different aspect** (4:3 captures against 16:9
  targets) are centre-cropped before resizing, never stretched.
* **The camera follows the image.** Undistorting, cropping and resizing all
  change the intrinsics; ``prepared_intrinsics`` gives those of the returned
  image. Describing a reference by the source photo's fx/cx instead mis-states
  its field of view (fisheye references come out at 101.6 deg vertical and HD
  ones at 61 deg on our body-worn fisheye + HD rig, not what the photo intrinsics say).
* **The operator is optionally painted out.** These captures are body-worn, so
  the person holding the rig can cover a fifth of a fisheye frame -- and because
  undistorting crops into the centre, their share of the final reference is
  larger still. Masked pixels are replaced with the median of the unmasked ones,
  which reads as empty wall instead of a black blob. Off by default: with a
  tight ``max_mask_frac`` the surviving person is small, and leaving them in
  keeps the pixels honest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .colmap_io import Camera
from .masks import mask_path_for


def prepare_reference(
    image_path: Path,
    cam: Camera,
    target_w: int,
    target_h: int,
    target_fov_deg: float,
    images_root: Path | None = None,
    image_name: str | None = None,
    mask_invalid: str = "black",
    fill_masked: bool = False,
) -> tuple[np.ndarray, float]:
    """Return ``(image at target size, masked fraction after cropping)``."""
    import cv2
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    tgt_fov = np.deg2rad(float(target_fov_deg))

    msk = None
    if images_root is not None and image_name is not None and mask_invalid != "none":
        mpath = mask_path_for(images_root, image_name)
        if mpath is not None:
            arr = np.asarray(Image.open(mpath).convert("L"))
            msk = (arr >= 128) if mask_invalid == "white" else (arr < 128)

    if cam.is_fisheye:
        fx, fy, cx, cy = cam.fx_fy_cx_cy
        k = np.asarray(cam.fisheye_k, dtype=np.float64).reshape(-1, 1)
        fy_new = (target_h / 2.0) / np.tan(tgt_fov / 2.0)
        K_old = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        # Undistort onto a canvas a bit larger than the target, then centre-crop,
        # so the kept region is never padded with black.
        big_w, big_h = int(target_w * 1.6), int(target_h * 1.6)
        K_big = np.array([[fy_new, 0, big_w / 2.0], [0, fy_new, big_h / 2.0], [0, 0, 1]])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_old, k, np.eye(3), K_big, (big_w, big_h), cv2.CV_16SC2)
        out = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if msk is not None:
            msk = cv2.remap(msk.astype(np.uint8) * 255, map1, map2, interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255) >= 128
        ch = int(round(big_w * target_h / target_w))
        top = max(0, (big_h - ch) // 2)
        out = out[top:top + ch]
        if msk is not None:
            msk = msk[top:top + ch]
    else:
        tgt_ar = target_w / target_h
        if w / h > tgt_ar:
            nw = int(round(h * tgt_ar)); left = (w - nw) // 2
            out = img[:, left:left + nw]
            if msk is not None:
                msk = msk[:, left:left + nw]
        else:
            nh = int(round(w / tgt_ar)); top = (h - nh) // 2
            out = img[top:top + nh]
            if msk is not None:
                msk = msk[top:top + nh]

    out = np.asarray(Image.fromarray(out).resize((target_w, target_h), Image.LANCZOS))
    if msk is None:
        return out, 0.0

    msk = np.asarray(Image.fromarray(msk.astype(np.uint8) * 255)
                     .resize((target_w, target_h), Image.NEAREST)) >= 128
    frac = float(msk.mean())
    if frac > 0 and fill_masked:
        keep = out[~msk]
        fill = np.median(keep, axis=0).astype(np.uint8) if keep.size else np.array([128, 128, 128], np.uint8)
        out = out.copy()
        out[msk] = fill
    return out, frac


# --------------------------------------------------------------------------- #
# intrinsics of a prepared reference
# --------------------------------------------------------------------------- #
def prepared_intrinsics(cam: Camera, target_w: int, target_h: int, target_fov_deg: float) -> dict:
    """Camera of the image ``prepare_reference`` returns, in transforms.json keys.

    Mirrors ``prepare_reference`` step by step -- keep the two in sync: fisheye
    -> undistort onto a 1.6x pinhole canvas at the target FOV, crop rows to the
    target aspect, resize; rectilinear -> centre-crop to the target aspect,
    resize, lens terms kept. Checked on that rig by projecting Gaussians
    into the source photo and the prepared image: colours agree best at exactly
    these intrinsics (+-1 % focal, +-3 px principal point).
    """
    tgt_fov = np.deg2rad(float(target_fov_deg))
    if cam.is_fisheye:
        fy_new = (target_h / 2.0) / np.tan(tgt_fov / 2.0)
        big_w, big_h = int(target_w * 1.6), int(target_h * 1.6)
        ch = int(round(big_w * target_h / target_w))
        top = max(0, (big_h - ch) // 2)
        crop_w, crop_h = big_w, min(ch, big_h - top)
        # OpenCV's K puts pixel centres on integers; transforms.json (like
        # COLMAP) on half-integers, hence +0.5.
        fx, fy, cx, cy = fy_new, fy_new, big_w / 2.0 + 0.5, big_h / 2.0 + 0.5 - top
        dist = (0.0, 0.0, 0.0, 0.0)
    else:
        w, h = cam.width, cam.height
        fx, fy, cx, cy = cam.fx_fy_cx_cy
        tgt_ar = target_w / target_h
        if w / h > tgt_ar:
            crop_w, crop_h = int(round(h * tgt_ar)), h
            cx -= (w - crop_w) // 2
        else:
            crop_w, crop_h = w, int(round(w / tgt_ar))
            cy -= (h - crop_h) // 2
        dist = cam.radial_tangential
    sx, sy = target_w / crop_w, target_h / crop_h
    return {"w": int(target_w), "h": int(target_h),
            "fl_x": float(fx * sx), "fl_y": float(fy * sy),
            "cx": float(cx * sx), "cy": float(cy * sy),
            "k1": dist[0], "k2": dist[1], "p1": dist[2], "p2": dist[3]}


def photo_intrinsics(cam: Camera) -> dict:
    """The source photo's own camera -- what the pipeline passed before
    ``refer.neighbor_intrinsics`` existed (``photo``). It mis-states every
    prepared reference's field of view; kept only to reproduce old runs."""
    fx, fy, cx, cy = cam.fx_fy_cx_cy
    return {"w": int(cam.width), "h": int(cam.height),
            "fl_x": float(fx), "fl_y": float(fy), "cx": float(cx), "cy": float(cy)}
