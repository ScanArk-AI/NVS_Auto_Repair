"""Both stages read COLMAP models (text and binary) with their own reader; they must agree."""

import numpy as np
import pytest


def test_repair_pipeline_reader(colmap_model):
    from repair_pipeline import colmap_io

    d, c2ws = colmap_model
    cams, images = colmap_io.read_model(d)
    assert set(cams) == {1, 2}
    assert cams[1].model_name == "PINHOLE" and cams[1].fx_fy_cx_cy == (500.0, 510.0, 480.0, 272.0)
    assert cams[2].model_name == "OPENCV" and not cams[2].is_fisheye
    by_name = {im.name: im for im in images}
    for name, c2w in c2ws.items():
        np.testing.assert_allclose(np.linalg.inv(by_name[name].w2c), c2w, atol=1e-6)


def test_finder_reader_agrees(colmap_model):
    from repair_view_finder import colmap_io as finder_io
    from repair_pipeline import colmap_io as repair_io

    d, c2ws = colmap_model
    finder_cams, finder_images = finder_io.read_colmap(str(d))
    repair_cams, repair_images = repair_io.read_model(d)
    assert set(finder_cams) == set(repair_cams)
    for cid, cam in repair_cams.items():
        assert finder_io.fx_fy(finder_cams[cid]) == cam.fx_fy_cx_cy[:2]
    repair_c2w = {im.name: np.linalg.inv(im.w2c) for im in repair_images}
    assert {im["name"] for im in finder_images} == set(c2ws)
    for im in finder_images:
        np.testing.assert_allclose(im["c2w"], repair_c2w[im["name"]], atol=1e-6)
        np.testing.assert_allclose(im["c2w"], c2ws[im["name"]], atol=1e-6)


# COLMAP's default camera model is SIMPLE_RADIAL (f, cx, cy, k); RADIAL is
# (f, cx, cy, k1, k2). Both have a single focal length.
_SINGLE_FOCAL = [(2, "SIMPLE_RADIAL", [700.0, 480.0, 272.0, 0.01]),
                 (3, "RADIAL", [700.0, 480.0, 272.0, 0.01, -0.002])]


@pytest.mark.parametrize("model_id, name, params", _SINGLE_FOCAL)
def test_repair_reader_single_focal_models(model_id, name, params):
    from repair_pipeline.colmap_io import Camera

    cam = Camera(id=1, model_id=model_id, width=960, height=544, params=np.array(params))
    assert cam.model_name == name
    assert cam.fx_fy_cx_cy == (700.0, 700.0, 480.0, 272.0)


@pytest.mark.parametrize("model_id, name, params", _SINGLE_FOCAL + [
    (8, "SIMPLE_RADIAL_FISHEYE", [700.0, 480.0, 272.0, 0.01]),
    (9, "RADIAL_FISHEYE", [700.0, 480.0, 272.0, 0.01, -0.002]),
])
def test_finder_reader_single_focal_models(model_id, name, params):
    from repair_view_finder.colmap_io import fx_fy

    assert fx_fy({"model": name, "params": params}) == (700.0, 700.0)
