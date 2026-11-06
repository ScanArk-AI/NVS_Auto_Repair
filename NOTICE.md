# Third-party components

nvs_repair is licensed under Apache-2.0 (see `LICENSE`). It contains two small
adapted files and depends on code and model weights that keep their own terms --
several of them **non-commercial**. Anything fetched at install time stays under
its own licence; nothing below is relicensed by this repository.

## Vendored source (inside the repo)

| Path | What | Origin |
|---|---|---|
| `subprojects/repair_pipeline/repair_pipeline/camera_rays.py` | camera-ray / PRoPE conditioning helpers | adapted from [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer), Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Apache-2.0; modified |
| `subprojects/repair_pipeline/repair_pipeline/frustum.py` | Gaussian-frustum reference selection | from ArtiFixer commit `abe7db7`, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Apache-2.0 |

## Not vendored, fetched at setup time

| Component | How it is used | Terms |
|---|---|---|
| [GAVIS](https://github.com/xsj01/gavis) (`6bc386f`) | cloned by `bootstrap.sh`, installed with `--no-deps`; visibility / uncertainty core of the finder | the repository has no licence file -- ask the authors before redistributing |
| [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) (`dr_aa`, `9c5c202`, with [glm](https://github.com/g-truc/glm)) | cloned and built by `bootstrap.sh` | INRIA/MPII Gaussian-Splatting licence, **non-commercial** |
| [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) (`86710c2`) | cloned and built by `bootstrap.sh` | INRIA/MPII Gaussian-Splatting licence, **non-commercial** |
| [gavis_rasterizer](https://github.com/xsj01/gavis_rasterizer) (`9fe5f62`) | cloned and built by `bootstrap.sh` (package `gavis_rasterization`) | ships the INRIA/MPII Gaussian-Splatting licence, **non-commercial** |
| [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer) (`a392c4d`) | cloned by `bootstrap.sh`, imported for the causal KV-cache pipeline and PRoPE transformer | Apache-2.0 (code) |
| [ArtiFixer weights](https://huggingface.co/nvidia/ArtiFixer) | `checkpoints/ArtiFixer/*.pt` | NVIDIA non-commercial licence |
| [Wan2.1 T2V](https://huggingface.co/Wan-AI) 1.3B / 14B Diffusers | base model + text encoder | Apache-2.0 |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | installed from PyPI by `bootstrap.sh`; renders opacity for the repair stage | Apache-2.0 |
| [nerfacc](https://github.com/nerfstudio-project/nerfacc) | installed from PyPI (finder requirements) | MIT |

## The Windows offline bundle

`nvs_repair-<version>-win64` (built by `windows/native/build_bundle.ps1`) ships,
besides this repository's code, compiled or copied third-party components. Their
licence texts are in the bundle's `licenses\` folder or next to the component:

| Component | In the bundle | Terms |
|---|---|---|
| CPython 3.12 ([python-build-standalone](https://github.com/astral-sh/python-build-standalone) build, with Tcl/Tk) | `runtime\` | PSF licence; bundled libraries under their own terms (`licenses\Python-LICENSE.txt`) |
| PyTorch 2.7.1+cu128, including NVIDIA CUDA runtime libraries | `runtime\Lib\site-packages\torch` | BSD-3-Clause; the CUDA libraries under NVIDIA's CUDA redistribution terms (`torch-*.dist-info`) |
| diff-gaussian-rasterization, simple-knn, gavis_rasterizer (compiled) | `runtime\Lib\site-packages` | INRIA/MPII Gaussian-Splatting licence, **non-commercial** (`licenses\`) |
| [glm](https://github.com/g-truc/glm) (compiled into the rasteriser and gsplat) | as above | MIT (`licenses\glm-copying.txt`) |
| gsplat 1.5.3, nerfacc 0.5.3 (compiled) | `runtime\Lib\site-packages` | Apache-2.0, MIT (`licenses\`) |
| Other Python packages (diffusers, transformers, numpy, ...) | `runtime\Lib\site-packages` | each under its own licence, in its `*.dist-info` folder |
| ArtiFixer code (`a392c4d`) | `ArtiFixer\` | Apache-2.0 (`licenses\ArtiFixer-LICENSE`) |
| Microsoft Visual C++ 2015-2022 redistributable | `vc_redist.x64.exe` | Microsoft's redistributable licence terms |

GAVIS and all model weights are **not** in the bundle: the app downloads them
on the user's machine (GAVIS from its GitHub repository at the pinned commit,
the weights from Hugging Face), as on Linux.

## What is not in the repo

Weights are never committed (`checkpoints/*` is ignored); `bootstrap.sh` downloads
them on the machine that will run the pipeline. Render outputs and caches are
ignored as well.

## Demo media

`docs/media/` holds renders of two scenes (*table*, *picnic*) from the
[Nerfbusters dataset](https://ethanweber.me/nerfbusters) -- F. Warburg*,
E. Weber*, M. Tancik, A. Hołyński, A. Kanazawa, "Nerfbusters: Removing Ghostly
Artifacts from Casually Captured NeRFs", ICCV 2023. No separate licence for the
data was found (the project's code is MIT); the images are shown to illustrate
this tool, with credit to the dataset's authors.
