"""Check prebuilt CUDA extension wheels in a clean environment (no compiler, no CUDA toolkit).

    python check_wheels.py            # after `pip install <wheels>` into this interpreter
    python check_wheels.py --no-gpu   # a machine without an NVIDIA GPU (CI): only load the kernels

Each extension is imported and run on a tiny CUDA input, so a wheel that was
built for the wrong GPU architecture ("no kernel image is available") or links
a DLL the machine lacks fails here rather than in the middle of a run. gsplat
and nerfacc must load their prebuilt kernels, not JIT-compile them: with no
compiler on PATH a JIT attempt fails, which is what we want to find out.
With ``--no-gpu`` each extension's compiled module is only imported, which
still catches missing DLLs and a JIT fallback but not a wrong architecture.
Prints one line per extension and exits non-zero if any failed.
"""

from __future__ import annotations

import sys
import traceback


def _simple_knn(torch):
    from simple_knn._C import distCUDA2

    pts = torch.rand(1000, 3, device="cuda")
    d = distCUDA2(pts)
    assert d.shape == (1000,) and torch.isfinite(d).all()


def _diff_gaussian_rasterization(torch):
    import diff_gaussian_rasterization._C  # noqa: F401 -- the full rasteriser needs camera setup; loading is the check


def _gavis_rasterization(torch):
    try:
        import gavis_rasterization._C  # noqa: F401
    except ImportError:
        import gavis_rasterizer._C  # noqa: F401


def _gsplat(torch):
    import gsplat.csrc  # noqa: F401 -- the prebuilt kernels; without it gsplat JIT-compiles on first use
    import gsplat

    n = 64
    means = torch.randn(n, 3, device="cuda") * 0.3 + torch.tensor([0.0, 0.0, 3.0], device="cuda")
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
    scales = torch.full((n, 3), 0.05, device="cuda")
    opac = torch.full((n,), 0.8, device="cuda")
    colors = torch.rand(n, 3, device="cuda")
    K = torch.tensor([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], device="cuda")[None]
    img, alpha, _ = gsplat.rasterization(means, quats, scales, opac, colors,
                                         torch.eye(4, device="cuda")[None], K, 64, 64)
    assert img.shape == (1, 64, 64, 3) and alpha.max() > 0


def _nerfacc(torch):
    import nerfacc.csrc  # noqa: F401 -- the prebuilt kernels; without it nerfacc JIT-compiles on first use
    from nerfacc.cuda._backend import _C

    assert _C is not None


CHECKS = {
    "simple_knn": _simple_knn,
    "diff_gaussian_rasterization": _diff_gaussian_rasterization,
    "gavis_rasterization": _gavis_rasterization,
    "gsplat": _gsplat,
    "nerfacc": _nerfacc,
}


# The compiled module of each extension, for --no-gpu.
MODULES = {
    "simple_knn": ["simple_knn._C"],
    "diff_gaussian_rasterization": ["diff_gaussian_rasterization._C"],
    "gavis_rasterization": ["gavis_rasterization._C", "gavis_rasterizer._C"],
    "gsplat": ["gsplat.csrc"],
    "nerfacc": ["nerfacc.csrc"],
}


def _import_any(names):
    import importlib

    for i, mod in enumerate(names):
        try:
            return importlib.import_module(mod)
        except ImportError:
            if i == len(names) - 1:
                raise


def main(argv=None) -> int:
    import torch

    no_gpu = "--no-gpu" in (sys.argv[1:] if argv is None else argv)
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if no_gpu:
        print("--no-gpu: loading the compiled modules only, nothing runs on a GPU")
    elif not torch.cuda.is_available():
        print("FAIL: CUDA is not available (driver?)")
        return 1
    checks = {n: (lambda torch, m=m: _import_any(m)) for n, m in MODULES.items()} if no_gpu else CHECKS
    failed = 0
    for name, check in checks.items():
        try:
            check(torch)
            print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001 -- report every failure, keep going
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
    print("all extensions load and run" if not failed else f"{failed} extension(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
