"""Find the right interpreter for each stage, and say so clearly when it is missing.

``bootstrap.sh`` builds one environment that serves both stages, but each stage
is still resolved on its own: a stage can be pointed at a different interpreter
(e.g. an older two-env install), and either way a missing module is reported by
name before anything runs.

Resolution order for a stage interpreter:

1. ``--set envs.<stage>=/path/to/python`` (highest)
2. ``configs/envs.local.yaml``, written by ``bootstrap.sh``
3. ``$WORKFLOW_<STAGE>_PY``
4. the current interpreter, **only if** every module the stage needs imports
5. otherwise: fail with the missing module names
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# What each stage actually imports. Kept deliberately short: these are the
# modules whose absence produces a confusing error deep inside the stage.
STAGE_IMPORTS: dict[str, tuple[str, ...]] = {
    # cv2 is deliberately absent: the finder never imports it.
    # "a|b" = either name will do: xsj01/gavis_rasterizer builds `gavis_rasterization`,
    # older vendored builds were renamed `gavis_rasterizer` (gavis_compat accepts both).
    "finder": ("torch", "omegaconf", "nerfacc", "gavis", "simple_knn",
               "diff_gaussian_rasterization", "gavis_rasterization|gavis_rasterizer"),
    "repair": ("torch", "diffusers", "transformers", "cv2", "h5py", "omegaconf",
               "gsplat", "einops", "scipy", "accelerate"),
}

# Compiled extensions are imported for real (after torch): find_spec alone says
# "ok" for a module whose DLL / .so cannot load, which on Windows is the usual
# failure ("DLL load failed while importing _C") and otherwise surfaces mid-stage.
COMPILED = ("simple_knn", "diff_gaussian_rasterization", "gavis_rasterization", "gavis_rasterizer")

_PROBE = (
    "import json,importlib,importlib.util as u,sys\n"
    "mods=sys.argv[1:]\n"
    "missing=[m for m in mods if all(u.find_spec(a) is None for a in m.split('|'))]\n"
    "torch=None\n"
    "if u.find_spec('torch'):\n"
    "    import torch as t; torch=t.__version__\n"
    f"    for m in mods:\n"
    f"        for a in m.split('|'):\n"
    f"            if a in {COMPILED!r} and m not in missing and u.find_spec(a):\n"
    "                try:\n"
    "                    importlib.import_module(a)\n"
    "                except Exception as e:\n"
    "                    missing.append(f'{a} (cannot load: {type(e).__name__}: {e})'[:200])\n"
    "print(json.dumps({'missing':missing,'torch':torch}))\n"
)


@dataclass
class Probe:
    python: str
    torch: str | None
    missing: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing

    def __str__(self) -> str:
        state = "ok" if self.ok else "missing " + ",".join(self.missing)
        return f"{self.python} (torch {self.torch or '?'}, {state})"


def probe(python: str, stage: str) -> Probe:
    env = {**os.environ, "PYTHONPATH": ""}
    proc = subprocess.run([python, "-c", _PROBE, *STAGE_IMPORTS[stage]],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if proc.returncode != 0:
        return Probe(python, None, [f"interpreter failed: {proc.stderr.strip().splitlines()[-1] if proc.stderr else '?'}"])
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return Probe(python, data["torch"], data["missing"])


def _configured(cfg, stage: str) -> str | None:
    envs = cfg.get("envs") or {}
    value = envs.get(stage)
    if value:
        return str(value)
    return os.environ.get(f"WORKFLOW_{stage.upper()}_PY")


def resolve(cfg, stage: str) -> str:
    """Return the interpreter to run ``stage`` with. Raises with a clear message."""
    candidates: list[str] = []
    configured = _configured(cfg, stage)
    if configured:
        candidates.append(configured)
    candidates.append(sys.executable)

    tried = []
    for python in candidates:
        if not Path(python).exists() and python != sys.executable:
            tried.append(f"{python}: not found")
            continue
        p = probe(python, stage)
        tried.append(str(p))
        if p.ok:
            return python

    raise RuntimeError(
        f"no interpreter can run the '{stage}' stage.\n"
        f"  needs: {', '.join(STAGE_IMPORTS[stage])}\n"
        f"  tried:\n    " + "\n    ".join(tried) + "\n"
        "  fix: run ./bootstrap.sh (writes configs/envs.local.yaml), or pass\n"
        f"       --set envs.{stage}=" + (r"C:\path\to\env\python.exe" if os.name == "nt" else "/path/to/env/bin/python")
    )


def _check_weights(cfg) -> list[str]:
    """The selected variant's weights must be readable from checkpoints/;
    `bootstrap.sh --variant 1.3b` installs only those."""
    from .config import variant

    problems = []
    name = variant(cfg)
    spec = cfg.checkpoints.artifixer[name]
    for key in ("checkpoint_pt", "model_id"):
        raw = str(spec[key])
        path = Path(raw) if Path(raw).is_absolute() else Path(cfg.checkpoints_dir) / raw
        if not path.exists():
            problems.append(f"{name}.{key}: {path}")
    te = Path(str(cfg.checkpoints.text_encoder))
    te = te if te.is_absolute() else Path(cfg.checkpoints_dir) / te
    if not te.exists():
        problems.append(f"text_encoder: {te}")
    return problems


def doctor(cfg) -> int:
    """Print a table of every stage's interpreter and the artifacts they need."""
    rows, failed = [], False
    for stage in ("finder", "repair"):
        try:
            python = resolve(cfg, stage)
        except RuntimeError as exc:
            rows.append((stage, "NOT RESOLVED", "-", "") )
            print(str(exc))
            failed = True
            continue
        p = probe(python, stage)
        rows.append((stage, python, p.torch or "?", "ok" if p.ok else ",".join(p.missing)))
        failed |= not p.ok

    print(f"{'stage':8}{'interpreter':58}{'torch':14}status")
    for stage, python, torch, status in rows:
        print(f"{stage:8}{python:58}{torch:14}{status}")

    print("\nartifacts")
    weight_problems = _check_weights(cfg)
    repo = Path(str(cfg.paths.artifixer_repo)) if cfg.get("paths") else None
    checks = [
        ("checkpoints", weight_problems == [], ", ".join(weight_problems)),
        ("ArtiFixer checkout", bool(repo and (repo / "model_training").is_dir()),
         str(repo) if repo else "paths.artifixer_repo not set"),
        ("find config", Path(cfg.find.config).is_file(), cfg.find.config),
        ("repair config", Path(cfg.repair.config).is_file(), cfg.repair.config),
        ("scene root", Path(cfg.scene.root).is_dir(), cfg.scene.root),
    ]
    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:20}{detail if not ok else ''}")
        failed |= not ok

    print(f"\noutput root: {cfg.out}")
    return 1 if failed else 0
