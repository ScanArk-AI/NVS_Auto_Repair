"""One pipeline run from the GUI's choices.  Owner: A.

Interface agreed between the modules -- app.py builds an ``Options`` and
calls ``run`` with an executor (``wsl.WSL`` or the native one); ``run``
executes ``commands(opts, executor.layout)`` one by one and reports
``Progress`` parsed from the output.

The steps, the same for every backend (only the paths differ, see
``steps.Layout``):

1. (optional, WSL only) copy the scene into the WSL file system -- reading a
   large scene from C: through /mnt is slow;
2. ``run.py make-config`` for the scene, with the chosen path preset and scale;
3. ``python -m nvs_repair.caption_edit`` writes the scene description into
   the generated workflow.yaml -- never through ``--set``, where a colon or
   quote in the text would be parsed as YAML;
4. ``run.py all`` on that config.

A step is argv, not a shell string.  The WSL backend gets each one as a bash
script from ``to_bash``: every word shlex-quoted, except that a ``HomePath``
keeps ``$HOME`` outside the quotes so it expands.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Callable

from . import wsl as _wsl
from .steps import Executor, HomePath, Layout, Step, dev_layout, native_layout   # noqa: F401 (re-exported)
from .wsl import WSL_LAYOUT

PRESETS = ("default", "object-centric", "object-centric-indoor")   # make-config path presets
MODES = ("check", "repair")    # check = model.backend=passthrough (no weights); repair = ArtiFixer 1.3B
STAGES = ("prepare", "find", "select", "repair", "done", "failed")  # every Progress.stage


@dataclass
class Options:
    scene: str                     # Windows path to the scene directory (3DGS ply + COLMAP + images)
    out: str                       # Windows path for the run's output root
    preset: str = "default"        # one of PRESETS
    units: str = "auto"            # "auto", "1" (metric) or a positive number, passed to --units-per-metre
    caption: str = ""              # literal scene description; "" = run without one (auto_caption=false)
    mode: str = "check"            # one of MODES
    limit: int | None = None       # only the first N paths
    copy_to_wsl: bool = False      # rsync the scene into WSL first; ignored where layout.scenes is None


@dataclass
class Progress:
    stage: str                     # one of STAGES; "prepare" = a step before run.py all, message = its label
    done: int = 0                  # paths finished in this stage (0 if unknown)
    total: int = 0                 # paths in this stage (0 if unknown)
    message: str = ""              # the line it was parsed from


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
_PRESET_FLAG = {"default": [], "object-centric": ["--object-centric"],
                "object-centric-indoor": ["--object-centric-indoor"]}


def _check(opts: Options) -> None:
    if opts.preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {opts.preset!r}")
    if opts.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {opts.mode!r}")
    units = str(opts.units).strip()
    if units.lower() != "auto":
        try:
            ok = float(units) > 0
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"units must be 'auto' or a positive number, got {opts.units!r}")
    if opts.limit is not None and (int(opts.limit) != opts.limit or opts.limit < 1):
        raise ValueError(f"limit must be a positive integer or None, got {opts.limit!r}")


def scene_name(opts: Options, layout: Layout | None = None) -> str:
    """make-config names the generated config after the scene directory."""
    layout = layout or WSL_LAYOUT
    name = layout.basename(layout.to_local(opts.scene))
    if not name:
        raise ValueError(f"the scene path has no directory name: {opts.scene!r}")
    return name


def workflow_config(opts: Options, layout: Layout | None = None) -> str:
    """``<repo>/configs/scenes/generated/<name>/workflow.yaml``."""
    layout = layout or WSL_LAYOUT
    return layout.join(layout.repo, "configs", "scenes", "generated", scene_name(opts, layout), "workflow.yaml")


def commands(opts: Options, layout: Layout | None = None) -> list[Step]:
    """The steps to run, in order.  Raises ValueError for options this layout cannot run."""
    layout = layout or WSL_LAYOUT
    _check(opts)
    name = scene_name(opts, layout)
    scene = layout.to_local(opts.scene)
    out = layout.to_local(opts.out)
    py, repo = layout.python, layout.repo
    steps = []

    if opts.copy_to_wsl and layout.scenes is not None:
        target = layout.join(layout.scenes, name)
        src = scene.rstrip("/") + "/"
        steps.append(Step("copy", ["rsync", "-a", "--mkpath", src, target]))
        scene = target

    steps.append(Step("config", [py, "run.py", "make-config", "--scene-root", scene,
                                 "--units-per-metre", str(opts.units).strip(), *_PRESET_FLAG[opts.preset]],
                      cwd=repo))

    cfg = workflow_config(opts, layout)
    steps.append(Step("caption", [py, "-m", "nvs_repair.caption_edit", cfg, opts.caption], cwd=repo))

    # The value is written as a JSON string, which is also a YAML scalar, so a
    # path containing ': ', '#' or a backslash stays that path when OmegaConf
    # parses --set.
    argv = [py, "run.py", "all", "--config", cfg,
            "--set", "out=" + json.dumps(out, ensure_ascii=False), "--set", "model.variant=1.3b"]
    if opts.mode == "check":
        argv += ["--set", "repair.set=[model.backend=passthrough]"]
    if opts.limit is not None:
        argv += ["--limit", str(int(opts.limit))]
    steps.append(Step("pipeline", argv, cwd=repo))
    return steps


def _bash_word(word: str) -> str:
    if isinstance(word, HomePath):
        return '"$HOME"/' + shlex.quote(word.rel) if word.rel else '"$HOME"'
    return shlex.quote(str(word))


def to_bash(step: Step) -> str:
    """One step as a bash script: ``[cd <cwd> && ][K=V ...] argv``, every word quoted."""
    words = [f"{k}={shlex.quote(v)}" for k, v in step.env.items()] + [_bash_word(w) for w in step.argv]
    cmd = " ".join(words)
    return f"cd {_bash_word(step.cwd)} && {cmd}" if step.cwd is not None else cmd


def scripts(opts: Options) -> list[str]:
    """``commands`` for WSL, as the bash scripts ``wsl.WSL.run`` takes."""
    return [to_bash(s) for s in commands(opts, WSL_LAYOUT)]


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
# Shapes taken from real run.py logs (see tests/test_windows_runner.py).
_FINISHED = re.compile(r"^=== (\S+) finished in \S+ \(exit (-?\d+)\)")
_STARTED = re.compile(r"^=== (find|repair:select|repair:repair|repair:progressive)\s+\(")
_COUNT = re.compile(r"^\[(\d+)/(\d+)\]\s")
_SCORED = re.compile(r"^\[rvf\] scored (\d+)/(\d+)")
_STAGE = {"find": "find", "repair:select": "select",
          "repair:repair": "repair", "repair:progressive": "repair"}


def parse_progress(line: str) -> Progress | None:
    """A ``Progress`` if this output line says where the run is, else None."""
    text = line.strip()
    if not text:
        return None
    if text.startswith("Traceback (most recent call last)"):
        return Progress("failed", message=text)
    m = _FINISHED.match(text)
    if m:
        if int(m.group(2)) != 0:
            return Progress("failed", message=text)
        # repair (or progressive) is the last stage run.py all runs
        if m.group(1) in ("repair:repair", "repair:progressive"):
            return Progress("done", message=text)
        return None
    if text.startswith("all trajectories already repaired"):
        return Progress("done", message=text)
    m = _STARTED.match(text)
    if m:
        return Progress(_STAGE[m.group(1)], message=text)
    m = _SCORED.match(text)
    if m:
        return Progress("find", int(m.group(1)), int(m.group(2)), text)
    m = _COUNT.match(text)
    if m:
        # select prints "overlap 85.7%, 7.3s"; repair and progressive add "repair 15.9s"
        stage = "repair" if re.search(r"\brepair \d", text) else "select"
        return Progress(stage, int(m.group(1)), int(m.group(2)), text)
    return None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
class WSLExecutor:
    """A ``wsl.WSL`` seen as an Executor: each step becomes one bash script."""

    def __init__(self, wsl):
        self.wsl = wsl
        self.layout = getattr(wsl, "layout", WSL_LAYOUT)

    def run_step(self, step: Step, on_line: Callable[[str], None]) -> int:
        return self.wsl.run(to_bash(step), on_line)

    def stop(self) -> None:
        self.wsl.stop()


def as_executor(backend) -> Executor:
    """An executor as is; anything else (``wsl.WSL``) wrapped in ``WSLExecutor``."""
    return backend if hasattr(backend, "run_step") else WSLExecutor(backend)


def run(opts: Options, executor, on_line: Callable[[str], None],
        on_progress: Callable[[Progress], None]) -> int:
    """Run ``commands(opts, executor.layout)`` in order, stop at the first failure; returns the exit code.

    ``executor`` is an Executor or a ``wsl.WSL``.  Each step before
    ``run.py all`` is announced as ``Progress("prepare", message=<label>)``;
    after that the stages come from the output.
    """
    executor = as_executor(executor)

    def line(text: str) -> None:
        on_line(text)
        p = parse_progress(text)
        if p is not None:
            on_progress(p)

    for step in commands(opts, executor.layout):
        if step.label != "pipeline":
            on_progress(Progress("prepare", message=step.label))
        rc = executor.run_step(step, line)
        if rc != 0:
            on_progress(Progress("failed", message=f"exit {rc}"))
            return rc
    on_progress(Progress("done", message="exit 0"))
    return 0
