"""windows/nvs_repair_gui/runner.py: commands, progress parsing, the run loop.

The commands are not only compared as strings: each WSL script is executed
by a real bash with HOME pointing at a temp dir, where a stand-in python
records the run.py argv (and creates the config make-config would) and runs
``-m nvs_repair.caption_edit`` for real -- so quoting, $HOME expansion and
the workflow.yaml edit are all exercised.  The same steps are also run
straight from argv, the way the native backend does. On Windows the shell
checks run explicitly inside WSL (with Linux Python and guest paths); the
native check uses sys.executable and a stand-in run.py, not a POSIX shebang.
"""

import json
import ntpath
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nvs_repair_gui import runner as R
from nvs_repair_gui import wsl as W

ROOT = Path(__file__).resolve().parents[1]

yaml = pytest.importorskip("yaml")


# --------------------------------------------------------------------------- #
# progress -- every line below is copied from a real run.py log
# --------------------------------------------------------------------------- #
REAL_LINES = [
    # run.py find on a real scene (orchestrator, --limit-views 4) and a finder log
    ("=== find  (17:38:38)", ("find", 0, 0)),
    ("[rvf] scored 40/120", ("find", 40, 120)),
    ("[rvf] camera 1: 5 trajectories (from 450 candidates)", None),
    ("=== find finished in 149s (exit 0)", None),
    # run.py all, photos mode
    ("=== repair:select  (14:29:21)", ("select", 0, 0)),
    ("[10/22] traj_c1_02: overlap 23.7%, 6.5s", ("select", 10, 22)),
    ("=== repair:select finished in 142s (exit 0)", None),
    ("captioning the scene with the VLM (--set workflow.auto_caption=false to skip)", None),
    ("=== repair:repair  (14:32:57)", ("repair", 0, 0)),
    ("[10/22] traj_c1_02: overlap 23.7%, repair 9.0s, 15.5s", ("repair", 10, 22)),
    ("=== repair:repair finished in 408s (exit 0)", ("done", 0, 0)),
    # run.py all, progressive mode
    ("reference_mode=progressive: references are picked during repair; no separate select stage", None),
    ("=== repair:progressive  (17:48:26)", ("repair", 0, 0)),
    ("[1/2] traj_c0_01: gen 1, 0 repaired refs, direction coverage 85.5% (12 photos) -> 85.5%, repair 15.0s",
     ("repair", 1, 2)),
    ("=== repair:progressive finished in 54s (exit 0)", ("done", 0, 0)),
    # resumed run with nothing left to do
    ("all trajectories already repaired; nothing to do (use --set workflow.force_repair=true to redo)",
     ("done", 0, 0)),
    # failures
    ("geometry.units_per_metre must be > 0 or auto, got -1.0", None),
    ("=== find finished in 7s (exit 1)", ("failed", 0, 0)),
    ("Traceback (most recent call last):", ("failed", 0, 0)),
    # noise
    ("    /opt/nvs_repair_runtime/envs/nvs/bin/python run.py repair --config x.yaml --set output.skip_existing=true",
     None),
    ("Loading weights: 100%|##########| 242/242 [05:41<00:00,  1.41s/it]", None),
    ("waiting for cuda: 79 GiB free, need ~92 GiB", None),
    ("", None),
]


@pytest.mark.parametrize("line, want", REAL_LINES)
def test_parse_progress_on_real_log_lines(line, want):
    p = R.parse_progress(line)
    if want is None:
        assert p is None
    else:
        assert (p.stage, p.done, p.total) == want
        assert p.message == line.strip()


def test_parse_progress_handles_crlf_and_padding():
    assert R.parse_progress("  [3/7] traj_c1_00: overlap 1.0%, repair 9.0s, 15.5s\r").done == 3


# --------------------------------------------------------------------------- #
# commands -- executed by a real bash against a stand-in python
# --------------------------------------------------------------------------- #
def bash_path(path):
    """Use WSL paths only for processes actually running inside WSL."""
    return W.win_to_wsl(str(path)) if os.name == "nt" else str(path)


def bash_prefix():
    if os.name == "nt":
        # Windows' bash.exe is a WSL launcher, not a host POSIX shell.
        return ["wsl.exe", "--exec", "bash"]
    return ["bash"]


def stand_in_home(tmp_path, *, shell):
    """A HOME with the install layout bootstrap.sh produces, python replaced."""
    h = tmp_path / "home dir"
    (h / "nvs_repair").mkdir(parents=True)
    bin_dir = h / "nvs_repair_runtime" / "envs" / "nvs" / "bin"
    bin_dir.mkdir(parents=True)
    py = bin_dir / "python"
    interpreter = "/usr/bin/python3" if shell and os.name == "nt" else sys.executable
    root = bash_path(ROOT) if shell else str(ROOT)
    source = textwrap.dedent(f"""\
        #!{interpreter}
        import json, os, sys
        home = os.environ["HOME"]
        if os.path.basename(sys.argv[0]) == "run.py":
            sys.argv.insert(1, "run.py")
        if sys.argv[1:2] == ["run.py"]:
            with open(os.path.join(home, "calls.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({{"cwd": os.getcwd(), "argv": sys.argv[1:]}}, ensure_ascii=False) + "\\n")
            if sys.argv[2] == "make-config":
                scene = sys.argv[sys.argv.index("--scene-root") + 1].rstrip("/")
                d = os.path.join(os.getcwd(), "configs", "scenes", "generated", os.path.basename(scene))
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "workflow.yaml"), "w", encoding="utf-8") as f:
                    f.write("scene:\\n  root: x\\nworkflow:\\n  auto_caption: true\\n")
            sys.exit(int(os.environ.get("STAND_IN_RC", "0")))
        if sys.argv[1:3] == ["-m", "nvs_repair.caption_edit"]:   # the real thing: step 3's edit
            import runpy
            sys.path.insert(0, {root!r})
            sys.argv = ["caption_edit"] + sys.argv[3:]
            runpy.run_module("nvs_repair.caption_edit", run_name="__main__")
        raise SystemExit("unexpected call: " + repr(sys.argv))
        """)
    py.write_text(source, encoding="utf-8", newline="\n")
    py.chmod(0o755)
    # Native CreateProcess cannot execute a shebang file. Keep run.py fake but
    # use the real interpreter, so commands still execute directly from argv.
    (h / "nvs_repair" / "run.py").write_text(source, encoding="utf-8", newline="\n")
    return h


@pytest.fixture
def home(tmp_path):
    # The scripts run in bash (WSL on Windows) with python3, rsync and PyYAML.
    # Hosted CI runners for Windows have no WSL distro: skip there, run wherever it exists.
    try:
        probe = subprocess.run(bash_prefix() + ["-c", "command -v python3 && command -v rsync && python3 -c 'import yaml'"],
                               capture_output=True, encoding="utf-8", errors="replace", timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"no shell to run the WSL scripts in: {exc}")
    if probe.returncode != 0:
        pytest.skip("shell tests need bash with python3, rsync and PyYAML "
                    f"({'WSL' if os.name == 'nt' else 'this machine'}): {probe.stdout}{probe.stderr}".strip())
    return stand_in_home(tmp_path, shell=True)


@pytest.fixture
def native_home(tmp_path):
    return stand_in_home(tmp_path, shell=False)


def bash(script, home, **env):
    # WSL does not inherit a Windows HOME/PATH as usable Linux paths. Set them
    # inside the guest instead, without altering the product-generated script.
    script = "export " + " ".join(shlex.quote(f"{k}={v}") for k, v in
                                    {"HOME": bash_path(home), **env}.items()) + "; " + script
    return subprocess.run(bash_prefix() + ["-c", script], capture_output=True,
                          encoding="utf-8", errors="replace")


def calls_of(home):
    return [json.loads(l) for l in (home / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


def run_all_scripts(opts, home):
    for script in R.scripts(opts):
        res = bash(script, home)
        assert res.returncode == 0, (script, res.stdout, res.stderr)
    return calls_of(home)


def generated_workflow(home, name):
    return home / "nvs_repair" / "configs" / "scenes" / "generated" / name / "workflow.yaml"


TRICKY_SCENE = "C:\\data\\my scene's $HOME 中文"          # spaces, quote, $, unicode
TRICKY_OUT = "D:\\out dir\\it's #1"
TRICKY_CAPTION = 'A room: walls "white", a desk\'s edge; $(rm -rf /) `x` # not a comment\nline two'


@pytest.mark.parametrize("preset, flag", [("default", None), ("object-centric", "--object-centric"),
                                          ("object-centric-indoor", "--object-centric-indoor")])
@pytest.mark.parametrize("units", ["auto", "1", "3.7"])
def test_make_config_argv(home, preset, flag, units):
    calls = run_all_scripts(R.Options(scene=TRICKY_SCENE, out=TRICKY_OUT, preset=preset, units=units), home)
    mk = calls[0]
    assert mk["cwd"] == bash_path(home / "nvs_repair")
    expected = ["run.py", "make-config", "--scene-root", W.win_to_wsl(TRICKY_SCENE), "--units-per-metre", units]
    assert mk["argv"] == expected + ([flag] if flag else [])


@pytest.mark.parametrize("mode", ["check", "repair"])
@pytest.mark.parametrize("limit", [None, 3])
def test_run_all_argv(home, mode, limit):
    opts = R.Options(scene=TRICKY_SCENE, out=TRICKY_OUT, mode=mode, limit=limit)
    calls = run_all_scripts(opts, home)
    assert [c["argv"][1] for c in calls] == ["make-config", "all"]
    argv = calls[1]["argv"]
    assert calls[1]["cwd"] == bash_path(home / "nvs_repair")
    assert argv[:4] == ["run.py", "all", "--config",
                        bash_path(generated_workflow(home, "my scene's $HOME 中文"))]
    sets = [argv[i + 1] for i, a in enumerate(argv) if a == "--set"]
    out_set = next(s for s in sets if s.startswith("out="))
    assert json.loads(out_set[len("out="):]) == W.win_to_wsl(TRICKY_OUT)
    assert "model.variant=1.3b" in sets
    assert ("repair.set=[model.backend=passthrough]" in sets) == (mode == "check")
    assert (argv[argv.index("--limit") + 1] == str(limit)) if limit else "--limit" not in argv


def test_out_set_value_survives_omegaconf():
    """The --set out=... value is JSON-quoted so OmegaConf keeps odd paths intact."""
    import shlex

    omegaconf = pytest.importorskip("omegaconf")
    for out in ("/home/u/x: y", "/home/u/a#b", "/home/u/it's", "C:\\data\\[odd] dir"):
        script = R.scripts(R.Options(scene="/s/scene", out=out))[-1]
        word = next(w for w in shlex.split(script) if w.startswith("out="))
        assert omegaconf.OmegaConf.from_dotlist([word]).out == W.win_to_wsl(out)


@pytest.mark.parametrize("caption", [TRICKY_CAPTION, "", "   "])
def test_caption_edit_writes_workflow_yaml(home, caption):
    run_all_scripts(R.Options(scene=TRICKY_SCENE, out=TRICKY_OUT, caption=caption), home)
    wf = yaml.safe_load(generated_workflow(home, "my scene's $HOME 中文").read_text(encoding="utf-8"))
    assert wf["scene"] == {"root": "x"}                    # the rest of the file is kept
    if caption.strip():
        assert wf["workflow"]["caption_text"] == caption    # byte-for-byte, nothing parsed
        assert "auto_caption" not in wf["workflow"]
    else:
        assert wf["workflow"] == {"auto_caption": False}


def test_copy_to_wsl_copies_and_uses_the_copy(home, tmp_path):
    src = tmp_path / "src's scene $X"
    (src / "sparse" / "0").mkdir(parents=True)
    (src / "point_cloud.ply").write_text("ply")
    (src / "sparse" / "0" / "images.txt").write_text("# x")
    calls = run_all_scripts(R.Options(scene=str(src), out=str(tmp_path / "out"), copy_to_wsl=True), home)
    copy = home / "nvs_repair_scenes" / src.name
    assert (copy / "point_cloud.ply").read_text() == "ply"
    assert (copy / "sparse" / "0" / "images.txt").is_file()
    assert calls[0]["argv"][3] == bash_path(copy)          # make-config reads the copy
    assert len(R.commands(R.Options(scene=str(src), out="/o", copy_to_wsl=True))) == 4


def test_scripts_are_quoted_not_injectable(home):
    """Nothing the user typed is executed: the $(...) in the caption stays text."""
    marker = home / "pwned"
    target = shlex.quote(bash_path(marker))
    opts = R.Options(scene=f"/s/$(touch {target})", out=f"/o/`touch {target}`",
                     caption=f"$(touch {target})")
    run_all_scripts(opts, home)
    assert not marker.exists()


@pytest.mark.parametrize("bad", [dict(preset="nope"), dict(mode="fast"), dict(units="-1"),
                                 dict(units="metres"), dict(limit=0), dict(scene="relative\\dir")])
def test_commands_reject_bad_options(bad):
    with pytest.raises(ValueError):
        R.commands(R.Options(**{"scene": "C:\\s\\scene", "out": "C:\\o", **bad}))


# --------------------------------------------------------------------------- #
# run -- the loop, against a fake WSL
# --------------------------------------------------------------------------- #
class FakeWSL:
    def __init__(self, outputs, rcs):
        self.outputs, self.rcs, self.scripts = list(outputs), list(rcs), []

    def run(self, script, on_line, root=False):
        self.scripts.append(script)
        for line in self.outputs.pop(0):
            on_line(line)
        return self.rcs.pop(0)


def test_run_streams_lines_and_progress_until_done():
    opts = R.Options(scene="C:\\s\\scene", out="C:\\o")
    outputs = [["scene -> ...workflow.yaml"], [],
               ["=== find  (17:38:38)", "[rvf] scored 20/40", "=== find finished in 9s (exit 0)",
                "=== repair:select  (14:29:21)", "[1/2] t: overlap 1%, 6.5s", "[2/2] t: overlap 1%, 6.5s",
                "=== repair:repair  (14:32:57)", "[1/2] t: overlap 1%, repair 9.0s, 15.5s",
                "=== repair:repair finished in 20s (exit 0)"]]
    wsl = FakeWSL(outputs, [0, 0, 0])
    lines, prog = [], []
    assert R.run(opts, wsl, lines.append, prog.append) == 0
    assert wsl.scripts == R.scripts(opts)
    assert lines == [l for o in outputs for l in o]
    assert [(p.stage, p.message) for p in prog[:2]] == [("prepare", "config"), ("prepare", "caption")]
    assert [(p.stage, p.done, p.total) for p in prog[2:]] == [
        ("find", 0, 0), ("find", 20, 40), ("select", 0, 0), ("select", 1, 2), ("select", 2, 2),
        ("repair", 0, 0), ("repair", 1, 2), ("done", 0, 0), ("done", 0, 0)]


def test_run_stops_at_the_first_failure():
    opts = R.Options(scene="C:\\s\\scene", out="C:\\o", copy_to_wsl=True)
    wsl = FakeWSL([["copying"], ["Traceback (most recent call last):", "SystemExit: no ply"]], [0, 2, 0, 0])
    prog = []
    assert R.run(opts, wsl, lambda _l: None, prog.append) == 2
    assert len(wsl.scripts) == 2                           # make-config failed; nothing after it ran
    assert [p.stage for p in prog] == ["prepare", "prepare", "failed", "failed"]
    assert [p.message for p in prog[:2]] == ["copy", "config"]
    assert prog[-1].message == "exit 2"


def test_every_progress_stage_is_listed():
    stages = {p.stage for line, _ in REAL_LINES if (p := R.parse_progress(line))}
    assert stages | {"prepare"} <= set(R.STAGES)


# --------------------------------------------------------------------------- #
# steps -- the backend-independent form
# --------------------------------------------------------------------------- #
def test_wsl_steps_render_home_unquoted_and_the_rest_quoted():
    steps = R.commands(R.Options(scene="C:\\data\\a b", out="C:\\o", caption="x"))
    assert [s.label for s in steps] == ["config", "caption", "pipeline"]
    assert all(s.env == {} for s in steps)
    assert R.to_bash(steps[1]) == (
        'cd "$HOME"/nvs_repair && "$HOME"/nvs_repair_runtime/envs/nvs/bin/python -m nvs_repair.caption_edit '
        '"$HOME"/\'nvs_repair/configs/scenes/generated/a b/workflow.yaml\' x')
    assert R.to_bash(R.Step("t", ["echo", "$HOME"], env={"A": "1 2"})) == "A='1 2' echo '$HOME'"


def test_wsl_executor_and_layout():
    assert W.WSL.layout is W.WSL_LAYOUT and W.WSL().layout is W.WSL_LAYOUT
    ex = R.as_executor(W.WSL())
    assert isinstance(ex, R.WSLExecutor) and ex.layout is W.WSL_LAYOUT
    native = FakeExecutor(R.dev_layout("C:\\nvs", "C:\\nvs\\venv\\Scripts\\python.exe"), [])
    assert R.as_executor(native) is native


WIN_SCENE = "C:\\Users\\me\\Desktop\\我的 场景"
WIN_OUT = "D:\\结果 out\\run #1"


def test_native_steps_are_plain_windows_argv():
    layout = R.native_layout("C:\\Program Files\\nvs_repair-0.1-win64\\")
    repo = "C:\\Program Files\\nvs_repair-0.1-win64\\app"
    py = "C:\\Program Files\\nvs_repair-0.1-win64\\runtime\\python.exe"
    assert (layout.repo, layout.python, layout.scenes) == (repo, py, None)
    opts = R.Options(scene=WIN_SCENE + "\\", out=WIN_OUT, preset="object-centric", caption='a "b": c',
                     mode="check", limit=2, copy_to_wsl=True)   # copying does not apply natively
    steps = R.commands(opts, layout)
    cfg = repo + "\\configs\\scenes\\generated\\我的 场景\\workflow.yaml"
    assert [(s.label, s.argv, s.cwd) for s in steps] == [
        ("config", [py, "run.py", "make-config", "--scene-root", WIN_SCENE, "--units-per-metre", "auto",
                    "--object-centric"], repo),
        ("caption", [py, "-m", "nvs_repair.caption_edit", cfg, 'a "b": c'], repo),
        ("pipeline", [py, "run.py", "all", "--config", cfg, "--set", "out=" + json.dumps(WIN_OUT, ensure_ascii=False),
                      "--set", "model.variant=1.3b", "--set", "repair.set=[model.backend=passthrough]",
                      "--limit", "2"], repo)]
    assert not any(isinstance(w, R.HomePath) for s in steps for w in s.argv)


def test_native_out_set_value_survives_omegaconf():
    """Backslashes are JSON-escaped ("C:\\\\x") and OmegaConf reads them back as one."""
    omegaconf = pytest.importorskip("omegaconf")
    layout = R.dev_layout("C:\\nvs", "C:\\nvs\\venv\\Scripts\\python.exe")
    for out in (WIN_OUT, "C:\\data\\x", "E:\\a: b\\it's\\[odd]", "C:/mixed/slashes 中文"):
        argv = R.commands(R.Options(scene=WIN_SCENE, out=out), layout)[-1].argv
        word = next(w for w in argv if w.startswith("out="))
        assert omegaconf.OmegaConf.from_dotlist([word]).out == ntpath.normpath(out)


@pytest.mark.parametrize("bad", ["relative\\dir", "\\no\\drive", "/home/u/scene", ""])
def test_native_rejects_paths_that_are_not_absolute_windows_paths(bad):
    layout = R.dev_layout("C:\\nvs", "C:\\nvs\\python.exe")
    with pytest.raises(ValueError):
        R.commands(R.Options(scene=bad, out="C:\\o"), layout)
    with pytest.raises(ValueError):
        R.dev_layout(bad or "x", "C:\\p.exe")
    with pytest.raises(ValueError):
        R.native_layout(bad or "x")


def test_steps_run_straight_from_argv(native_home, tmp_path):
    """What the native executor does -- no shell -- against the same stand-in run.py."""
    home = native_home
    layout = R.Layout("native", str(home / "nvs_repair"), sys.executable,
                      None, lambda p: p, os.path)
    scene = str(tmp_path / "my scene's $HOME 中文")
    opts = R.Options(scene=scene, out=str(tmp_path / "out"), caption=TRICKY_CAPTION, copy_to_wsl=True)
    for step in R.commands(opts, layout):
        res = subprocess.run(step.argv, cwd=step.cwd, capture_output=True, encoding="utf-8",
                             env={**os.environ, "HOME": str(home), "PYTHONPATH": str(ROOT),
                                  "PYTHONUTF8": "1", **step.env})
        assert res.returncode == 0, (step, res.stderr)
    calls = calls_of(home)
    assert [c["argv"][1] for c in calls] == ["make-config", "all"]
    assert calls[0]["argv"][3] == scene
    wf = yaml.safe_load(generated_workflow(home, "my scene's $HOME 中文").read_text(encoding="utf-8"))
    assert wf["workflow"]["caption_text"] == TRICKY_CAPTION


class FakeExecutor:
    def __init__(self, layout, rcs):
        self.layout, self.rcs, self.steps = layout, list(rcs), []

    def run_step(self, step, on_line):
        self.steps.append(step)
        on_line(f"ran {step.label}")
        return self.rcs.pop(0)

    def stop(self):
        pass


def test_run_with_a_native_executor():
    ex = FakeExecutor(R.dev_layout("C:\\nvs", "C:\\nvs\\python.exe"), [0, 0, 0])
    lines, prog = [], []
    assert R.run(R.Options(scene=WIN_SCENE, out=WIN_OUT, copy_to_wsl=True), ex, lines.append, prog.append) == 0
    assert [s.label for s in ex.steps] == ["config", "caption", "pipeline"]
    assert ex.steps[0].argv[0] == "C:\\nvs\\python.exe"
    assert lines == ["ran config", "ran caption", "ran pipeline"]
    assert [(p.stage, p.message) for p in prog] == [("prepare", "config"), ("prepare", "caption"),
                                                    ("done", "exit 0")]


# --------------------------------------------------------------------------- #
# nvs_repair.caption_edit
# --------------------------------------------------------------------------- #
def test_caption_edit_usage(capsys):
    from nvs_repair import caption_edit
    assert caption_edit.main(["only-one"]) == 2
    assert "python -m nvs_repair.caption_edit" in capsys.readouterr().err
