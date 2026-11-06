"""windows/nvs_repair_gui/native.py and native_setup.py (the no-WSL bundle).

The executor runs real subprocesses (the current Python stands in for the
bundle's); the setup logic runs against a fake executor.
"""

import os
import posixpath
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from nvs_repair_gui import native as N
from nvs_repair_gui import native_setup as NS
from nvs_repair_gui.steps import Layout, Step

ROOT = Path(__file__).resolve().parent.parent


def layout(repo=ROOT, python=sys.executable):
    return Layout("native", str(repo), str(python), None, lambda p: p, posixpath)


# --------------------------------------------------------------------------- #
# executor
# --------------------------------------------------------------------------- #
def test_base_env():
    env = N.base_env(layout(), {"PATH": "/usr/bin", "PYTHONPATH": "/elsewhere", "PYTHONHOME": "/x"})
    assert env["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert env["PYTHONUTF8"] == "1" and env["PYTHONUNBUFFERED"] == "1" and env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env


def test_run_step_streams_lines_env_cwd_and_exit_code(tmp_path):
    script = ("import os, sys; print('héllo 中文'); sys.stdout.write('10%\\r50%\\n'); "
              "print(os.getcwd()); print(os.environ['STEP_VAR']); sys.exit(4)")
    lines = []
    rc = N.Native(layout(repo=tmp_path)).run_step(
        Step("pipeline", [sys.executable, "-c", script], env={"STEP_VAR": "from-step"}), lines.append)
    assert rc == 4
    assert lines == ["héllo 中文", "10%", "50%", str(tmp_path), "from-step"]


def test_missing_program():
    lines = []
    assert N.Native(layout()).run_step(Step("x", ["no-such-program-xyz"]), lines.append) == 127
    assert "not found" in lines[0]
    assert N.Native(layout()).capture(["no-such-program-xyz"])[0] == 127


@pytest.mark.skipif(sys.platform == "win32", reason="checks the POSIX process-group path; Windows uses taskkill /T")
def test_stop_ends_the_process_tree(tmp_path):
    marker = tmp_path / "child.pid"
    script = ("import subprocess, sys, time; "
              f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              f"open({str(marker)!r}, 'w').write(str(c.pid)); print('started', flush=True); time.sleep(60)")
    ex = N.Native(layout())
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("rc", ex.run_step(Step("p", [sys.executable, "-c", script]),
                                                                             lambda l: None)))
    t.start()
    for _ in range(100):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.05)
    child = int(marker.read_text())
    ex.stop()
    t.join(10)
    assert not t.is_alive() and result["rc"] != 0
    time.sleep(0.2)
    assert subprocess.run(["kill", "-0", str(child)], capture_output=True).returncode != 0   # grandchild gone too


def test_capture():
    rc, out = N.Native(layout()).capture([sys.executable, "-c", "print('ok')"])
    assert (rc, out.strip()) == (0, "ok")


# --------------------------------------------------------------------------- #
# setup: parsing and checks
# --------------------------------------------------------------------------- #
def test_parse_smi_and_driver():
    assert NS.parse_smi("NVIDIA GeForce RTX 4090, 24564 MiB, 572.83\n") == ("NVIDIA GeForce RTX 4090, 24564 MiB", "572.83")
    assert NS.parse_smi("NVIDIA RTX A6000, Ada, 49140 MiB, 580.12") == ("NVIDIA RTX A6000, Ada, 49140 MiB", "580.12")
    assert NS.parse_smi("") == ("", "")
    assert NS.driver_ok("572.61") and NS.driver_ok("580.1") and not NS.driver_ok("566.36") and not NS.driver_ok("")


class FakeExec:
    def __init__(self, lay, answers=None, run_rc=0):
        self.layout = lay
        self.answers = answers or {}
        self.run_rc = run_rc
        self.ran = []

    def capture(self, argv, timeout=120, env=None):
        joined = " ".join(map(str, argv))
        for key, ans in self.answers.items():
            if key in joined:
                return ans
        return (1, "")

    def run_step(self, step, on_line):
        self.ran.append(step)
        on_line(f"ran {step.label}")
        return self.run_rc


GOOD = {"nvidia-smi": (0, "NVIDIA GeForce RTX 4090, 24564 MiB, 572.83\n"),
        "torch.cuda.is_available": (0, '["2.7.1+cu128", true]\n'),
        "find_spec('gavis')": (0, "True\n")}


def test_status_ready(tmp_path):
    st = NS.status(FakeExec(layout(repo=tmp_path), GOOD), vcredist=lambda: True)
    assert st.ready and st.installed and not st.weights and not st.hints
    assert st.driver == "572.83" and st.gpu.endswith("24564 MiB")


@pytest.mark.parametrize("answers, vc, hint", [
    ({**GOOD, "nvidia-smi": (1, "")}, True, "No NVIDIA GPU"),
    ({**GOOD, "nvidia-smi": (0, "RTX 4090, 24564 MiB, 551.23")}, True, "too old"),
    (GOOD, False, "vc_redist"),
    ({**GOOD, "torch.cuda.is_available": (1, "ImportError: DLL load failed")}, True, "unzip every volume"),
    ({**GOOD, "torch.cuda.is_available": (0, '["2.7.1+cu128", false]')}, True, "cannot use the GPU"),
    ({**GOOD, "find_spec('gavis')": (0, "False")}, True, "download GAVIS"),
])
def test_status_hints(tmp_path, answers, vc, hint):
    st = NS.status(FakeExec(layout(repo=tmp_path), answers), vcredist=lambda: vc)
    assert not st.ready and any(hint in h for h in st.hints), st.hints


def test_weights_present(tmp_path):
    lay = layout(repo=tmp_path)
    w = Path(NS.weights_dir(lay))
    assert not NS.weights_present(lay)
    (w / "Wan-AI/Wan2.1-T2V-1.3B-Diffusers").mkdir(parents=True)
    (w / "Wan-AI/Wan2.1-T2V-1.3B-Diffusers/model_index.json").write_text("{}")
    (w / "artifixer-1.3b.pt").write_text("x")
    assert NS.weights_present(lay)


# --------------------------------------------------------------------------- #
# setup: install
# --------------------------------------------------------------------------- #
def test_steps_use_pinned_gavis_and_mirrors():
    url, commit = NS.gavis_source(layout())
    assert url.startswith("https://github.com/") and len(commit) >= 7
    plan = NS.steps(layout(), with_weights=True,
                    mirrors={"GH_PROXY": "https://mirror.example/", "HF_ENDPOINT": "https://hf.example", "EVIL": "x"})
    (_, _, gavis), (_, _, weights) = plan
    assert gavis.argv[-1] == f"https://mirror.example/{url}/archive/{commit}.zip"
    assert "--no-deps" in gavis.argv and gavis.argv[0] == sys.executable
    assert weights.env == {"HF_ENDPOINT": "https://hf.example"}
    assert weights.argv[-1].endswith(posixpath.join("checkpoints", "ArtiFixer"))
    assert len(NS.steps(layout(), with_weights=False)) == 1
    compile(NS.WEIGHTS_SCRIPT, "weights", "exec")


def test_install_downloads_only_what_is_missing():
    ex = FakeExec(layout(), {**GOOD, "find_spec('gavis')": (0, "False")})
    lines = []
    assert NS.install(ex, str(ROOT), lines.append, with_weights=True, vcredist=lambda: True) == 0
    labels = [s.label for s in ex.ran]
    assert labels[0] == "install" and ("weights" in labels) == (not NS.weights_present(ex.layout))
    assert "--no-cache-dir" in ex.ran[0].argv               # nothing left in %LOCALAPPDATA%\pip after uninstalling
    assert lines[-1] == "[setup] finished"

    ex = FakeExec(layout(), GOOD)                                   # GAVIS there already
    NS.install(ex, str(ROOT), [].append, with_weights=False, vcredist=lambda: True)
    assert ex.ran == []


def test_install_stops_without_a_working_runtime_and_reports_failures():
    lines = []
    ex = FakeExec(layout(), {**GOOD, "torch.cuda.is_available": (1, "")})
    assert NS.install(ex, str(ROOT), lines.append, vcredist=lambda: True) == 2
    assert ex.ran == [] and "needs you first" in lines[0]

    lines = []
    ex = FakeExec(layout(), {**GOOD, "find_spec('gavis')": (0, "False")}, run_rc=1)
    assert NS.install(ex, str(ROOT), lines.append, vcredist=lambda: True) == 1
    assert "FAILED (exit 1)" in lines[-1]


def test_check_wheels_no_gpu_only_imports_the_compiled_modules(monkeypatch, capsys):
    """CI has no NVIDIA GPU: --no-gpu loads each extension's kernels and runs nothing."""
    import importlib.util
    import types

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        __version__="2.7.1+cu128", version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(is_available=lambda: False)))
    for mod in ["simple_knn", "simple_knn._C", "diff_gaussian_rasterization", "diff_gaussian_rasterization._C",
                "gavis_rasterizer", "gavis_rasterizer._C", "gsplat", "gsplat.csrc"]:   # the older gavis package name
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    monkeypatch.setitem(sys.modules, "nerfacc", None)                                # nerfacc's kernels are missing
    path = Path(__file__).resolve().parent.parent / "windows" / "native" / "check_wheels.py"
    spec = importlib.util.spec_from_file_location("check_wheels", path)
    check_wheels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_wheels)

    assert check_wheels.main(["--no-gpu"]) == 1
    out = capsys.readouterr().out
    assert "ok    gavis_rasterization" in out and "ok    gsplat" in out
    assert "FAIL  nerfacc" in out and "1 extension(s) failed" in out
    assert check_wheels.main([]) == 1                                                # without the flag a GPU is required
    assert "CUDA is not available" in capsys.readouterr().out
