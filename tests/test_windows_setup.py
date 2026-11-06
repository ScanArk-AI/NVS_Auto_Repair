"""windows/nvs_repair_gui/setup.py: status detection, install steps, and the step scripts in real bash."""

import os
import shutil
import subprocess

import pytest

from nvs_repair_gui import setup as S

LISTING = "  NAME            STATE           VERSION\n* Ubuntu-24.04    Running         2\n  docker-desktop  Stopped         2\n"


class FakeWSL:
    """Answers meta/capture by substring; records what run() was asked to do."""

    distro = "Ubuntu-24.04"

    def __init__(self, meta=None, capture=None, run_rc=0):
        self.meta_ans = {"--status": (0, "Default Version: 2"), "-l -v": (0, LISTING), **(meta or {})}
        self.capture_ans = capture or {}
        self.run_rc = run_rc
        self.ran = []

    def meta(self, *args, timeout=60):
        return self.meta_ans.get(" ".join(args), (0, ""))

    def capture(self, script, root=False, timeout=120):
        for key, ans in self.capture_ans.items():
            if key in script:
                return ans
        return (1, "")

    def run(self, script, on_line, root=False):
        self.ran.append((script, root))
        on_line(f"ran {script[:20]}")
        return self.run_rc


READY = {"getent passwd": (0, "me\n"), "nvidia-smi": (0, "NVIDIA GeForce RTX 4090, 24564 MiB\n"),
         "envs/nvs/bin/python": (0, "installed\nweights\n")}


def test_parse_distros():
    assert S.parse_distros(LISTING) == {"Ubuntu-24.04": 2, "docker-desktop": 2}
    assert S.parse_distros("") == {}


def test_status_ready():
    st = S.status(FakeWSL(capture=READY))
    assert st.ready and st.weights and not st.hints
    assert st.gpu.startswith("NVIDIA") and S.gpu_memory_gib(st.gpu) == 24.0


@pytest.mark.parametrize("fake, hint", [
    (FakeWSL(meta={"--status": (1, "")}), "wsl --install"),
    (FakeWSL(meta={"-l -v": (0, "  NAME STATE VERSION\n  Debian Running 2\n")}), "wsl --install -d Ubuntu-24.04"),
    (FakeWSL(meta={"-l -v": (0, "  NAME STATE VERSION\n* Ubuntu-24.04 Stopped 1\n")}), "--set-version"),
    (FakeWSL(capture={"getent passwd": (0, "")}), "create a user"),
    (FakeWSL(capture={"getent passwd": (0, "me\n")}), "NVIDIA driver"),
])
def test_status_hints(fake, hint):
    st = S.status(fake)
    assert not st.ready and any(hint in h for h in st.hints)


def test_install_stops_with_hints_before_wsl_is_ready():
    lines = []
    fake = FakeWSL(meta={"--status": (1, "")})
    assert S.install(fake, r"C:\nvs\source", lines.append) == 2
    assert fake.ran == [] and "needs you first" in lines[0]


def test_install_runs_only_missing_steps():
    # packages and CUDA present, env missing -> copy (always) + bootstrap
    # the env check also mentions envs/nvs/bin/python, so its answer must come first
    fake = FakeWSL(capture={"sha256sum": (1, ""), **READY, "dpkg -s": (0, ""), "/bin/nvcc": (0, "")})
    lines = []
    assert S.install(fake, r"C:\nvs\source", lines.append, with_weights=False,
                     mirrors={"GH_PROXY": "https://mirror.example/", "EVIL": "x"}) == 0
    scripts = [s for s, _ in fake.ran]
    assert len(scripts) == 2 and "rsync" in scripts[0] and "/mnt/c/nvs/source/" in scripts[0]
    assert "bootstrap.sh --variant 1.3b --skip-weights" in scripts[1]
    assert "GH_PROXY=https://mirror.example/" in scripts[1] and "EVIL" not in scripts[1]
    assert sum("already done" in l for l in lines) == 2 and lines[-1] == "[setup] finished"


def test_system_packages_include_python_extension_headers():
    packages = S.steps("/mnt/d/source", with_weights=False)[0]
    assert "python3.12-dev" in packages.run
    assert "python3.12-dev" in packages.check


def test_install_reports_failure():
    fake = FakeWSL(capture={**READY}, run_rc=100)
    lines = []
    assert S.install(fake, r"C:\nvs\source", lines.append) == 100
    assert fake.ran[0][1] is True                      # apt step runs as root
    assert "FAILED (exit 100)" in lines[-1]


bash = pytest.mark.skipif(not shutil.which("bash") or not shutil.which("rsync"), reason="needs bash and rsync")


def _bash(script, home):
    return subprocess.run(["bash", "-c", script], env={**os.environ, "HOME": str(home)},
                          capture_output=True, text=True)


@bash
def test_copy_step_syncs_and_protects_local_state(tmp_path):
    src, home = tmp_path / "source", tmp_path / "home"
    (src / "configs").mkdir(parents=True)
    (src / "run.py").write_text("v2")
    (src / "configs" / "workflow.yaml").write_text("x: 1")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref")             # running from a clone: not copied
    (src / "build").mkdir()
    (src / "build" / "nvs_repair.exe").write_text("exe")
    repo = home / "nvs_repair"
    for rel, text in (("run.py", "v1"), ("stale.py", "old"), ("outputs/a/b.png", "keep"),
                      ("checkpoints/ArtiFixer/w.pt", "keep"), ("configs/envs.local.yaml", "keep")):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    copy = next(s for s in S.steps(str(src)) if s.name.startswith("copy"))
    r = _bash(copy.run, home)
    assert r.returncode == 0, r.stderr
    assert (repo / "run.py").read_text() == "v2" and (repo / "configs/workflow.yaml").exists()
    assert not (repo / "stale.py").exists()                                   # mirrored
    assert not (repo / ".git").exists() and not (repo / "build").exists()
    for rel in ("outputs/a/b.png", "checkpoints/ArtiFixer/w.pt", "configs/envs.local.yaml"):
        assert (repo / rel).read_text() == "keep"                             # local state survives


@bash
def test_env_step_check_tracks_bootstrap_version_and_weights(tmp_path):
    home = tmp_path / "home"
    py = home / "nvs_repair_runtime/envs/nvs/bin/python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    (home / "nvs_repair_runtime/ArtiFixer/.git").mkdir(parents=True)
    (home / "nvs_repair").mkdir()
    (home / "nvs_repair/bootstrap.sh").write_text("echo v1\n")
    env_step = lambda w: next(s for s in S.steps("/src", with_weights=w) if s.name.startswith("Python"))
    assert _bash(env_step(False).check, home).returncode != 0                 # no stamp yet
    _bash(f"cd {home}/nvs_repair && sha256sum bootstrap.sh | cut -d' ' -f1 > {home}/.nvs_repair_bootstrap.sha256", home)
    assert _bash(env_step(False).check, home).returncode == 0                 # installed, same bootstrap
    assert _bash(env_step(True).check, home).returncode != 0                  # weights still missing
    (home / "nvs_repair/bootstrap.sh").write_text("echo v2\n")
    assert _bash(env_step(False).check, home).returncode != 0                 # bootstrap changed -> re-run
