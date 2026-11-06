"""Detect WSL / GPU / install state, and the first-time install.  Owner: C.

app.py calls ``status`` to fill the "Environment" row and ``install`` behind
the "Install / update" button.

Two things are left to the user on purpose: installing WSL itself (admin rights
and a reboot) and the distro's first launch (it asks for a user name and
password interactively, which would hang a non-interactive call). ``status``
reports them as ``hints``. Everything after that is ``install``: a list of
steps, each skipped when its check already passes, so pressing the button again
is always safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .wsl import DISTRO, WSL, q, win_to_wsl

APT_PACKAGES = "python3.12-venv python3.12-dev python3-yaml git build-essential rsync wget ca-certificates"
CUDA = "/usr/local/cuda-12.8"
CUDA_KEYRING = "https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
REPO = "$HOME/nvs_repair"
RUNTIME = "$HOME/nvs_repair_runtime"
WEIGHTS = f"{REPO}/checkpoints/ArtiFixer"
STAMP = "$HOME/.nvs_repair_bootstrap.sha256"     # bootstrap.sh the install was made with
MIRROR_VARS = ("GH_PROXY", "HF_ENDPOINT", "HF_HUB_DISABLE_XET")


@dataclass
class Status:
    wsl: bool = False              # wsl.exe works
    distro: bool = False           # the Ubuntu-24.04 distro exists
    wsl2: bool = False             # ... and runs as WSL 2
    user: bool = False             # it has a regular user (first launch done)
    gpu: str = ""                  # "NVIDIA GeForce RTX 4090, 24564 MiB", "" if none visible
    installed: bool = False        # bootstrap finished: env + ArtiFixer checkout
    weights: bool = False          # 1.3B weights present
    hints: list[str] = field(default_factory=list)   # what the user has to do by hand, if anything

    @property
    def ready(self) -> bool:
        """Enough to run a "check" (passthrough) run."""
        return self.wsl and self.distro and self.wsl2 and self.user and bool(self.gpu) and self.installed


def parse_distros(listing: str) -> dict[str, int]:
    """``wsl -l -v`` -> {name: version}. The default distro is marked with ``*``."""
    out = {}
    for line in listing.splitlines():
        cols = line.replace("*", " ").split()
        if len(cols) >= 3 and cols[-1].isdigit() and cols[0].upper() != "NAME":
            out[cols[0]] = int(cols[-1])
    return out


def status(wsl: WSL) -> Status:
    st = Status()
    rc, _ = wsl.meta("--status")
    st.wsl = rc == 0
    if not st.wsl:
        st.hints.append("Install WSL: open PowerShell as administrator, run "
                        f"`wsl --install -d {DISTRO}`, then restart Windows.")
        return st
    distros = parse_distros(wsl.meta("-l", "-v")[1])
    st.distro = wsl.distro in distros
    if not st.distro:
        st.hints.append(f"Install the Ubuntu distro: run `wsl --install -d {wsl.distro}` in PowerShell.")
        return st
    st.wsl2 = distros[wsl.distro] == 2
    if not st.wsl2:
        st.hints.append(f"Switch it to WSL 2: run `wsl --set-version {wsl.distro} 2` in PowerShell.")
        return st
    rc, out = wsl.capture("getent passwd 1000 | cut -d: -f1", root=True)
    st.user = rc == 0 and bool(out.strip())
    if not st.user:
        st.hints.append(f"Open \"{wsl.distro}\" from the Start menu once and create a user name and password.")
        return st
    rc, out = wsl.capture("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1")
    st.gpu = out.strip() if rc == 0 else ""
    if not st.gpu:
        st.hints.append("No NVIDIA GPU is visible inside WSL: update the NVIDIA driver on Windows "
                        "(one with WSL support), then run `wsl --shutdown` and try again.")
    rc, out = wsl.capture(f'test -x {RUNTIME}/envs/nvs/bin/python && test -d {RUNTIME}/ArtiFixer/.git '
                          f'&& echo installed; test -f {WEIGHTS}/artifixer-1.3b.pt '
                          f'&& test -d {WEIGHTS}/Wan-AI/Wan2.1-T2V-1.3B-Diffusers && echo weights')
    st.installed = "installed" in out
    st.weights = "weights" in out
    return st


@dataclass
class Step:
    name: str
    run: str                       # bash script that does the step
    check: str | None = None       # exits 0 when the step is already done; None = always run
    root: bool = False


def steps(source_wsl: str, with_weights: bool = True, mirrors: dict[str, str] | None = None) -> list[Step]:
    """The install steps for a ``source/`` folder at ``source_wsl`` (a WSL path)."""
    env = " ".join(f"{k}={q(v)}" for k, v in (mirrors or {}).items() if k in MIRROR_VARS and v)
    weights_check = (f" && test -f {WEIGHTS}/artifixer-1.3b.pt && test -d {WEIGHTS}/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                     if with_weights else "")
    return [
        Step("system packages",
             check=f"dpkg -s {APT_PACKAGES} >/dev/null 2>&1",
             run=f"apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {APT_PACKAGES}",
             root=True),
        Step("CUDA 12.8 toolkit (to compile three extensions; the driver comes from Windows)",
             check=f"test -x {CUDA}/bin/nvcc",
             run=(f"cd /tmp && wget -q {CUDA_KEYRING} -O cuda-keyring.deb && dpkg -i cuda-keyring.deb "
                  f"&& apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-12-8"),
             root=True),
        # Always: fast, and keeps WSL's copy at the version of the exe next to it.
        Step("copy the source into WSL",
             run=(f"mkdir -p {REPO} && rsync -a --checksum --delete --exclude outputs/ --exclude checkpoints/ArtiFixer "
                  f"--exclude configs/envs.local.yaml --exclude configs/scenes/generated/ --exclude .cache/ "
                  f"--exclude .git/ --exclude build/ "
                  f"{q(source_wsl.rstrip('/') + '/')} {REPO}/")),
        Step("Python environment, extensions, ArtiFixer" + (" and 1.3B weights" if with_weights else ""),
             check=(f"test -x {RUNTIME}/envs/nvs/bin/python && test -d {RUNTIME}/ArtiFixer/.git "
                    f'&& test "$(sha256sum {REPO}/bootstrap.sh | cut -d" " -f1)" = "$(cat {STAMP} 2>/dev/null)"'
                    + weights_check),
             run=(f"cd {REPO} && export PATH={CUDA}/bin:$PATH CUDA_HOME={CUDA} {env} && "
                  f"./bootstrap.sh --variant 1.3b{'' if with_weights else ' --skip-weights'} && "
                  f"sha256sum bootstrap.sh | cut -d' ' -f1 > {STAMP}")),
    ]


def install(wsl: WSL, source_dir: str, on_line: Callable[[str], None], with_weights: bool = True,
            mirrors: dict[str, str] | None = None) -> int:
    """Run the install steps that are not yet satisfied; returns the exit code.

    ``source_dir`` is the Windows path of the release's ``source/`` folder.
    ``mirrors`` may hold GH_PROXY / HF_ENDPOINT / HF_HUB_DISABLE_XET.
    """
    st = status(wsl)
    if not (st.wsl and st.distro and st.wsl2 and st.user):
        for hint in st.hints:
            on_line(f"[setup] needs you first: {hint}")
        return 2
    plan = steps(win_to_wsl(source_dir), with_weights, mirrors)
    for i, step in enumerate(plan, start=1):
        tag = f"[setup {i}/{len(plan)}] {step.name}"
        if step.check and wsl.capture(step.check, root=step.root, timeout=300)[0] == 0:
            on_line(f"{tag}: already done")
            continue
        on_line(f"{tag} ...")
        rc = wsl.run(step.run, on_line, root=step.root)
        if rc != 0:
            on_line(f"{tag}: FAILED (exit {rc})")
            return rc
        on_line(f"{tag}: done")
    on_line("[setup] finished")
    return 0


def gpu_memory_gib(gpu: str) -> float | None:
    """``"NVIDIA GeForce RTX 4090, 24564 MiB"`` -> 24.0 (for the "needs >= 16 GB" warning)."""
    m = re.search(r"(\d+)\s*MiB", gpu)
    return round(int(m.group(1)) / 1024, 1) if m else None
