"""Environment check and first-run downloads for the native (no-WSL) bundle.  Owner: C.

Same interface as setup.py -- ``status(executor)`` and ``install(executor,
source_dir, on_line, with_weights=, mirrors=)`` -- so app.py treats both alike.

The bundle already holds Python, torch and every compiled extension. What it
cannot ship, and so downloads here on first run:

* GAVIS, which has no licence file: pip installs it straight from its authors'
  GitHub archive at the commit pinned in configs/workflow.yaml;
* the ArtiFixer 1.3B weights (NVIDIA, non-commercial) and the Wan2.1 1.3B base
  model, from Hugging Face (``HF_ENDPOINT`` for a mirror).

What the user has to install themselves is reported as ``hints``: an NVIDIA
driver new enough for CUDA 12.8, and the VC++ 2015-2022 x64 runtime.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .steps import Layout, Step

MIN_DRIVER = (572, 61)          # CUDA 12.8 Update 1/2 on Windows; torch 2.7.1+cu128 is built against 12.8
MIRROR_VARS = ("GH_PROXY", "HF_ENDPOINT", "HF_HUB_DISABLE_XET")
ARTIFIXER_REPO = "nvidia/ArtiFixer"
WAN_13B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Runs inside the bundle's Python: the 1.3B checkpoint and its Wan base model.
# A dropped connection is retried; huggingface_hub resumes the partial file.
WEIGHTS_SCRIPT = """
import os, sys, time
from pathlib import Path
import httpx
from huggingface_hub import hf_hub_download, snapshot_download
dest = Path(sys.argv[1])
dest.mkdir(parents=True, exist_ok=True)

def fetch(fn, *args, tries=5, **kw):
    for n in range(1, tries + 1):
        try:
            return fn(*args, **kw)
        except (httpx.TransportError, OSError) as exc:
            if n == tries:
                via = " through the proxy set in HTTP(S)_PROXY" if os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") else ""
                endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
                sys.exit(f"[weights] could not download from {endpoint}{via}: {exc}")
            print(f"[weights] {exc}; retrying ({n}/{tries - 1})", flush=True)
            time.sleep(10 * n)

for name in ("artifixer-1.3b.pt", "config.json", "README.md"):
    print(f"[weights] {name}", flush=True)
    fetch(hf_hub_download, %r, name, local_dir=dest)
print("[weights] %s", flush=True)
fetch(snapshot_download, %r, local_dir=dest / "Wan-AI" / "Wan2.1-T2V-1.3B-Diffusers")
print("[weights] done", flush=True)
""" % (ARTIFIXER_REPO, WAN_13B, WAN_13B)


@dataclass
class NativeStatus:
    driver: str = ""               # "572.83"; "" if nvidia-smi could not be read
    driver_ok: bool = False        # >= MIN_DRIVER
    gpu: str = ""                  # "NVIDIA GeForce RTX 4090, 24564 MiB" (same format as setup.Status)
    vcredist: bool = False         # VC++ 2015-2022 x64 runtime installed
    runtime: bool = False          # the bundle's Python imports torch and sees CUDA
    gavis: bool = False            # GAVIS installed
    weights: bool = False          # 1.3B checkpoint + Wan2.1 1.3B base model present
    hints: list[str] = field(default_factory=list)

    @property
    def installed(self) -> bool:
        return self.runtime and self.gavis

    @property
    def ready(self) -> bool:
        """Enough for a "Check only" run."""
        return self.driver_ok and self.vcredist and bool(self.gpu) and self.installed


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def parse_smi(text: str) -> tuple[str, str]:
    """``nvidia-smi --query-gpu=name,memory.total,driver_version`` -> ``(gpu, driver)``."""
    line = next((l for l in text.splitlines() if l.count(",") >= 2), "")
    if not line:
        return "", ""
    name, mem, driver = (c.strip() for c in line.rsplit(",", 2))
    return f"{name}, {mem}", driver


def driver_ok(driver: str) -> bool:
    m = re.match(r"(\d+)\.(\d+)", driver or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= MIN_DRIVER


def vcredist_installed() -> bool:
    """The VC++ 2015-2022 x64 runtime (torch's DLLs and our extensions need it)."""
    if sys.platform != "win32":
        return True
    import winreg

    for key in (r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                if winreg.QueryValueEx(k, "Installed")[0] == 1:
                    return True
        except OSError:
            continue
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return all((system32 / dll).is_file() for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"))


def weights_dir(layout: Layout) -> str:
    return layout.join(layout.repo, "checkpoints", "ArtiFixer")


def weights_present(layout: Layout) -> bool:
    w = Path(weights_dir(layout))
    return (w / "artifixer-1.3b.pt").is_file() and (w / "Wan-AI" / "Wan2.1-T2V-1.3B-Diffusers" / "model_index.json").is_file()


def gavis_source(layout: Layout) -> tuple[str, str]:
    """``(url, commit)`` of GAVIS from the repo's configs/workflow.yaml."""
    import yaml

    cfg = yaml.safe_load(Path(layout.repo, "configs", "workflow.yaml").read_text(encoding="utf-8"))
    g = cfg["third_party"]["gavis"]
    return str(g["url"]).rstrip("/"), str(g["commit"])


def status(executor, vcredist: Callable[[], bool] = vcredist_installed) -> NativeStatus:
    layout = executor.layout
    st = NativeStatus()
    rc, out = executor.capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], timeout=30)
    if rc == 0:
        st.gpu, st.driver = parse_smi(out)
    st.driver_ok = driver_ok(st.driver)
    st.vcredist = vcredist()
    rc, out = executor.capture([layout.python, "-c",
                                "import json, torch; print(json.dumps([torch.__version__, torch.cuda.is_available()]))"],
                               timeout=180)
    torch_info = None
    if rc == 0:
        try:
            torch_info = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
    st.runtime = bool(torch_info and torch_info[1])
    rc, out = executor.capture([layout.python, "-c", "import importlib.util as u; print(u.find_spec('gavis') is not None)"])
    st.gavis = rc == 0 and out.strip().endswith("True")
    st.weights = weights_present(layout)

    if not st.gpu:
        st.hints.append("No NVIDIA GPU found (nvidia-smi): install or repair the NVIDIA driver.")
    elif not st.driver_ok:
        st.hints.append(f"NVIDIA driver {st.driver or '?'} is too old for CUDA 12.8: update to "
                        f"{MIN_DRIVER[0]}.{MIN_DRIVER[1]} or newer.")
    if not st.vcredist:
        st.hints.append("Install the Microsoft Visual C++ 2015-2022 x64 runtime "
                        "(vc_redist.x64.exe in this folder), then press Re-check.")
    if torch_info is None:
        st.hints.append("The bundled Python cannot load torch: unzip every volume of the download again "
                        "into a short path (e.g. C:\\nvs_repair).")
    elif not st.runtime and st.gpu:
        st.hints.append("torch cannot use the GPU: update the NVIDIA driver, then press Re-check.")
    if st.runtime and not st.gavis:
        st.hints.append("Press Install / update to download GAVIS (a few MB).")
    return st


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def steps(layout: Layout, with_weights: bool = True,
          mirrors: dict[str, str] | None = None) -> list[tuple[str, Callable[[], bool] | None, Step]]:
    """``(name, already_done, step)`` for each download, in order."""
    mirrors = {k: v for k, v in (mirrors or {}).items() if k in MIRROR_VARS and v}
    url, commit = gavis_source(layout)
    archive = f"{url}/archive/{commit}.zip"
    if mirrors.get("GH_PROXY") and archive.startswith("https://github.com/"):
        archive = mirrors["GH_PROXY"].rstrip("/") + "/" + archive
    hf_env = {k: mirrors[k] for k in ("HF_ENDPOINT", "HF_HUB_DISABLE_XET") if k in mirrors}
    out = [
        (f"GAVIS (from its authors' repository, commit {commit[:7]})", None,
         Step("install", [layout.python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                          "--disable-pip-version-check", "--no-cache-dir", archive], cwd=layout.repo)),
    ]
    if with_weights:
        out.append(("ArtiFixer 1.3B weights and the Wan2.1 1.3B base model (about 33 GB)",
                    lambda: weights_present(layout),
                    Step("weights", [layout.python, "-c", WEIGHTS_SCRIPT, weights_dir(layout)],
                         cwd=layout.repo, env=hf_env)))
    return out


def install(executor, source_dir: str, on_line: Callable[[str], None], with_weights: bool = True,
            mirrors: dict[str, str] | None = None, vcredist: Callable[[], bool] = vcredist_installed) -> int:
    """Download what the bundle cannot ship; returns the exit code.

    ``source_dir`` is the bundle's ``app`` folder; it is already in place, so
    nothing is copied (the argument keeps the interface of setup.install).
    """
    st = status(executor, vcredist)
    if not st.runtime:
        for hint in st.hints:
            on_line(f"[setup] needs you first: {hint}")
        return 2
    plan = steps(executor.layout, with_weights, mirrors)
    for i, (name, done, step) in enumerate(plan, start=1):
        tag = f"[setup {i}/{len(plan)}] {name}"
        if (done and done()) or (step.label == "install" and st.gavis):
            on_line(f"{tag}: already done")
            continue
        on_line(f"{tag} ...")
        rc = executor.run_step(step, on_line)
        if rc != 0:
            on_line(f"{tag}: FAILED (exit {rc})")
            return rc
        on_line(f"{tag}: done")
    for hint in st.hints:
        if "Install / update" not in hint:
            on_line(f"[setup] still to do by hand: {hint}")
    on_line("[setup] finished")
    return 0
