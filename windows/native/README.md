# Native Windows bundle (no WSL) -- build notes

The native bundle is a folder with its own Python, torch 2.7.1+cu128 and every
dependency installed, plus prebuilt CUDA extension wheels, so that the user's
PC needs only an NVIDIA driver (>= 572.61) and the VC++ 2015-2022 x64 runtime.
These notes are for the **build machine** that produces it.

## Build machine

| Need | Version | Why |
|---|---|---|
| Windows 10/11 x64 | | |
| NVIDIA GPU + driver | >= 572.61 | the final wheel check runs CUDA kernels |
| Visual Studio 2022 Build Tools | workload "Desktop development with C++" (MSVC v143 + Windows SDK) | compiles the extensions |
| CUDA Toolkit | 12.8.x exactly (nvcc on PATH) | torch 2.7.1+cu128 refuses a different CUDA major; 12.8 also covers Blackwell |
| Python | 3.12 x64 from python.org, with the `py` launcher | the bundle's interpreter version |
| Git for Windows | | fetches sources at pinned commits |
| 7-Zip | | splits the bundle into 1900 MiB volumes (GitHub Release assets must be < 2 GiB) |
| Disk | ~40 GB free | torch alone unpacks to ~6 GB per environment |

Turn on long paths once (admin PowerShell), the build trees are deep:
`New-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`

Behind a firewall set `GH_PROXY` (a GitHub mirror prefix) before running.

## 1. Extension wheels

```powershell
powershell -ExecutionPolicy Bypass -File windows\native\build_wheels.ps1
```

Builds `simple_knn`, `diff_gaussian_rasterization`, `gavis_rasterization`
(sources pinned in `configs/workflow.yaml`), `nerfacc` 0.5.3 and `gsplat` 1.5.3
for `TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0+PTX"` (Ampere, Ada, Hopper,
Blackwell), into `build\wheels\` with `manifest.json` (commits, hashes,
toolchain) and `licenses\`. `gavis_rasterizer` needs `_USE_MATH_DEFINES` under
MSVC; the script passes it through `NVCC_APPEND_FLAGS` / `CL` rather than
patching the source.

It then installs the wheels into a fresh venv with **no MSVC and no CUDA
toolkit on PATH** and runs `check_wheels.py`, which imports each extension and
runs it on a tiny CUDA input. gsplat and nerfacc must load their prebuilt
`csrc` modules; if either would fall back to compiling at first use, the check
fails. Log: `build\build_wheels.log`.

## 2. The bundle

```powershell
powershell -ExecutionPolicy Bypass -File windows\native\build_bundle.ps1
```

Needs the wheels from step 1. It downloads Python 3.12 from
python-build-standalone (`-PbsTag`, default `20261026`) into `runtime\`,
installs torch 2.7.1+cu128 (pinned by a constraints file), the five wheels
(`--no-deps`), both stages' `requirements.txt` and the wheels' own small
dependencies. It then exports this repository's `HEAD` into `app\` (commit
first: uncommitted changes are not included) and ArtiFixer's code at the
pinned commit into `ArtiFixer\`. It writes `app\configs\envs.local.yaml`
with the relative `artifixer_repo: ../ArtiFixer` and no interpreter, so the
stages use the bundle's own `python.exe` wherever it is unpacked. It also adds
`licenses\`, `vc_redist.x64.exe`, `README.txt`, `debug.cmd` and
`nvs_repair.exe`, a small PyInstaller launcher built from `launcher.py` that
starts `runtime\pythonw.exe -m nvs_repair_gui`.

Before it packs anything, it checks the folder with a clean PATH (no MSVC, no
CUDA toolkit): `check_wheels.py`, the GUI modules, and that the config finds
ArtiFixer. It then writes `build\nvs_repair-<version>-win64.7z.001`, `.002`,
... (volumes of `-VolumeSize`, default 1900m = 1900 MiB, about 1.99 GB) and `.sha256`. `-NoArchive`
stops after the folder, which is useful for testing it in place. Log:
`build\build_bundle.log`.

GAVIS and the weights are not in the bundle. On first run the app's
"Install / update" downloads them; see `windows/nvs_repair_gui/native_setup.py`.

## CI

`.github/workflows/windows-native.yml` runs both scripts on a hosted Windows
runner for a `v*` tag (or by hand) and attaches the volumes and `.sha256` to
the Release. The runner installs CUDA 12.8 and uses its own VS 2022, but has no
NVIDIA GPU, so both checks run with `-NoGpuCheck`: every compiled module must
load without MSVC or the CUDA toolkit on PATH. Before announcing a release,
unpack the volumes on a machine with an NVIDIA GPU and do a Repair run. `NVS_BUILD_PYTHON` makes both scripts use that interpreter instead of the
`py` launcher.

## Licences

The INRIA rasteriser, simple-knn and gavis_rasterizer are under the
Gaussian-Splatting licence: redistribution of the compiled wheels is allowed
for non-commercial research and evaluation only, with the full licence text
included (`build\wheels\licenses\`). GAVIS itself has no licence file, so it is
**not** bundled: the app downloads it from its authors' repository on first
run. ArtiFixer's code is Apache-2.0 and is bundled; its weights (NVIDIA,
non-commercial) are downloaded on first run.
