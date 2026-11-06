# Assemble the offline Windows bundle (no WSL) and split it into 7z volumes:
#
#   nvs_repair-<version>-win64\
#     nvs_repair.exe        launcher (starts runtime\pythonw.exe -m nvs_repair_gui)
#     debug.cmd             the same with a console, for error output
#     runtime\              Python 3.12 (python-build-standalone) + torch 2.7.1+cu128 + every dependency
#                           + the five prebuilt CUDA extension wheels (build_wheels.ps1)
#     app\                  this repository at HEAD (git archive), envs.local.yaml -> ..\ArtiFixer
#     ArtiFixer\            ArtiFixer's code at the pinned commit (Apache-2.0)
#     licenses\             licence texts (the INRIA rasterisers are non-commercial)
#     vc_redist.x64.exe     Microsoft's VC++ 2015-2022 x64 runtime installer
#     README.txt
#
#   powershell -ExecutionPolicy Bypass -File windows\native\build_bundle.ps1 [-Version 0.1.0]
#
# Run build_wheels.ps1 first. Needs Python 3.12 (py launcher), git, tar.exe
# (Windows 10+), 7-Zip, network. GAVIS and the weights are NOT bundled: the app
# downloads them on first run (see windows\native\README.md). Log:
# build\build_bundle.log.
param(
    [string]$Version = "",
    [string]$Wheels = "build\wheels",
    [string]$PbsTag = "20261026",          # python-build-standalone release
    [string]$PyMinor = "3.12",
    [string]$VolumeSize = "1900m",         # 7-Zip m = MiB; GitHub Release assets must be < 2 GiB
    [switch]$NoArchive,
    [switch]$NoGpuCheck                    # no NVIDIA GPU here (CI): the self-check only loads the kernels
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$build = Join-Path $root "build"
$cache = Join-Path $build "cache"
New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (-not [IO.Path]::IsPathRooted($Wheels)) { $Wheels = Join-Path $root $Wheels }
Start-Transcript -Path (Join-Path $build "build_bundle.log") -Force | Out-Null

function Say([string]$m) { Write-Host "`n[bundle] $m" }
function Run([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $exe $($argv -join ' ')" }
}
function Fetch([string]$url, [string]$dest) {
    if (-not (Test-Path $dest)) {
        Write-Host "  downloading $url"
        Invoke-WebRequest -Uri $url -OutFile "$dest.part" -UseBasicParsing
        Move-Item "$dest.part" $dest
    }
}

try {
    # --- prerequisites ------------------------------------------------------------
    Say "prerequisites"
    if ($env:NVS_BUILD_PYTHON) { $py = $env:NVS_BUILD_PYTHON } else {      # CI: the setup-python interpreter
        $py = (& py "-$PyMinor" -c "import sys; print(sys.executable)").Trim()
        if ($LASTEXITCODE -ne 0) { throw "Python $PyMinor not found (py launcher)" }
    }
    $own = (& $py -c "import sys; sys.path.insert(0, 'windows'); import nvs_repair_gui; print(nvs_repair_gui.__version__)").Trim()
    if (-not $Version) { $Version = $own }
    if ($Version -ne $own) { throw "version $Version does not match nvs_repair_gui.__version__ ($own)" }
    foreach ($t in @("git", "tar")) { if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t not found" } }
    $sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source
    if (-not $sevenZip -and (Test-Path "$env:ProgramFiles\7-Zip\7z.exe")) { $sevenZip = "$env:ProgramFiles\7-Zip\7z.exe" }
    if (-not $sevenZip -and -not $NoArchive) { throw "7-Zip not found (install it, or pass -NoArchive)" }
    $expected = @("simple_knn", "diff_gaussian_rasterization", "gavis_rasterization", "nerfacc", "gsplat")
    $whls = @(Get-ChildItem $Wheels -Filter *.whl -ErrorAction SilentlyContinue)
    foreach ($e in $expected) {
        if (-not ($whls | Where-Object { $_.Name -like "$e-*" })) { throw "no $e wheel in ${Wheels}: run build_wheels.ps1 first" }
    }
    & $py -m pip install --quiet --disable-pip-version-check pyyaml
    $wf = & $py -c "import yaml, json; d = yaml.safe_load(open('configs/workflow.yaml', encoding='utf-8')); print(json.dumps({'torch': d['torch_version'], 'index': d['torch_index'], 'artifixer': d['artifixer_commit']}))" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "could not read configs/workflow.yaml" }
    Write-Host "  nvs_repair $Version, torch $($wf.torch), ArtiFixer $($wf.artifixer)"

    $name = "nvs_repair-$Version-win64"
    $stage = Join-Path $build "bundle\$name"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null

    # --- runtime: python-build-standalone -------------------------------------------
    Say "Python $PyMinor (python-build-standalone $PbsTag)"
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/$PbsTag" -UseBasicParsing
    $asset = $rel.assets | Where-Object { $_.name -like "cpython-$PyMinor.*-x86_64-pc-windows-msvc-install_only.tar.gz" } | Select-Object -First 1
    if (-not $asset) { throw "no cpython-$PyMinor Windows install_only asset in release $PbsTag" }
    $tgz = Join-Path $cache $asset.name
    Fetch $asset.browser_download_url $tgz
    $unpack = Join-Path $build "bundle\_python"
    if (Test-Path $unpack) { Remove-Item -Recurse -Force $unpack }
    New-Item -ItemType Directory -Force -Path $unpack | Out-Null
    Run tar @("-xzf", $tgz, "-C", $unpack)
    Move-Item (Join-Path $unpack "python") (Join-Path $stage "runtime")
    Remove-Item -Recurse -Force $unpack
    $rpy = Join-Path $stage "runtime\python.exe"
    Write-Host "  $(& $rpy -V)"

    # --- dependencies ---------------------------------------------------------------
    Say "torch $($wf.torch)+cu128, the prebuilt extensions, both stages' requirements"
    $constraints = Join-Path $build "bundle\constraints.txt"
    "torch==$($wf.torch)+cu128" | Set-Content -Path $constraints -Encoding ASCII
    $pip = @("-m", "pip", "--disable-pip-version-check")
    Run $rpy ($pip + @("install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"))
    Run $rpy ($pip + @("install", "--quiet", "-c", $constraints, "--extra-index-url", $wf.index, "torch==$($wf.torch)"))
    # Our wheels first, so requirements naming nerfacc are satisfied by the prebuilt one.
    Run $rpy ($pip + @("install", "--quiet", "--no-deps") + @($whls | ForEach-Object { $_.FullName }))
    foreach ($req in @("subprojects\repair_view_finder\requirements.txt", "subprojects\repair_pipeline\requirements.txt")) {
        Run $rpy ($pip + @("install", "--quiet", "-c", $constraints, "--extra-index-url", $wf.index, "-r", (Join-Path $root $req)))
    }
    # gsplat's and nerfacc's own dependencies (their wheels went in --no-deps), the
    # orchestrator's (pyyaml) and the weights download (huggingface_hub).
    Run $rpy ($pip + @("install", "--quiet", "-c", $constraints, "jaxtyping", "rich", "ninja", "pyyaml", "huggingface_hub"))
    & $rpy @($pip + @("check"))
    if ($LASTEXITCODE -ne 0) { Write-Warning "pip check reports conflicts (above); the self-check below decides" }
    $tv = (& $rpy -c "import torch; print(torch.__version__)").Trim()
    if ($tv -ne "$($wf.torch)+cu128") { throw "bundle torch is $tv, expected $($wf.torch)+cu128" }

    # --- app and ArtiFixer ----------------------------------------------------------
    Say "app (git archive HEAD) and ArtiFixer $($wf.artifixer)"
    $appzip = Join-Path $build "bundle\app.zip"
    Run git @("archive", "--format=zip", "-o", $appzip, "HEAD")
    Expand-Archive -Path $appzip -DestinationPath (Join-Path $stage "app")
    Remove-Item $appzip
    # Relative, so the bundle works wherever it is unpacked; the interpreter is
    # left out on purpose: the stages run under the bundle's own python.exe.
    "# Written by build_bundle.ps1.`npaths:`n  artifixer_repo: ../ArtiFixer`n" |
        Set-Content -Path (Join-Path $stage "app\configs\envs.local.yaml") -Encoding UTF8
    $afsrc = Join-Path $cache "ArtiFixer"
    if (-not (Test-Path "$afsrc\.git")) {
        & git clone --quiet https://github.com/nv-tlabs/ArtiFixer.git $afsrc
        if ($LASTEXITCODE -ne 0) {
            if (-not $env:GH_PROXY) { throw "could not clone ArtiFixer (set GH_PROXY?)" }
            Run git @("clone", "--quiet", ($env:GH_PROXY.TrimEnd("/") + "/https://github.com/nv-tlabs/ArtiFixer.git"), $afsrc)
        }
    }
    Run git @("-C", $afsrc, "checkout", "--quiet", $wf.artifixer)
    $afzip = Join-Path $build "bundle\artifixer.zip"
    Run git @("-C", $afsrc, "archive", "--format=zip", "-o", $afzip, "HEAD")
    Expand-Archive -Path $afzip -DestinationPath (Join-Path $stage "ArtiFixer")
    Remove-Item $afzip

    # --- licences, VC++ runtime, launcher, README --------------------------------------
    Say "licences, VC++ runtime, launcher"
    $lic = Join-Path $stage "licenses"
    New-Item -ItemType Directory -Force -Path $lic | Out-Null
    Copy-Item (Join-Path $Wheels "licenses\*") $lic
    # glm (MIT) is compiled into diff_gaussian_rasterization and gsplat
    Copy-Item (Join-Path $PSScriptRoot "licenses\glm-copying.txt") $lic
    Copy-Item (Join-Path $stage "ArtiFixer\LICENSE") (Join-Path $lic "ArtiFixer-LICENSE")
    Copy-Item (Join-Path $stage "app\LICENSE") (Join-Path $lic "nvs_repair-LICENSE")
    Copy-Item (Join-Path $stage "app\NOTICE.md") (Join-Path $lic "nvs_repair-NOTICE.md")
    Copy-Item (Join-Path $stage "runtime\LICENSE.txt") (Join-Path $lic "Python-LICENSE.txt") -ErrorAction SilentlyContinue
    Fetch "https://aka.ms/vs/17/release/vc_redist.x64.exe" (Join-Path $cache "vc_redist.x64.exe")
    Copy-Item (Join-Path $cache "vc_redist.x64.exe") $stage

    Run $py @("-m", "pip", "install", "--quiet", "--upgrade", "pyinstaller")
    Run $py @("-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed", "--name", "nvs_repair",
              "--distpath", (Join-Path $build "bundle\launcher"), "--workpath", (Join-Path $build "bundle\launcher_work"),
              "--specpath", (Join-Path $build "bundle"), (Join-Path $PSScriptRoot "launcher.py"))
    Copy-Item (Join-Path $build "bundle\launcher\nvs_repair.exe") $stage
    @"
@echo off
rem Start nvs_repair with a console window, so errors stay visible.
cd /d "%~dp0"
set PYTHONPATH=%~dp0app\windows
set PYTHONUTF8=1
set PYTHONNOUSERSITE=1
"%~dp0runtime\python.exe" -m nvs_repair_gui
pause
"@ | Set-Content -Path (Join-Path $stage "debug.cmd") -Encoding ASCII
    @"
nvs_repair $Version for Windows (offline bundle, no WSL)

1. Needs an NVIDIA GPU with driver 572.61 or newer, and the Microsoft VC++
   2015-2022 x64 runtime: run vc_redist.x64.exe once if the app asks for it.
2. Keep this folder on a short path (e.g. C:\nvs_repair) and start nvs_repair.exe.
3. The first time, press "Install / update": it downloads GAVIS (a few MB) and,
   with Mode = Repair, the ArtiFixer 1.3B weights (about 33 GB).
4. Full instructions: app\docs\windows-offline.md.
   Licences: licenses\ and app\NOTICE.md; the Python packages' own licences
   are in runtime\Lib\site-packages\*.dist-info. Parts of this bundle (the 3DGS
   rasterisers) and ArtiFixer's weights are for non-commercial research and
   evaluation only.

If nothing appears when you start nvs_repair.exe, run debug.cmd to see the error.
"@ | Set-Content -Path (Join-Path $stage "README.txt") -Encoding UTF8

    # --- self-check without MSVC or the CUDA toolkit on PATH ---------------------------------
    Say "self-check of the bundle (clean PATH)"
    $saved = @{ PATH = $env:PATH; CUDA_PATH = $env:CUDA_PATH; CUDA_HOME = $env:CUDA_HOME; PYTHONPATH = $env:PYTHONPATH }
    $env:PATH = "$(Join-Path $stage 'runtime');$env:SystemRoot\System32;$env:SystemRoot"
    foreach ($k in @("CUDA_PATH", "CUDA_HOME")) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
    $env:PYTHONPATH = Join-Path $stage "app\windows"
    try {
        & $rpy (Join-Path $PSScriptRoot "check_wheels.py") @(if ($NoGpuCheck) { "--no-gpu" }); $rc1 = $LASTEXITCODE
        & $rpy -c "import nvs_repair_gui, nvs_repair_gui.app, nvs_repair_gui.native, nvs_repair_gui.native_setup; print('gui', nvs_repair_gui.__version__)"; $rc2 = $LASTEXITCODE
        Push-Location (Join-Path $stage "app")
        & $rpy -c "from nvs_repair.config import load; c = load('configs/scenes/example.yaml'); import os; print('ArtiFixer at', c.paths.artifixer_repo, os.path.isdir(c.paths.artifixer_repo))"; $rc3 = $LASTEXITCODE
        Pop-Location
    } finally {
        foreach ($k in $saved.Keys) { if ($saved[$k]) { Set-Item "env:$k" $saved[$k] } else { Remove-Item "env:$k" -ErrorAction SilentlyContinue } }
    }
    if ($rc1 -or $rc2 -or $rc3) { throw "bundle self-check failed (see above)" }

    $size = (Get-ChildItem $stage -Recurse -File | Measure-Object -Property Length -Sum).Sum
    Write-Host ("  {0} unpacked: {1:N1} GB" -f $name, ($size / 1GB))

    # --- split archive ----------------------------------------------------------------
    if (-not $NoArchive) {
        Say "7z volumes of $VolumeSize"
        $arch = Join-Path $build "$name.7z"
        Get-ChildItem $build -Filter "$name.7z*" | Remove-Item -Force
        Run $sevenZip @("a", "-t7z", "-mx=7", "-mmt=on", "-v$VolumeSize", $arch, $stage)
        $vols = Get-ChildItem $build -Filter "$name.7z.*"
        $vols | ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name } |
            Set-Content -Path (Join-Path $build "$name.sha256") -Encoding ASCII
        $vols | ForEach-Object { Write-Host ("  {0}  {1:N2} GB" -f $_.Name, ($_.Length / 1GB)) }
    }
    Say "done: $stage"
}
finally {
    Stop-Transcript | Out-Null
}
