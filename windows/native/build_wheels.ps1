# Build the CUDA extension wheels the native Windows bundle ships, so that end
# users need neither a compiler nor a CUDA toolkit:
#   simple_knn, diff_gaussian_rasterization, gavis_rasterization (finder),
#   nerfacc 0.5.3 (finder, via GAVIS), gsplat 1.5.3 (repair stage, opacity).
#
#   powershell -ExecutionPolicy Bypass -File windows\native\build_wheels.ps1
#       [-Out build\wheels] [-Arch "8.6;8.9;9.0;12.0+PTX"] [-Jobs 4]
#
# Build machine only (see windows\native\README.md): Visual Studio 2022 Build
# Tools with "Desktop development with C++", CUDA Toolkit 12.8 (nvcc on PATH),
# Python 3.12 with the py launcher, Git, and an NVIDIA GPU for the final check.
# The whole run is logged to build\build_wheels.log.
param(
    [string]$Out = "build\wheels",
    [string]$Arch = "8.6;8.9;9.0;12.0+PTX",    # Ampere, Ada, Hopper, Blackwell (+PTX for newer)
    [int]$Jobs = 4,
    [string]$PyVersion = "3.12",
    [switch]$NoGpuCheck                         # no NVIDIA GPU here (CI): the check only loads the kernels
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$build = Join-Path $root "build"
New-Item -ItemType Directory -Force -Path $build | Out-Null
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $root $Out }
Start-Transcript -Path (Join-Path $build "build_wheels.log") -Force | Out-Null

function Say([string]$m) { Write-Host "`n[wheels] $m" }
# Native commands do not stop the script on failure; check every exit code.
function Run([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "failed (exit $LASTEXITCODE): $exe $($argv -join ' ')" }
}

try {
    # --- prerequisites ----------------------------------------------------------
    Say "prerequisites"
    if ($env:NVS_BUILD_PYTHON) { $py = $env:NVS_BUILD_PYTHON } else {      # CI: the setup-python interpreter
        $py = (& py "-$PyVersion" -c "import sys; print(sys.executable)").Trim()
        if ($LASTEXITCODE -ne 0) { throw "Python $PyVersion not found (install it from python.org with the py launcher)" }
    }
    Write-Host "  python: $py"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found (Git for Windows)" }
    $nvcc = (& nvcc --version | Out-String)
    if ($LASTEXITCODE -ne 0) { throw "nvcc not found: install the CUDA Toolkit 12.8 and reopen the shell" }
    if ($nvcc -notmatch "release 12\.8") { throw "nvcc is not CUDA 12.8 (torch 2.7.1+cu128 needs it):`n$nvcc" }
    Write-Host "  nvcc: $(($nvcc -split "`n" | Select-String 'release').ToString().Trim())"

    # MSVC: load vcvars64 into this session.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "Visual Studio Installer not found: install VS 2022 Build Tools" }
    $vs = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
    if (-not $vs) { throw "no Visual Studio with the C++ x64 tools: add 'Desktop development with C++' to VS 2022 Build Tools" }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe not on PATH after vcvars64.bat" }
    Write-Host "  msvc: $vs"

    # --- build environment ---------------------------------------------------
    Say "build venv with torch 2.7.1+cu128"
    $venv = Join-Path $build "wheelenv"
    if (-not (Test-Path "$venv\Scripts\python.exe")) { Run $py @("-m", "venv", $venv) }
    $vpy = "$venv\Scripts\python.exe"
    $env:PATH = "$venv\Scripts;$env:PATH"
    Run $vpy @("-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel", "ninja", "numpy", "pyyaml")
    Run $vpy @("-m", "pip", "install", "--quiet", "torch==2.7.1+cu128", "--index-url", "https://download.pytorch.org/whl/cu128")
    $tv = (& $vpy -c "import torch; print(torch.__version__)").Trim()
    if ($tv -ne "2.7.1+cu128") { throw "build venv has torch $tv, expected 2.7.1+cu128" }

    # --- sources at pinned commits ---------------------------------------------
    Say "sources"
    $tp = & $vpy -c "import yaml, json; print(json.dumps(yaml.safe_load(open('configs/workflow.yaml', encoding='utf-8'))['third_party']))" | ConvertFrom-Json
    $sources = [ordered]@{
        simple_knn                  = @($tp.simple_knn.url, $tp.simple_knn.commit)
        diff_gaussian_rasterization = @($tp.diff_gaussian_rasterization.url, $tp.diff_gaussian_rasterization.commit)
        gavis_rasterizer            = @($tp.gavis_rasterizer.url, $tp.gavis_rasterizer.commit)
        nerfacc                     = @("https://github.com/nerfstudio-project/nerfacc", "v0.5.3")
        gsplat                      = @("https://github.com/nerfstudio-project/gsplat", "v1.5.3")
    }
    $srcRoot = Join-Path $build "src"
    New-Item -ItemType Directory -Force -Path $srcRoot | Out-Null
    foreach ($name in $sources.Keys) {
        $url, $ref = $sources[$name]
        $dir = Join-Path $srcRoot $name
        if (-not (Test-Path "$dir\.git")) {
            & git clone --quiet $url $dir
            if ($LASTEXITCODE -ne 0) {
                if (-not $env:GH_PROXY -or $url -notlike "https://github.com/*") { throw "could not clone $url (set GH_PROXY to a GitHub mirror prefix?)" }
                Run git @("clone", "--quiet", ($env:GH_PROXY.TrimEnd("/") + "/" + $url), $dir)
            }
        }
        & git -C $dir checkout --quiet --force $ref 2>$null
        if ($LASTEXITCODE -ne 0) { Run git @("-C", $dir, "fetch", "--quiet", "--tags", "origin"); Run git @("-C", $dir, "checkout", "--quiet", "--force", $ref) }
        Run git @("-C", $dir, "submodule", "update", "--quiet", "--init", "--recursive")
        Write-Host "  $name at $((& git -C $dir rev-parse --short HEAD).Trim()) ($ref)"
    }

    # torch 2.7's non-ninja MSVC hook is incompatible with recent setuptools.
    # Keep CUDA sources on nvcc rather than cl.exe's plain C++ path.
    $nerfSetup = Join-Path $srcRoot "nerfacc\setup.py"
    $nerfText = [IO.File]::ReadAllText($nerfSetup)
    if ($nerfText.Contains("use_ninja=False")) {
        [IO.File]::WriteAllText($nerfSetup, $nerfText.Replace("use_ninja=False", "use_ninja=True"), (New-Object Text.UTF8Encoding($false)))
    }

    # --- build ------------------------------------------------------------------
    Say "build wheels for TORCH_CUDA_ARCH_LIST=$Arch"
    $env:DISTUTILS_USE_SDK = "1"           # use the MSVC environment loaded above
    $env:TORCH_CUDA_ARCH_LIST = $Arch
    $env:MAX_JOBS = "$Jobs"
    $env:BUILD_NO_CUDA = "0"               # gsplat / nerfacc: compile the kernels now, not at first use
    # gavis_rasterizer uses M_PI, which MSVC only defines with _USE_MATH_DEFINES;
    # its setup.py fixes the nvcc flags, so pass it through nvcc's and cl's env.
    $env:NVCC_APPEND_FLAGS = "-D_USE_MATH_DEFINES"
    $env:CL = "/D_USE_MATH_DEFINES"
    # The non-ninja MSVC build needs C++17 for torch 2.7 headers.
    $env:_CL_ = "/std:c++17"
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    Get-ChildItem $Out -Filter *.whl | Remove-Item -Force
    foreach ($name in $sources.Keys) {
        Say "  $name"
        Run $vpy @("-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "--verbose", "-w", $Out, (Join-Path $srcRoot $name))
    }
    Get-ChildItem $Out -Filter *.whl | ForEach-Object { Write-Host "  $($_.Name)  $([math]::Round($_.Length / 1MB, 1)) MB" }

    # --- licences shipped with the wheels -----------------------------------------
    $lic = Join-Path $Out "licenses"
    New-Item -ItemType Directory -Force -Path $lic | Out-Null
    foreach ($name in $sources.Keys) {
        $f = Get-ChildItem (Join-Path $srcRoot $name) -File | Where-Object { $_.Name -match "^licen[cs]e" } | Select-Object -First 1
        if ($f) { Copy-Item $f.FullName (Join-Path $lic "$name-$($f.Name)") } else { Write-Host "  WARNING: no licence file in $name" }
    }

    # --- check in a clean env: no MSVC, no CUDA toolkit, like an end user's PC ---------
    Say "check the wheels in a clean environment"
    $chk = Join-Path $build "wheelcheck"
    if (Test-Path $chk) { Remove-Item -Recurse -Force $chk }
    Run $py @("-m", "venv", $chk)
    $cpy = "$chk\Scripts\python.exe"
    Run $cpy @("-m", "pip", "install", "--quiet", "--upgrade", "pip")
    Run $cpy @("-m", "pip", "install", "--quiet", "torch==2.7.1+cu128", "--index-url", "https://download.pytorch.org/whl/cu128")
    Run $cpy @("-m", "pip", "install", "--quiet", "jaxtyping", "rich", "typing_extensions", "numpy", "packaging")
    $wheels = @(Get-ChildItem $Out -Filter *.whl | ForEach-Object { $_.FullName })
    Run $cpy (@("-m", "pip", "install", "--quiet", "--no-deps") + $wheels)
    $saved = @{ PATH = $env:PATH; CUDA_PATH = $env:CUDA_PATH; CUDA_HOME = $env:CUDA_HOME; CL = $env:CL; _CL_ = $env:_CL_; NVCC_APPEND_FLAGS = $env:NVCC_APPEND_FLAGS }
    $env:PATH = "$chk\Scripts;$env:SystemRoot\System32;$env:SystemRoot"
    foreach ($k in @("CUDA_PATH", "CUDA_HOME", "CL", "_CL_", "NVCC_APPEND_FLAGS")) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
    try {
        & $cpy (Join-Path $PSScriptRoot "check_wheels.py") @(if ($NoGpuCheck) { "--no-gpu" })
        $checkRc = $LASTEXITCODE
    } finally {
        foreach ($k in $saved.Keys) { if ($saved[$k]) { Set-Item "env:$k" $saved[$k] } }
    }
    if ($checkRc -ne 0) { throw "wheel check failed (see above)" }

    # --- manifest -------------------------------------------------------------
    $manifest = [ordered]@{
        built = (Get-Date -Format o); torch = $tv; arch = $Arch
        nvcc = (($nvcc -split "`n" | Select-String 'release').ToString().Trim())
        msvc = $vs
        sources = [ordered]@{}
        wheels = [ordered]@{}
    }
    foreach ($name in $sources.Keys) {
        $manifest.sources[$name] = @{ url = $sources[$name][0]; ref = $sources[$name][1]; commit = (& git -C (Join-Path $srcRoot $name) rev-parse HEAD).Trim() }
    }
    Get-ChildItem $Out -Filter *.whl | ForEach-Object { $manifest.wheels[$_.Name] = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower() }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $Out "manifest.json") -Encoding UTF8
    Say "done: $Out (manifest.json, licenses\)"
}
finally {
    Stop-Transcript | Out-Null
}
