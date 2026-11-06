# Build the Windows release: build\nvs_repair-<version>-windows.zip holding
#   nvs_repair.exe   the GUI (PyInstaller, one file, no console window)
#   source\          this repository at HEAD (git archive), which the GUI copies into WSL
#   README.txt       where to start
#
#   powershell -ExecutionPolicy Bypass -File windows\build.ps1 [-Version 0.1.0]
#
# Needs Python 3.10+ with tkinter, and git, on PATH. -Version defaults to
# nvs_repair_gui.__version__; when given (CI passes the tag) it must match it.
param([string]$Version = "")
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
$own = (& python -c "import sys; sys.path.insert(0, 'windows'); import nvs_repair_gui; print(nvs_repair_gui.__version__)").Trim()
if (-not $Version) { $Version = $own }
if ($Version -ne $own) { throw "version $Version does not match nvs_repair_gui.__version__ ($own); bump it first" }

Write-Host "[build] nvs_repair $Version"
& python -m pip install --quiet --upgrade pyinstaller
if ($LASTEXITCODE) { throw "pip install pyinstaller failed" }

$build = Join-Path $root "build"
& python -m PyInstaller --noconfirm --clean --onefile --windowed --name nvs_repair `
    --paths "$root\windows" --distpath "$build\dist" --workpath "$build\work" --specpath $build `
    "$root\windows\launch.py"   # absolute: --specpath makes relative paths resolve against build\
if ($LASTEXITCODE) { throw "PyInstaller failed" }

$name = "nvs_repair-$Version-windows"
$pkg = Join-Path $build $name
if (Test-Path $pkg) { Remove-Item -Recurse -Force $pkg }
New-Item -ItemType Directory -Path $pkg | Out-Null
Copy-Item "$build\dist\nvs_repair.exe" $pkg

# source\ is exactly what is committed; .gitattributes keeps dev-only files out.
& git archive --format=zip -o "$build\source.zip" HEAD
if ($LASTEXITCODE) { throw "git archive failed" }
Expand-Archive -Path "$build\source.zip" -DestinationPath "$pkg\source"

@"
nvs_repair $Version for Windows

1. Needs Windows 10 22H2 or 11 with WSL 2, an NVIDIA GPU with a WSL-capable
   driver, 16 GB of GPU memory for repairs, and about 60 GB of disk.
2. Start nvs_repair.exe. It checks WSL and the GPU, tells you what (if
   anything) to do by hand, then installs the rest with "Install / update".
3. Full instructions: source\docs\windows-wsl.md.

Keep nvs_repair.exe next to the source folder.
"@ | Set-Content -Path "$pkg\README.txt" -Encoding UTF8

$zip = Join-Path $build "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$pkg\*" -DestinationPath $zip

# The exe has no console, so ask it to write its version to a file instead.
$check = Join-Path $build "version.txt"
if (Test-Path $check) { Remove-Item -Force $check }
Start-Process -Wait -FilePath "$pkg\nvs_repair.exe" -ArgumentList @("--version-file", "`"$check`"")
$reported = if (Test-Path $check) { (Get-Content $check -Raw).Trim() } else { "<nothing>" }
if ($reported -ne $Version) { throw "the built exe reported '$reported', expected '$Version'" }

Write-Host "[build] $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB), exe starts and reports $reported"
