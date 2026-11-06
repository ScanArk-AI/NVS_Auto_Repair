# Windows: offline bundle (no WSL)

[中文](windows-offline.zh-CN.md)

## Install and run

**Needs**: Windows 10 22H2 or 11 (x64), an NVIDIA GPU with driver 572.61 or
newer, [7-Zip](https://www.7-zip.org) to unpack, and about 45 GB of free disk
(the bundle, the weights and outputs). No Python, CUDA toolkit or compiler.

1. From the [Releases page](https://github.com/ScanArk-AI/NVS_Auto_Repair/releases), download all `nvs_repair-<version>-win64.7z.*`
   volumes and the `.sha256` file into one folder. Open the `.001` volume with
   7-Zip and extract it to a short path, e.g. `C:\nvs_repair\`.
2. Start `nvs_repair-<version>-win64\nvs_repair.exe`. The top row checks the
   driver, the GPU, the VC++ runtime and the bundled Python; if it reports the
   VC++ runtime missing, run `vc_redist.x64.exe` from the same folder once.
3. With **Mode: Check only**, press **Install / update**: it downloads GAVIS
   (a few MB from GitHub). Select **Repair (1.3B)** and press it again to
   download the weights (~33 GB) into `app\checkpoints\ArtiFixer\`.
4. Then as steps 4-5 of the [WSL version](windows-wsl.md): pick the scene folder, the output
   folder and the kind of capture, **Run** with **Check only**, then with
   **Repair (1.3B)**, and **Open results**.

The scene is read where it is and outputs go straight to the folder you pick.
**Stop** ends the whole run, including the GPU work. A scene description used
before is reused from `app\.cache\captions\`.
Settings are kept in `%APPDATA%\nvs_repair\`; to uninstall, delete the bundle
folder and that one. If the window does not appear, run `debug.cmd` in the
bundle folder to see the error. Mirrors for GitHub / Hugging Face are under
*Advanced*.

The bundle ships compiled code under several licences, some **non-commercial**
(see [NOTICE.md](../NOTICE.md) and the bundle's `licenses\` folder).
To build it yourself, see [windows/native/README.md](../windows/native/README.md).
