# Windows：离线包（免 WSL）

[English](windows-offline.md)

## 安装和运行

**要求**：Windows 10 22H2 或 Windows 11（x64）；NVIDIA 显卡，驱动 572.61 或更新；用 [7-Zip](https://www.7-zip.org) 解压；约 45 GB 可用磁盘（离线包、权重和输出）。不需要 Python、CUDA 工具链或编译器。

1. 从 [Releases 页面](https://github.com/ScanArk-AI/NVS_Auto_Repair/releases)把所有 `nvs_repair-<版本>-win64.7z.*` 分卷和 `.sha256` 文件下载到同一个文件夹。用 7-Zip 打开 `.001` 分卷，解压到一个短路径，比如 `C:\nvs_repair\`。
2. 运行 `nvs_repair-<版本>-win64\nvs_repair.exe`。最上面一行会检查驱动、显卡、VC++ 运行库和自带的 Python；如果提示缺少 VC++ 运行库，运行同一文件夹里的 `vc_redist.x64.exe` 装一次。
3. **Mode** 选 **Check only**，点 **Install / update**：下载 GAVIS（从 GitHub 下载，几 MB）。改选 **Repair (1.3B)** 再点一次，把权重（约 33 GB）下载到 `app\checkpoints\ArtiFixer\`。
4. 之后和 [WSL 版](windows-wsl.zh-CN.md)的第 4、5 步一样：选场景目录、输出目录和采集类型，先用 **Check only** 点 **Run**，再用 **Repair (1.3B)** 点 **Run**，最后点 **Open results**。

场景直接在原位置读取，输出直接写到你选的文件夹。**Stop** 会结束整个运行，包括显卡上的计算。用过的场景描述会从 `app\.cache\captions\` 直接复用。设置保存在 `%APPDATA%\nvs_repair\`；卸载时删掉离线包文件夹和这个文件夹即可。窗口没有出来时，运行离线包文件夹里的 `debug.cmd` 查看报错。GitHub 和 Hugging Face 的镜像在 *Advanced* 里设置。

离线包里带有多种许可证的编译代码，其中一部分**仅限非商用**（见 [NOTICE.md](../NOTICE.md) 和离线包里的 `licenses\` 文件夹）。想自己组包，见 [windows/native/README.md](../windows/native/README.md)。
