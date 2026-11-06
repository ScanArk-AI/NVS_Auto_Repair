# Linux：安装和运行

[English](linux.md)

## 环境要求

Linux、NVIDIA GPU、CUDA 12.8 工具链、Python 3.12、git。ArtiFixer 1.3B 在 **16 GB** 显卡上就能跑（修复阶段会自动打开低显存模式）；14B 模型需要约 89 GB。

## 安装

```bash
git clone https://github.com/ScanArk-AI/NVS_Auto_Repair.git && cd NVS_Auto_Repair
./bootstrap.sh --variant 1.3b         # 一个 venv、CUDA 扩展、ArtiFixer、1.3B 权重（约 33 GiB）
```

`bootstrap.sh` 会把所有东西装在 `~/nvs_repair_runtime` 下（用 `--root` 可以改），权重下载到 `checkpoints/`，并把各环境的解释器路径写进 `configs/envs.local.yaml`。重复运行会跳过已经装好的部分。`--variant 14b` 或 `all` 会额外下载 14B 模型（多约 135 GiB）；`--only finder|repair` 只安装其中一个阶段需要的部分；`--skip-weights` 跳过权重下载。

网络无法直连 GitHub 或 Hugging Face 时，运行前可以设置 `GH_PROXY`（GitHub 镜像前缀）、`HF_ENDPOINT`（Hugging Face 镜像）和 `HF_HUB_DISABLE_XET=1`，都是可选的。

两个阶段共用一个虚拟环境（torch 2.7.1，cu128）。编排层会直接调用这个环境的解释器，不需要手动激活。

## 快速开始

```bash
PY=~/nvs_repair_runtime/envs/nvs/bin/python

# 1. 根据场景目录生成配置（3DGS ply + sparse/0 + images/，其它目录结构见 docs/configuration.zh-CN.md）
$PY run.py make-config --scene-root /data/my_scene    # 米制采集加 --units-per-metre 1；围着主体拍的加
                                                       # --object-centric 或 --object-centric-indoor（见 docs/configuration.zh-CN.md）
CFG=configs/scenes/generated/my_scene/workflow.yaml

# 2. 检查环境、权重和路径
$PY run.py doctor --config $CFG

# 3. 连通性检查：只跑两条路径，渲染帧原样返回（不需要权重，也不生成 caption）
$PY run.py all --config $CFG --limit 2 --set 'repair.set=[model.backend=passthrough]' --set out=outputs/my_scene_smoke

# 4. 正式运行（场景描述自己写，或者配置 VLM，见 docs/configuration.zh-CN.md）
$PY run.py all --config $CFG --set "workflow.caption_text=A small office with white walls, a grey carpet and a glass partition."
```

`run.py plan --config …` 会打印解析后的配置和每个阶段将要执行的完整命令，不占用 GPU。每个阶段也可以单独运行：`run.py find | select | repair`。

## 输出

每次运行一个根目录（配置里的 `out`，默认 `outputs/<配置文件名>`）：

```
<out>/
├── 1_find/
│   ├── repair/trajectories/traj_c0_00/   frames/（渲染帧）、opacity/、poses.json、transforms.json、meta.json
│   └── review/                           所选路径和最差视角的总览图
├── 2_repair/
│   └── traj_c0_00/
│       ├── references/                   selected.json（选了哪些照片、为什么）+ prepared/ 处理后的图片
│       ├── repaired/00000.png …          修复后的帧（无损）
│       ├── repaired.mp4
│       └── before_after.mp4              渲染和修复结果左右对比
├── caption.h5 / caption.txt              模型所用的场景描述
└── workflow.json                         各阶段状态、轨迹清单、模型版本、参考视角的选法
```

重新运行会从中断的地方继续：已有的轨迹、参考视角、修复结果都会保留。要重做某个阶段，加 `--set workflow.force_find|force_select|force_repair=true`。

## 环境变量

| 变量 | 使用者 | 说明 |
|---|---|---|
| `RUNTIME`、`PYTHON` | `bootstrap.sh` | 安装根目录（同 `--root`）、用来创建虚拟环境的解释器 |
| `GH_PROXY`、`HF_ENDPOINT`、`HF_HUB_DISABLE_XET` | `bootstrap.sh` | 镜像，都是可选的 |
| `NVS_REPAIR_VLM_*` | 生成场景描述 | 见[场景描述](configuration.zh-CN.md#场景描述) |
| `WORKFLOW_FINDER_PY`、`WORKFLOW_REPAIR_PY` | 编排层 | 不用 `envs.local.yaml` 时，分别指定两个阶段的解释器 |
| `NVS_REPAIR_WEIGHTS`、`ARTIFIXER_REPO` | 单独运行 `repair_pipeline` 时 | 权重目录、ArtiFixer 代码目录（通过编排层运行时会自动传入） |
| `NVS_REPAIR_FONT` | 对比视频 | 标签用的 TrueType 字体（默认用 Pillow 自带的） |

## 权重

下载到 `checkpoints/ArtiFixer/`，不提交进仓库；配置里用相对路径引用。各文件和大小见 [checkpoints/README.md](../checkpoints/README.md)。已经下载过的话可以直接软链接过来：`ln -s /path/to/ArtiFixer checkpoints/ArtiFixer`。

## 常见问题

**`doctor` 报找轨迹阶段缺 `gavis`、`simple_knn`、`diff_gaussian_rasterization` 或 `gavis_rasterization`**
CUDA 扩展没有编译成功。确认 `nvcc` 在 `PATH` 上，且版本和 torch 一致（CUDA 12.8），然后执行 `./bootstrap.sh --only finder`。

**运行停在等待显存**
修复阶段会在 `model.device` 上最多等 `workflow.vram_wait_min` 分钟，直到空出足够的显存（1.3B 需要 12.5 GB，14B 需要 92 GB）。关掉占用这块显卡的其它程序，或者把 `model.device` 换成另一块显卡。

## 开发

```bash
pip install numpy scipy omegaconf pyyaml pillow pytest
python -m pytest            # 只用 CPU；CI 在 Python 3.10 和 3.12 上跑同样的测试
```

GPU 部分用上面的 passthrough 运行，以及 1.3B 跑两三条路径来检查。
