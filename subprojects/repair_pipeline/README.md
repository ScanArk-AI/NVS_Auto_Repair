# repair_pipeline

给 [`repair_view_finder`](../repair_view_finder) 找到的待修复轨迹配参考视角，
再用 ArtiFixer 修复，最后抽出无损帧。

```
repair_view_finder   →   repair_pipeline
找待修复轨迹              选参考视角 → 渲 opacity → ArtiFixer 修复 → 抽帧
```

平时由仓库根目录的 `run.py` 调用；配置项、参考图怎么选、修复后端和场景描述见
[docs/configuration.zh-CN.md](../../docs/configuration.zh-CN.md)，权重见
[checkpoints/README.md](../../checkpoints/README.md)。这里只写单独运行和代码结构。

## 单独运行

通常不用单独安装：仓库根目录的 `./bootstrap.sh` 会建好两个阶段共用的环境、
clone ArtiFixer 并下载权重（见 [docs/linux.zh-CN.md](../../docs/linux.zh-CN.md)），`run.py all`
会调用本项目。只想单独跑修复这一步时：

```bash
PY=~/nvs_repair_runtime/envs/nvs/bin/python           # bootstrap.sh 建的环境
export ARTIFIXER_REPO=~/nvs_repair_runtime/ArtiFixer  # 或在配置里写 model.repo
cd subprojects/repair_pipeline

cp configs/scenes/example.yaml configs/scenes/my-scene.yaml   # 改 scene.* 和 input.repair_root
$PY run.py all --config configs/scenes/my-scene.yaml --dry-run   # 只打印解析后的配置
$PY run.py all --config configs/scenes/my-scene.yaml --limit 1
```

## 输入

repair_view_finder 的输出目录（`input.repair_root`）：

```
<repair_root>/
├── scene.json                    units_per_metre，决定 model.camera_scale
└── trajectories/traj_c0_00/
    ├── frames/00000..00024.png   渲染帧
    ├── opacity/                  缺了会自动补渲
    └── poses.json                每帧 OpenCV c2w + 内参
```

场景（`scene.*`）：3DGS ply、COLMAP 模型（`.bin` 或 `.txt`）、照片，以及可选的
动态物体 mask `<images>/mask_<cam>/<stem>.png`。mask 的极性用 `scene.mask_invalid`
指定（`black` / `white` / `none`），见 `repair_pipeline/masks.py`。

## 输出

```
<output.dir>/
├── traj_c0_00/
│   ├── references/selected.json  选中的照片、重叠率、mask 占比
│   ├── references/prepared/      处理成目标尺寸的参考图
│   ├── repaired/00000.png …      修复后的帧（无损）
│   ├── repaired.mp4
│   └── before_after.mp4
├── caption.h5                    本次用的文本条件
└── run.json                      每条轨迹的耗时和参考图重叠率
```

`repaired.mp4` 是 H.264，只用于查看；加回 3DGS 训练请用 `repaired/*.png`。

## 代码结构

| 模块 | 作用 |
|---|---|
| `cli.py`、`config.py` | 命令行；`configs/method.yaml` → 场景配置 → `--set` 逐层合并 |
| `pipeline.py` | 串起各阶段：选参考图 → 补渲 opacity → 修复 → 写帧和视频 |
| `references.py`、`frustum.py`、`coverage.py`、`masks.py` | 按共享高斯和视角覆盖挑参考照片，过滤被 mask 遮住太多的照片 |
| `prepare.py`、`pack.py` | 参考图去畸变、裁剪到目标视场；拼出模型要的相机字典 |
| `progressive.py` | 渐进参考模式（`workflow.reference_mode=progressive`） |
| `opacity.py`、`render_backend.py`、`ply_io.py` | 用 gsplat 渲 alpha（子进程，显存在模型加载前释放） |
| `backends.py`、`model.py`、`camera_rays.py` | 修复模型接口；ArtiFixer 推理（直接组装张量，不走它的评测脚手架） |
| `caption.py` | 用 Wan 的 UMT5 编码场景描述，写 `caption.h5` |
| `colmap_io.py`、`poses.py`、`frames.py` | COLMAP 读取、相机坐标约定、帧和视频读写 |

`scripts/download_weights.sh` 由 `bootstrap.sh` 调用；14B 只发布了 fp32，下载后用
`scripts/convert_ckpt_bf16.py` 转成 bf16。`tools/nerfstudio_to_colmap.py` 把 nerfstudio
的 `transforms.json` 转成 COLMAP 文本模型（用法见 docs/configuration.zh-CN.md）。

## 改推理代码时容易出错的地方

都是不报错、只会让结果变差的类型（`model.py` 开头有详细说明）：

- RGB 和参考图是 **[0, 1]** 的 float，VAE 预处理自己会映射到 [-1, 1]；
- `rendered_opacity`、`camera_rays` 必须是 **bf16**，`w2cs`、`Ks`、`neighbor_*` 必须是 **float32**；
- 帧数要补到 `T ≡ 1 (mod 4)`，且潜帧数能被 `frames_per_block` 整除；
- transforms 里的位姿是 OpenGL 约定，`poses.json` 是 OpenCV 约定（见 `poses.py`）。
