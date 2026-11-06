# 配置

[English](configuration.md)

## 输入

- **输入**：训练好的 3DGS 模型（`.ply`，INRIA / gsplat 的格式）、训练它所用的 COLMAP 模型（`.bin` 或 `.txt`），以及照片。动态物体的 mask 可选。
- **场景尺度。** 所有 `*_m` 参数都以米为单位，找轨迹时用 `geometry.units_per_metre`（每米对应多少场景单位）换算：
  - 米制采集（比如和激光雷达配准过的）填 `1.0`；
  - 知道真实尺度就填具体数字；
  - 填 `auto`（`make-config` 默认写的就是它）则按手持或身上背着的采集设备自动估计：相机离地高度的中位数 ÷ `geometry.capture_height_m`（1.2 m）；
  - 无人机、地面机器人或跨楼层的采集，请直接填数字。

  实际用的值会写进 `1_find/repair/scene.json`。
- **上方向和地面。** 上方向根据相机的朝向估计；地面取相机路径下方高斯最密的那一层。两者都可以手动指定：`geometry.up` 和 `geometry.floor_m`（场景单位）。新路径默认在离地 0.4 m 的高度（`trajectories.height_m`）。
- **采集方式。** 默认的路径设置针对室内贴地移动的机器人视角；围着主体拍的采集要用物体中心的设置，见下文"选哪套视角参数"。路径画面会跟随相机方向：竖屏相机输出 544×960，横屏相机输出 960×544。

## 场景配置和选项

场景配置只负责把两个阶段接起来，每个阶段保留自己的配置，所以也能单独运行。`make-config` 会一次写好三个文件，能识别的目录结构有：

- 3DGS 模型：场景根目录下的 `point_cloud.ply`、`splat.ply`，或者最新一次迭代的 `point_cloud/iteration_<N>/point_cloud.ply`；
- COLMAP 模型：`sparse/0` 或 `sparse`；
- 照片：`images/`。

其它结构可以复制 `configs/scenes/example*.yaml`，手动填写路径。

**nerfstudio 数据**（`transforms.json`）：先把相机转换成 COLMAP 文本模型，再把 `scene.sparse` 指向输出目录、`scene.images` 指向 `transforms.json` 所在的目录：

```bash
python subprojects/repair_pipeline/tools/nerfstudio_to_colmap.py data/transforms.json data/sparse_ns \
    --dataparser-transforms outputs/<run>/splatfacto/<date>/dataparser_transforms.json
```

相机必须和要修复的 3DGS 在同一个坐标系里：

- splatfacto 训练出的模型，要传入那次训练的 `dataparser_transforms.json`；
- 不传的话，脚本会撤销 `applied_transform`，回到采集时的坐标系；
- dataparser 会缩放场景，所以转换后的场景单位不再是米：保留 `geometry.units_per_metre: auto`，或者填真实的尺度；
- 支持 PINHOLE、OPENCV、OPENCV_FISHEYE 三种相机，内参可以是全局的，也可以逐帧不同。

默认值在 `configs/workflow.yaml` 里，任何一项都可以用 `--set key=value` 临时覆盖。`repair.set` 和 `find.set` 里的内容会原样传给对应阶段。

| 选项 | 默认值 | 说明 |
|---|---|---|
| `model.variant` | `1.3b` | `1.3b` 或 `14b` |
| `workflow.reference_mode` | `photos` | 参考视角的选法，见下文 |
| `workflow.caption_text` | `null` | 自己写的场景描述 |
| `workflow.auto_caption` | `true` | 没有 `caption_text` 时用 VLM 生成描述；`false` 表示不用描述 |
| `workflow.vram_wait_min` | `30` | 修复所用的 GPU 显存不够时，最多等待的分钟数 |
| `repair.set: model.low_vram=…` | 自动 | `true` 让 1.3B 在 16 GB 显卡上运行（输出不变）；显式设置则以你的为准 |
| `find.set: geometry.units_per_metre=…` | `make-config` 写 `auto` | 每米多少场景单位，见上文"输入" |
| `find.set: geometry.up=… / geometry.floor_m=…` | 自动估计 | 上方向、地面高度（场景单位） |
| `find.set: trajectories.height_m=…` | `0.4` | 新路径离地高度，单位为米；写成列表（如 `[0.3, 0.8]`）则每个高度都试 |
| `repair.set: model.backend=…` | `artifixer` | `artifixer`、`passthrough` 或 `package.module:Class` |
| `repair.set: refer.num_views=…` | `12` | 每条路径的参考视角数 |
| `repair.set: refer.strategy=…` | `coverage` | `coverage`（按观察方向分散挑选）或 `overlap`（ArtiFixer 的共享高斯前 12） |
| `repair.set: refer.max_mask_frac=…` | `0.10` | 动态物体 mask 占比超过这个值的照片不参与选择 |

## 选哪套视角参数

找轨迹这一步有三套路径参数，分别对应三类采集，用 `make-config` 的开关选择（也可以直接复制 finder 的示例文件）。所有 `*_m` 参数的单位都是米，用 `geometry.units_per_metre` 换算成场景单位。

| `trajectories.*` | 默认：室内机器人视角 | 物体中心·户外（`--object-centric`） | 物体中心·室内（`--object-centric-indoor`） |
|---|---|---|---|
| 文件 | `method.yaml` | `example_object_centric.yaml` | `example_object_centric_indoor.yaml` |
| `height_m` | `0.4` | `0.8` | `[0.3, 0.55, 0.8, 1.0, 1.2]` |
| `pitches`（度，正数向上看） | `[0, 10, 20]` | `[-20, -35]` | `[-30, -15, 0, 10]` |
| `forward_band_m` | `[0.2, 1.5]` | `[0.3, 3.0]` | `[0.25, 3.0]` |
| `forward_height_band_m` | `[0.15, 3.0]`（地面不算） | `[0.0, 3.0]`（地面也算） | `[0.0, 3.0]` |
| `forward_min_opacity` | `0.0` | `0.5` | `0.5` |
| `per_camera` | `12` | `12` | `30`（帧对约束也放宽了） |

- **默认**：模拟离地 20–40 cm、在室内低处移动的机器人视角。路径离地 0.4 m，平视或仰视，要求正前方 0.2–1.5 m 内有不属于地面的东西。
- **物体中心·户外**：在开阔地上围着单个主体拍。路径更高、低头看，地面也算"前方有东西"，前方距离范围更大，并忽略手持 3DGS 常见的半透明浮点。场景开阔、想要更多路径时，可以试试 `height_m: [0.3, 0.8]`、`pitches: [-10, -25, -40]`、`forward_band_m: [0.2, 4.0]` 和 `per_camera: 30`。
- **物体中心·室内**：围着一张桌子或在小房间里拍，四周的东西都很近。路径在 0.3–1.2 m 之间取多个高度，既有俯视也有平视，环绕一整圈。

某台相机选出的路径太少时，finder 会打印有多少候选被"前方距离"检查卡掉；选用和采集方式匹配的预设，或者放宽 `forward_band_m` 即可。finder 的全部参数见 [`subprojects/repair_view_finder/configs/method.yaml`](../subprojects/repair_view_finder/configs/method.yaml) 里的注释。

## 参考视角

每条路径用 12 张真实照片当参考。默认（`refer.strategy: coverage`）是逐张挑选：每一张都挑"能补上前面已选照片没覆盖到的部分最多"的那张。一个高斯只有被照片从和路径视线夹角 30° 以内的方向看到才算覆盖（`refer.angle_deg`）；候选范围是和这条路径共享高斯最多的 64 张照片（`refer.candidates`）。

在视频采集上，这样选出的参考会分布在整段采集里，而不是一串几乎相同的连续帧。设 `refer.strategy=overlap` 则沿用 ArtiFixer 自己的规则，即取共享高斯最多的 12 张。

`workflow.reference_mode=progressive` 会按"照片覆盖度从高到低"的顺序逐条修复路径。后面的路径可以把最多 4 张照片换成前面路径已经修好的帧，前提是这些帧能从 30° 以内的方向看到照片没拍到的部分。`python -m nvs_repair.experimental` 可以把两种选法分别跑到不同目录，并生成对比视频。

送进模型的每张参考图，用的都是它实际处理后的内参：鱼眼照片会去畸变，其它照片会居中裁剪。设 `refer.neighbor_intrinsics=photo` 可以恢复旧行为，也就是直接传原照片的内参。

## 修复后端

`model.backend` 决定用什么把渲染帧修成修复帧：

- `artifixer`（默认）：[ArtiFixer](https://github.com/nv-tlabs/ArtiFixer) 1.3B 或 14B。
- `passthrough`：原样返回渲染帧。不需要权重，也不生成 caption，用来检查安装或新场景能不能端到端跑通。
- `package.module:ClassName`：你自己的模型。它会拿到流程准备好的全部输入：渲染帧、透明度、缩放到目标尺寸的参考图、所有相机，以及编码后的 caption。接口见 `subprojects/repair_pipeline/repair_pipeline/backends.py`。

## 场景描述

ArtiFixer 需要一段描述场景的文字，写清楚房间是什么样，不用写相机怎么动。编码结果会存进 `.cache/captions/`，之后在任何输出目录里遇到同样的文字都直接复用。描述按以下顺序获取：

1. `workflow.caption_text`：你自己写的描述，不联网。
2. 通过 HTTP API 调用视觉语言模型（VLM）。默认（`workflow.caption_source: photos`）给它看在整个采集中均匀挑出的 4 张真实照片；设成 `renders` 则看最多 4 条路径各自的中间一帧渲染，读不到照片时也会自动改用这种方式：

   | 环境变量 | 说明 |
   |---|---|
   | `NVS_REPAIR_VLM_URL` | 接口地址或 API 根地址，比如 `https://api.openai.com/v1` |
   | `NVS_REPAIR_VLM_MODEL` | 模型名 |
   | `NVS_REPAIR_VLM_KEY` | 访问令牌（本地服务可以不填） |
   | `NVS_REPAIR_VLM_API` | `chat`（兼容 OpenAI 的 Chat Completions：OpenAI、vLLM、Ollama 等，默认）或 `responses` |

3. `--set workflow.auto_caption=false`：不使用描述。

场景描述生成失败时，运行会停下并说明原因。
