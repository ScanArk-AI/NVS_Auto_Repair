# Configuration

[中文](configuration.zh-CN.md)

## Inputs

- **Input**: a trained 3DGS model (`.ply`, the INRIA/gsplat layout), the COLMAP
  model it was trained from (`.bin` or `.txt`), and the photos. Dynamic-object
  masks are optional.
- **Scene scale.** Every `*_m` setting is in metres; the finder converts with
  `geometry.units_per_metre` (scene units per metre). Use `1.0` for a metric
  capture (e.g. LiDAR-registered), the real number if you know it, or `auto`
  (what `make-config` writes) to estimate it from a handheld or body-worn
  capture: the median camera height above the floor divided by
  `geometry.capture_height_m` (1.2 m). For drone, ground-robot or multi-storey
  captures, give the number. The value used is written to
  `1_find/repair/scene.json`.
- **Up and floor.** Up is estimated from the cameras' orientation, the floor as
  the densest layer of Gaussians below the camera path; override with
  `geometry.up` and `geometry.floor_m` (scene units). New paths run 0.4 m above
  the floor (`trajectories.height_m`).
- **Kind of capture.** The default path settings are for a robot moving low
  through rooms; captures walking around a subject need the object-centric
  settings -- see "Choosing the path settings" below. Paths follow the camera's
  orientation: portrait cameras give 544x960 clips, landscape ones 960x544.

## Scene configs and options

A scene config only wires the two stages together; each stage keeps its own
config so it can also be run standalone. `make-config` writes all three files
and recognises a 3DGS `.ply` at the scene root (`point_cloud.ply`, `splat.ply`)
or the latest `point_cloud/iteration_<N>/point_cloud.ply`, a COLMAP model in
`sparse/0` or `sparse`, and `images/`. For anything else, copy
`configs/scenes/example*.yaml` and fill in the paths.

**nerfstudio data** (`transforms.json`): convert the cameras to a COLMAP text
model once, then point `scene.sparse` at the output and `scene.images` at the
directory holding `transforms.json`:

```bash
python subprojects/repair_pipeline/tools/nerfstudio_to_colmap.py data/transforms.json data/sparse_ns \
    --dataparser-transforms outputs/<run>/splatfacto/<date>/dataparser_transforms.json
```

The cameras must be in the coordinate frame of the 3DGS you are repairing. For a
splatfacto model, pass that run's `dataparser_transforms.json`; without it the
script undoes `applied_transform` and returns the capture frame. Note that the
dataparser's scale means scene units are no longer metres: keep
`geometry.units_per_metre: auto` or give the real value. PINHOLE, OPENCV and
OPENCV_FISHEYE cameras are supported, with global or per-frame intrinsics.

Defaults live in `configs/workflow.yaml`; anything can be overridden per run
with `--set key=value`, and `repair.set` / `find.set` pass overrides straight
through to a stage.

| Option | Default | |
|---|---|---|
| `model.variant` | `1.3b` | `1.3b` or `14b` |
| `workflow.reference_mode` | `photos` | how reference views are chosen, see below |
| `workflow.caption_text` | `null` | your own scene description |
| `workflow.auto_caption` | `true` | caption with a VLM when there is no `caption_text`; `false` = no description |
| `workflow.vram_wait_min` | `30` | wait this long for the repair GPU to have room |
| `repair.set: model.low_vram=…` | automatic | `true` runs 1.3B on a 16 GB GPU (same output); set it to force either way |
| `find.set: geometry.units_per_metre=…` | `auto` from make-config | scene units per metre, see "Inputs" |
| `find.set: geometry.up=… / geometry.floor_m=…` | estimated | up vector, floor height in scene units |
| `find.set: trajectories.height_m=…` | `0.4` | height of new paths above the floor, in metres; a list such as `[0.3, 0.8]` tries each |
| `repair.set: model.backend=…` | `artifixer` | `artifixer`, `passthrough`, or `package.module:Class` |
| `repair.set: refer.num_views=…` | `12` | reference views per path |
| `repair.set: refer.strategy=…` | `coverage` | `coverage` (spread by viewing direction) or `overlap` (ArtiFixer's top shared Gaussians) |
| `repair.set: refer.max_mask_frac=…` | `0.10` | skip photos whose dynamic-object mask covers more than this |

## Choosing the path settings

The finder has three sets of path settings, one per kind of capture; pick one
with `make-config` (or copy the finder's example file). All `*_m` values are
metres, converted with `geometry.units_per_metre`.

| `trajectories.*` | Default: indoor, robot height | Object-centric, outdoor (`--object-centric`) | Object-centric, indoor (`--object-centric-indoor`) |
|---|---|---|---|
| file | `method.yaml` | `example_object_centric.yaml` | `example_object_centric_indoor.yaml` |
| `height_m` | `0.4` | `0.8` | `[0.3, 0.55, 0.8, 1.0, 1.2]` |
| `pitches` (deg, + looks up) | `[0, 10, 20]` | `[-20, -35]` | `[-30, -15, 0, 10]` |
| `forward_band_m` | `[0.2, 1.5]` | `[0.3, 3.0]` | `[0.25, 3.0]` |
| `forward_height_band_m` | `[0.15, 3.0]` (floor does not count) | `[0.0, 3.0]` (ground counts) | `[0.0, 3.0]` |
| `forward_min_opacity` | `0.0` | `0.5` | `0.5` |
| `per_camera` | `12` | `12` | `30` (plus looser pair constraints) |

- **Default** -- views from a robot moving 20-40 cm above the floor indoors:
  paths at 0.4 m looking level or up, with something other than floor 0.2-1.5 m
  straight ahead.
- **Object-centric, outdoor** -- walking around a single subject on open ground:
  paths higher and looking down, the ground counts as something ahead, a longer
  forward range, and translucent floaters (common in handheld 3DGS) ignored.
  For more paths in a wide scene, try `height_m: [0.3, 0.8]`,
  `pitches: [-10, -25, -40]`, `forward_band_m: [0.2, 4.0]` and `per_camera: 30`.
- **Object-centric, indoor** -- walking around a table or in a small room where
  everything is close: several heights from 0.3 to 1.2 m and both downward and
  level views, all the way round.

When a camera yields few paths, the finder prints how many candidates the
forward-distance check rejected; pick the preset that matches the capture, or
widen `forward_band_m`. Every finder setting is described in the comments of
[`subprojects/repair_view_finder/configs/method.yaml`](../subprojects/repair_view_finder/configs/method.yaml).

## Reference views

Every path is repaired with 12 real photos. By default (`refer.strategy:
coverage`) they are picked one at a time, each covering the most of what the
path sees that the earlier picks do not, where a Gaussian only counts as covered
if a photo sees it from within 30° of the path's own view (`refer.angle_deg`);
the candidates are the 64 photos sharing the most Gaussians with the path
(`refer.candidates`). On video captures this spreads the references over the
whole walk instead of taking a run of near-identical consecutive frames.
`refer.strategy=overlap` keeps ArtiFixer's own rule, the 12
photos sharing the most Gaussians.

`workflow.reference_mode=progressive` repairs paths one after another, best
photo coverage first, and lets a later path swap up to 4 of its photos for
frames already repaired on an earlier path -- where those frames see, from
within 30°, parts of the scene the photos do not. `python -m nvs_repair.experimental` runs both modes side by side into
separate folders and builds comparison videos.

Each reference is described to the model by the intrinsics of the image it is
actually given (fisheye photos are undistorted, others centre-cropped);
`refer.neighbor_intrinsics=photo` restores the earlier behaviour of passing the
source photo's intrinsics.

## Repair backends

`model.backend` selects what turns renders into repaired frames:

- `artifixer` (default): [ArtiFixer](https://github.com/nv-tlabs/ArtiFixer) 1.3B or 14B.
- `passthrough`: returns the renders unchanged -- no weights, no caption; for
  checking an install or a new scene end to end.
- `package.module:ClassName`: your own model. It receives everything the
  pipeline prepared (renders, opacity, references at the target size, all
  cameras, the encoded caption); see `subprojects/repair_pipeline/repair_pipeline/backends.py`.

## Captions

ArtiFixer is conditioned on a one-paragraph description of the scene (what the
room is, not how the camera moves). Each encoded description is kept in
`.cache/captions/` and reused when the same text comes up again, in any
output directory. The description is taken from, in order:

1. `workflow.caption_text` -- written by you; no network.
2. A vision-language model behind an HTTP API, shown 4 of the real photos spread
   over the capture (`workflow.caption_source: photos`, the default) or the middle
   render of up to 4 paths (`renders`, also the fallback when no photo is readable):

   | Variable | |
   |---|---|
   | `NVS_REPAIR_VLM_URL` | endpoint or API base, e.g. `https://api.openai.com/v1` |
   | `NVS_REPAIR_VLM_MODEL` | model name |
   | `NVS_REPAIR_VLM_KEY` | bearer token (omit for local servers) |
   | `NVS_REPAIR_VLM_API` | `chat` (OpenAI-compatible Chat Completions: OpenAI, vLLM, Ollama, …; default) or `responses` |

3. `--set workflow.auto_caption=false` -- no description at all.

If captioning fails, the run stops and says why.
