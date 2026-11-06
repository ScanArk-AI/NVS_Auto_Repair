# repair_view_finder

Stage 1 of nvs_repair: given a 3DGS scene (PLY + the COLMAP model it was
trained from), find the camera paths through it that most need repair, and
render them as short clips for the repair stage.

`run.py all` at the repository root runs this stage for you. Installation is
in [docs/linux.md](../../docs/linux.md); scene scale and the three path presets
in [docs/configuration.md](../../docs/configuration.md#choosing-the-path-settings).
This page covers the method, running the stage on its own, and its outputs.

## Method

1. **Uncertainty.** [GAVIS](https://github.com/xsj01/gavis) estimates how
   uncertain each rendered pixel is from how few visible Gaussians explain it.
   One visibility field is built per camera, with that camera's intrinsics and
   40 of its views (`field.*`).
2. **Candidate poses.** Each SfM frame of a camera (up to 160 sampled,
   `trajectories.query_target`) is moved along gravity to
   `trajectories.height_m` above the floor, levelled, and tilted to each of
   `trajectories.pitches`. Every height in a `height_m` list gets its own set.
3. **Frame pairs.** A pair `(i, j)` becomes a candidate path when the two
   frames are close in time (`max_gap`), far enough apart (`dist_m`), overlap in
   view (`iou`), at least one frame between them is uncertain
   (`uq_percentile`), the straight line between them keeps `clearance_m` from
   the scene, and -- checked lazily on the paths that get picked -- something
   other than floor lies `forward_band_m` ahead along the view axis
   (`forward_*`). The path is `frames` poses: min-jerk position, SLERP rotation.
4. **Greedy pick.** Per camera, up to `per_camera` paths are picked by
   `scene_weight` x new scene coverage + `path_weight` x new path coverage +
   `uq_weight` x normalised uncertainty, keeping `min_sep` frames between paths
   at the same height (relaxed if the pool runs out).
5. **Render.** Paths are rendered from the full PLY at 960x544 (544x960 for a
   portrait camera, `orientation`), `fov_y_deg` 75, `fps` 15.

Alongside the paths, the SfM views themselves are ranked by how much
high-uncertainty volume near the floor they see (`scoring.*`,
`geometry.height_band_m`) for the review pictures.

Every `*_m` value is in metres and converted with `geometry.units_per_metre`;
all defaults, with comments, are in [configs/method.yaml](configs/method.yaml).

## Running it on its own

```bash
cp configs/scene_template.yaml configs/scenes/my_scene.yaml   # fill in scene.root
python run.py --config configs/scenes/my_scene.yaml
python run.py --config configs/scenes/my_scene.yaml --limit-views 4          # quick smoke test
python run.py --config configs/scenes/my_scene.yaml --out /tmp/rvf \
    --set views.cameras=2 trajectories.per_camera=4
```

A scene config is merged over `configs/method.yaml`; `--set key=value`
overrides either. The examples in `configs/scenes/` are the three presets.

## Outputs

```
<out>/
  run.json, candidates.json, scores.csv    config used, ranked SfM views
  review/
    index.html                             the path clips, by camera
    top40_views.png, map_topdown.png       ranked views, top-down map
  repair/                                  what the repair stage reads
    manifest.json                          paths, views, defects
    scene.json                             units_per_metre, up, floor
    views.json, poses.json, defects.json
    trajectories/traj_c<cam>_<k>/
      frames/00000.png ...                 the rendered path
      trajectory.mp4
      poses.json                           OpenCV c2w / w2c per frame
      transforms.json                      nerfstudio format
      meta.json                            pair, clearance, forward distance (metres)
```

Poses are OpenCV c2w (x right, y down, z forward) in scene units.
