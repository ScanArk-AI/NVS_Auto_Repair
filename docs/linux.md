# Linux: install and run

[中文](linux.zh-CN.md)

## Requirements

Linux with an NVIDIA GPU, the CUDA 12.8 toolkit, Python 3.12 and git.
ArtiFixer 1.3B runs on a **16 GB** GPU (the repair stage switches its
low-memory mode on by itself); the 14B model needs about 89 GB.

## Install

```bash
git clone https://github.com/ScanArk-AI/NVS_Auto_Repair.git && cd NVS_Auto_Repair
./bootstrap.sh --variant 1.3b         # one venv, CUDA extensions, ArtiFixer, 1.3B weights (~33 GiB)
```

`bootstrap.sh` builds everything under `~/nvs_repair_runtime` (`--root` to
change), downloads weights into `checkpoints/`, and writes the interpreter paths
to `configs/envs.local.yaml`. It is idempotent: re-running skips what is already
there. `--variant 14b` or `all` adds the 14B model (~135 GiB more);
`--only finder|repair` installs just one stage's part; `--skip-weights` skips the download.

Behind a firewall, set any of `GH_PROXY` (a GitHub mirror prefix),
`HF_ENDPOINT` (a Hugging Face mirror) and `HF_HUB_DISABLE_XET=1` before running it.

Both stages share one virtual environment (torch 2.7.1, cu128). The
orchestrator calls its interpreter itself, so there is nothing to activate.

## Quick start

```bash
PY=~/nvs_repair_runtime/envs/nvs/bin/python

# 1. Write the scene configs from a scene directory
#    (3DGS ply + sparse/0 + images/, see docs/configuration.md for other layouts)
$PY run.py make-config --scene-root /data/my_scene    # --units-per-metre 1 if metric; for a walk around a
                                                       # subject --object-centric / --object-centric-indoor
                                                       # (see docs/configuration.md)
CFG=configs/scenes/generated/my_scene/workflow.yaml

# 2. Check environments, weights and paths
$PY run.py doctor --config $CFG

# 3. Wiring check: two paths, renders passed through unchanged (no weights, no caption)
$PY run.py all --config $CFG --limit 2 --set 'repair.set=[model.backend=passthrough]' --set out=outputs/my_scene_smoke

# 4. The real run (describe the scene yourself, or configure a VLM -- see docs/configuration.md)
$PY run.py all --config $CFG --set "workflow.caption_text=A small office with white walls, a grey carpet and a glass partition."
```

`run.py plan --config …` prints the resolved configuration and the exact
command for each stage without touching the GPU. Each stage can also be run on
its own: `run.py find | select | repair`.

## Output

One root per run (`out` in the config, default `outputs/<config name>`):

```
<out>/
├── 1_find/
│   ├── repair/trajectories/traj_c0_00/   frames/ (renders), opacity/, poses.json, transforms.json, meta.json
│   └── review/                           overview of the chosen paths and the worst views
├── 2_repair/
│   └── traj_c0_00/
│       ├── references/                   selected.json (which photos and why) + prepared/ images
│       ├── repaired/00000.png …          repaired frames (lossless)
│       ├── repaired.mp4
│       └── before_after.mp4              render vs repaired, side by side
├── caption.h5 / caption.txt              the scene description the model was conditioned on
└── workflow.json                         stage status, trajectories, variant, reference mode
```

Re-running continues where it stopped: existing trajectories, reference sets and
repaired clips are kept (`--set workflow.force_find|force_select|force_repair=true`
to redo a stage).

## Environment variables

| Variable | Used by | |
|---|---|---|
| `RUNTIME`, `PYTHON` | `bootstrap.sh` | install root (same as `--root`), interpreter used to create the venv |
| `GH_PROXY`, `HF_ENDPOINT`, `HF_HUB_DISABLE_XET` | `bootstrap.sh` | mirrors, all optional |
| `NVS_REPAIR_VLM_*` | captioning | see [Captions](configuration.md#captions) |
| `WORKFLOW_FINDER_PY`, `WORKFLOW_REPAIR_PY` | orchestrator | interpreter per stage, if not using `envs.local.yaml` |
| `NVS_REPAIR_WEIGHTS`, `ARTIFIXER_REPO` | `repair_pipeline` run on its own | weights directory, ArtiFixer checkout (the orchestrator passes both itself) |
| `NVS_REPAIR_FONT` | comparison videos | a TrueType font for the labels (default: Pillow's) |

## Weights

Downloaded into `checkpoints/ArtiFixer/` (never committed); configs refer to
them by relative path. See [checkpoints/README.md](../checkpoints/README.md) for
the files and sizes. To reuse an existing download, symlink it:
`ln -s /path/to/ArtiFixer checkpoints/ArtiFixer`.

## Troubleshooting

**`doctor` says the find stage is missing `gavis`, `simple_knn`, `diff_gaussian_rasterization` or `gavis_rasterization`**
The CUDA extensions did not build. Make sure `nvcc` is on `PATH` and its version
matches the torch wheel (CUDA 12.8), then `./bootstrap.sh --only finder`.

**The run waits for GPU memory**
The repair stage waits up to `workflow.vram_wait_min` minutes for enough free
memory on `model.device` (12.5 GB for 1.3B, 92 GB for 14B). Free other
programs on that GPU, or point `model.device` at another one.

## Development

```bash
pip install numpy scipy omegaconf pyyaml pillow pytest
python -m pytest            # CPU only; CI runs the same on Python 3.10 and 3.12
```

The GPU stages are checked with the passthrough run above and a 1.3B run on two
or three paths.
