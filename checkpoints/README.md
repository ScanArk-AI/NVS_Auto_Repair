# checkpoints

Weights used by the repair stage. Kept at the repository root so configuration
never carries a machine-specific path: configs refer to `checkpoints/ArtiFixer/...`
and this directory is resolved relative to the repo.

Contents are **not** tracked by git (see `.gitignore`); they can be symlinks to
an existing download:

```bash
ln -s /path/to/ArtiFixer checkpoints/ArtiFixer
```

`bootstrap.sh` downloads into here on a fresh machine. What is needed:

| path | size | source |
|---|---|---|
| `ArtiFixer/artifixer-1.3b.pt` | 6.3 GiB | `nvidia/ArtiFixer` |
| `ArtiFixer/artifixer-14b-bf16.pt` | 32 GiB | converted from `artifixer-14b.pt` by `subprojects/repair_pipeline/scripts/convert_ckpt_bf16.py` |
| `ArtiFixer/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/` | 27 GiB | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| `ArtiFixer/Wan-AI/Wan2.1-T2V-14B-Diffusers/` | 101 GiB | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |

Only the variant you actually run is required (`./bootstrap.sh --variant 1.3b`
downloads just the 1.3B files). The caption text encoder comes from the same
Wan repo as the model: both repos ship the identical UMT5-XXL, so
`checkpoints.text_encoder: null` (the default) picks whichever one is installed.
