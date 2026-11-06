"""Minimal ArtiFixer inference: one clip in, frames out, no eval harness.

The stock repo's path is ``model_eval/run_inference.py`` -> dataset -> PNG
round-trip -> mp4 encode. Measured on this scene family, only ~6% of that wall
time was the diffusion; ~76% was writing 305 comparison/sidecar PNGs. So this
driver builds the tensors directly and never touches those code paths.

Everything here is a thin call into the repo's own modules -- the model maths is
not reimplemented:

    get_kv_cache_pipe        model_training/utils/train_utils.py
    pipe.forward_inference   model_training/pipeline/kv_cache_pipeline.py

``compute_camera_rays`` / ``load_encoded_prompt`` are vendored in
``camera_rays.py``: importing them from ``model_training.data.utils`` drags in
``threedgrut`` (a CUDA extension) and the whole DL3DV dataset stack.

Checkpoint loading is inlined (~10 lines) so that nothing from ``model_eval``
is imported at all -- that package is the harness this driver replaces.

Contracts that are easy to get wrong and fail silently (verified against
``reconstructed_colmap_eval.py`` and the pipeline):

* RGB and neighbours are float **[0, 1]** -- the VAE preprocess maps them to
  [-1, 1] itself, and passing [-1, 1] makes it warn and skip the mapping.
* ``rendered_opacity`` and ``camera_rays`` must be **bf16**; ``w2cs``/``Ks``/
  ``neighbor_*`` must be **float32**. Mixing raises inside PRoPE.
* ``T_padded ≡ 1 (mod 4)`` and its latent count must be divisible by
  ``frames_per_block``; the pipeline asserts the latter.
* The batch dimension is always 1 (the KV cache is sized per batch).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _pad_frames(num_frames: int, frames_per_block: int, scale: int = 4) -> int:
    """Frames to append so VAE alignment and block divisibility both hold."""
    n = num_frames
    n += (scale - (n - 1) % scale) % scale
    latent = (n - 1) // scale + 1
    padded_latent = math.ceil(latent / frames_per_block) * frames_per_block
    return (padded_latent - 1) * scale + 1 - num_frames


def _pad_last(x, pad: int, dim: int):
    """Repeat the last slice ``pad`` times along ``dim`` (matches pad_temporal)."""
    import torch

    if pad <= 0:
        return x
    last = x.select(dim, -1).unsqueeze(dim)
    return torch.cat([x, last.expand(*[-1 if i != dim else pad for i in range(x.dim())])], dim=dim)


def _torch_attention_compat() -> None:
    """Let ArtiFixer's transformer import on torch < 2.10.

    It imports ``activate_flash_attention_impl`` / ``list_flash_attention_impls``
    at module level to opt into FA4 kernels; older torch has neither. With an
    empty implementation list its own selection (``_select_attention_config``)
    skips FA4 and keeps flex_attention on the default Triton backend, which is
    what a torch without FA4 would do anyway. No-op on newer torch.
    """
    import torch.nn.attention as attn

    if not hasattr(attn, "list_flash_attention_impls"):
        attn.list_flash_attention_impls = lambda: []
    if not hasattr(attn, "activate_flash_attention_impl"):
        def _unavailable(name):
            raise RuntimeError(f"flash attention impl {name!r} needs torch >= 2.10")
        attn.activate_flash_attention_impl = _unavailable


def _build_pipe(cfg, device):
    """Build the distilled KV-cache repair pipeline from the base repo, nothing else.

    Same construction as ArtiFixer's ``get_kv_cache_pipe`` (train_utils.py:354),
    inlined because importing ``train_utils`` pulls the whole training dataset
    stack (``data.dl3dv_*``) and ``threedgrut``, a CUDA extension.
    """
    import torch
    from diffusers import AutoencoderKLWan, WanTransformer3DModel

    _torch_attention_compat()
    from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND

    backend = cfg.get("attention_backend") or default_attention_backend(_DEFAULT_ATTENTION_BACKEND)
    vae = AutoencoderKLWan.from_pretrained(
        cfg.model_id, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device)

    from model_training.pipeline.kv_cache_pipeline import ArtifixerKvCachePipeline

    return ArtifixerKvCachePipeline(
        vae=vae,
        transformer=WanTransformer3DModel.from_config(
            cfg.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ),
        tokenizer=None,
        text_encoder=None,
        frames_per_block=int(cfg.frames_per_block),
        local_attn_size=int(cfg.local_attn_size),
        sink_size=int(cfg.sink_size),
        gradient_checkpointing=False,
        checkpoint_every_n_blocks=1,
        attention_backend=backend,
    ).to(torch.bfloat16).to(device)


def default_attention_backend(artifixer_default: str | None, platform: str | None = None) -> str | None:
    """ArtiFixer's auto choice, except on Windows: SDPA (``native``).

    ArtiFixer picks ``_flash_3`` (Dao-AILab FA3) on any sm_90 GPU, and FA3 has
    no Windows build.
    """
    import sys

    return "native" if (platform or sys.platform) == "win32" else artifixer_default


# model.low_vram = trim_kv_cache + this many references per VAE encode. Measured
# on 1.3B, 960x544, 25-frame clips, 12 references, flash/cuDNN SDPA disabled (the
# Windows worst case): every repair phase <= 11.4 GB allocated / 13.1 GB per
# nvidia-smi, against 19.5 GB (reference encode) and 16.3 GB (denoise) without.
# With the default kernels the output is bit-identical to low_vram: false.
LOW_VRAM_NEIGHBORS_PER_ENCODE = 2


def _trim_kv_cache(pipe) -> None:
    """Size the self-attention KV cache to the clip instead of ``local_attn_size``.

    The KV-cache pipeline allocates ``local_attn_size`` latent frames (21 → ~7.9 GB
    for 1.3B at 960x544) whatever the clip length. Attention only ever reads the
    filled part (``[:local_end_index]``) and nothing is evicted until the clip
    outgrows the buffer, so a buffer of ``min(local_attn_size, latent frames)``
    gives the same result; a 25-frame clip (7 latent frames) needs a third.
    ``RepairModel.repair`` sets ``pipe._nvs_cache_frames`` per clip.
    """
    original = pipe._initialize_kv_cache

    def initialize(batch_size, frame_seq_length, num_frames):
        cap = getattr(pipe, "_nvs_cache_frames", None)
        return original(batch_size, frame_seq_length, min(num_frames, cap) if cap else num_frames)

    pipe._initialize_kv_cache = initialize


def _load_state_dict(model, checkpoint_pt: Path) -> None:
    """Stream the checkpoint onto the module (same semantics as ArtiFixer's loader).

    ``mmap=True`` keeps the 63 GB fp32 file out of anonymous memory; the bf16
    conversion happens in ``load_state_dict``'s copy.
    """
    import torch

    try:
        state = torch.load(checkpoint_pt, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, OSError):
        # old torch has no mmap=; some filesystems / Windows setups cannot map the file
        state = torch.load(checkpoint_pt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"expected a state dict in {checkpoint_pt}, got {type(state).__name__}")
    model.load_state_dict(state)


def _install_meta_build() -> None:
    """Build the transformer straight to bf16 instead of fp32-then-cast.

    ``WanTransformer3DModel.from_config(..., torch_dtype=bfloat16)`` ignores
    ``torch_dtype`` (diffusers never reads it), so the 14B is built in float32
    (57 GB) before ``.to(bfloat16)`` adds a second 28 GB copy. Constructing on
    the meta device and materialising with ``to_empty`` removes that peak.
    """
    import torch
    from diffusers import WanTransformer3DModel

    original = WanTransformer3DModel.from_config.__func__

    def from_config_meta(cls, config=None, return_unused_kwargs=False, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        prev = torch.get_default_dtype()
        if torch_dtype is not None:
            torch.set_default_dtype(torch_dtype)
        try:
            with torch.device("meta"):
                model = original(cls, config, return_unused_kwargs=False, **kwargs)
        finally:
            torch.set_default_dtype(prev)
        model = model.to_empty(device="cpu")
        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                tensor.zero_()
        return model

    WanTransformer3DModel.from_config = classmethod(from_config_meta)


@dataclass
class Clip:
    """One trajectory's inputs, already prepared."""

    rgb: np.ndarray            # (T, H, W, 3) uint8
    opacity: np.ndarray        # (T, H, W) uint8
    neighbors: np.ndarray      # (M, H, W, 3) uint8
    transforms: dict
    targets: list[int]
    neighbor_indices: list[int]
    prompt_h5: Path | None


class RepairModel:
    """Loads the checkpoint once; ``repair`` can then be called per clip."""

    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.model.get("repo"):
            raise FileNotFoundError("no ArtiFixer checkout configured: set model.repo or $ARTIFIXER_REPO "
                                    "(nvs_repair's bootstrap.sh clones one and sets it for you)")
        repo = Path(cfg.model.repo).expanduser()
        if not repo.is_dir():
            raise FileNotFoundError(f"model.repo does not exist: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        self.repo = repo
        self.pipe = None
        self.device = None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        import torch

        _install_meta_build()
        pipe = _build_pipe(self.cfg.model, torch.device(self.cfg.model.device))

        _load_state_dict(pipe.transformer, Path(self.cfg.model.checkpoint_pt))
        pipe.transformer.eval()
        if self.cfg.model.get("trim_kv_cache") or self.cfg.model.get("low_vram"):
            _trim_kv_cache(pipe)
        self.pipe = pipe
        self.device = torch.device(self.cfg.model.device)

    # -- inference ---------------------------------------------------------
    def repair(self, clip: Clip) -> np.ndarray:
        """Run one clip. Returns ``(T, H, W, 3)`` float32 in [0, 1]."""
        import torch
        from .camera_rays import compute_camera_rays, load_encoded_prompt

        if self.pipe is None:
            self.load()

        cfg = self.cfg.model
        T, H, W = clip.rgb.shape[0], clip.rgb.shape[1], clip.rgb.shape[2]
        pad = _pad_frames(T, int(cfg.frames_per_block))

        def to_tensor(rgb_u8):
            return torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

        rgb = _pad_last(to_tensor(clip.rgb), pad, dim=0).unsqueeze(0)                # (1,T,3,H,W)
        import torch as _t

        op = _t.from_numpy(np.ascontiguousarray(clip.opacity).astype(np.float32) / 255.0)  # (T,H,W)
        opacity = _pad_last(op, pad, dim=0).unsqueeze(0)                                   # (1,T,H,W)
        neighbors = to_tensor(clip.neighbors).unsqueeze(0)                           # (1,M,3,H,W)

        cond = compute_camera_rays(
            transforms=clip.transforms,
            frame_indices=clip.targets,
            neighbor_indices=clip.neighbor_indices,
            scale=float(cfg.camera_scale),
            image_shape=(H, W),
            skip_vae_check=True,
        )
        lat_pad = ((T + pad - 1) // 4 + 1) - ((T - 1) // 4 + 1)
        self.pipe._nvs_cache_frames = (T + pad - 1) // 4 + 1      # read by _trim_kv_cache
        camera_rays = _pad_last(cond["camera_rays"], lat_pad, dim=0)
        w2cs = _pad_last(cond["w2cs"], lat_pad, dim=0)
        Ks = _pad_last(cond["Ks"], lat_pad, dim=0)

        # No prompt file = ArtiFixer's own zero-prompt fallback.
        prompt, _ = load_encoded_prompt([Path(clip.prompt_h5)] if clip.prompt_h5 else [])
        prompt = prompt.unsqueeze(0)

        dev = self.device
        if cfg.get("seed") is not None:
            torch.manual_seed(int(cfg.seed))
            torch.cuda.manual_seed_all(int(cfg.seed))
        with torch.no_grad():
            out = self.pipe.forward_inference(
                rendered_rgb=rgb.to(dev),
                rendered_opacity=opacity.to(dtype=torch.bfloat16, device=dev),
                neighbors=neighbors.to(dev),
                camera_rays=camera_rays.to(dtype=torch.bfloat16, device=dev).unsqueeze(0),
                w2cs=w2cs.to(dtype=torch.float32, device=dev).unsqueeze(0),
                neighbor_w2cs=cond["neighbor_w2cs"].to(dtype=torch.float32, device=dev).unsqueeze(0),
                Ks=Ks.to(dtype=torch.float32, device=dev).unsqueeze(0),
                neighbor_Ks=cond["neighbor_Ks"].to(dtype=torch.float32, device=dev).unsqueeze(0),
                prompt=prompt.to(dtype=torch.bfloat16, device=dev),
                num_inference_steps=int(cfg.num_inference_steps),
                show_progress=False,
                max_neighbors_per_encode=cfg.get("max_neighbors_per_encode") or (
                    LOW_VRAM_NEIGHBORS_PER_ENCODE if cfg.get("low_vram") else None),
            )
        frames = out[0, :T].float().clamp(0, 1).cpu().numpy()   # (T,3,H,W)
        return np.transpose(frames, (0, 2, 3, 1))
