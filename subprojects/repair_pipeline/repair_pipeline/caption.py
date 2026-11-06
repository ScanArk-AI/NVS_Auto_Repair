"""Write the ``caption.h5`` prompt file ArtiFixer loads.

The file is an HDF5 dataset of shape ``(512, 4096)`` bfloat16 (h5py has no
bf16 dtype, so the bits are stored as uint16 and reinterpreted on load) with the
raw caption in ``attrs["caption"]``. ArtiFixer's loader falls back to a zero
prompt of this shape, which is what an unconditioned run gets here too.
"""

from __future__ import annotations

from pathlib import Path

MAX_SEQUENCE_LENGTH = 512   # ArtiFixer's constants.MAX_SEQUENCE_LENGTH
TEXT_EMBED_DIM = 4096       # UMT5-XXL hidden width


def encode_caption(caption: str, text_encoder_model_id: str,
                   max_sequence_length: int = MAX_SEQUENCE_LENGTH):
    """Encode with Wan's UMT5-XXL text encoder -> ``(seq_len, 4096)`` bf16."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_id, subfolder="tokenizer")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = load_text_encoder(Path(text_encoder_model_id) / "text_encoder", device, torch.bfloat16)

    inputs = tokenizer([caption], padding="max_length", max_length=max_sequence_length,
                       truncation=True, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        embeds = encoder(inputs.input_ids.to(encoder.device),
                         inputs.attention_mask.to(encoder.device)).last_hidden_state

    # Keep only the real tokens and let the writer zero-pad. With
    # padding="max_length" the encoder emits content for every one of the 512
    # positions -- pad-token outputs are not zero (measured mean |value| 0.0605
    # vs 0.0584 for real tokens) -- and ArtiFixer's attention has no mask to
    # suppress them, so a full-width embedding would feed it fabricated rows.
    seq_len = int(inputs.attention_mask.gt(0).sum(dim=1)[0].item())
    out = embeds[0][:seq_len].to(torch.bfloat16).cpu()
    # The caller may load the 14B model on this GPU next (or gate on free
    # VRAM); hand the ~11 GB encoder back instead of leaving it cached.
    del encoder, embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# Bytes of checkpoint data staged in CPU memory at a time.
LOAD_CHUNK_BYTES = 256 << 20


def load_text_encoder(path: Path, device, dtype):
    """Load a UMT5 encoder checkpoint into ``dtype`` on ``device``, reading
    each shard front to back in blocks of at most ``LOAD_CHUNK_BYTES``.

    Wan ships the encoder as 22.7 GB of float32 in five shards.
    ``UMT5EncoderModel.from_pretrained`` (transformers 5.5) maps every shard at
    once and materialises tensors in float32 on four threads before casting:
    on a 32 GB Windows machine that took 7.5 minutes and sometimes failed to
    allocate the 4.2 GB embedding. Reading through safetensors' memory map one
    shard at a time fixed the memory but not the speed -- Windows faulted the
    mapping in at ~40 MB/s from a disk that reads ~120 MB/s sequentially. So
    the model is built on the meta device, its parameters are allocated once
    directly in ``dtype`` (11.4 GB for bf16), and the shards are read with
    plain sequential reads into one reused buffer. The weights are the same as
    ``from_pretrained`` gives.
    """
    import json
    import time

    import torch
    from transformers import UMT5Config, UMT5EncoderModel

    path = Path(path)
    with torch.device("meta"):
        encoder = UMT5EncoderModel(UMT5Config.from_pretrained(path))
    # The checkpoint stores the tied embedding once, as shared.weight.
    encoder.encoder.embed_tokens.weight = encoder.shared.weight
    if any(True for _ in encoder.buffers()):
        raise RuntimeError("UMT5EncoderModel gained buffers; load_text_encoder only allocates parameters")
    allocated = {}
    for module in encoder.modules():
        for name, p in module._parameters.items():
            if p is not None:
                if id(p) not in allocated:   # once per tensor, so the tie costs nothing
                    allocated[id(p)] = torch.nn.Parameter(
                        torch.empty(p.shape, dtype=dtype, device=device), requires_grad=False)
                module._parameters[name] = allocated[id(p)]

    index = path / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
              if index.is_file() else ["model.safetensors"])
    params = dict(encoder.named_parameters())
    missing = set(params)
    total, t0 = 0, time.monotonic()
    with torch.no_grad():
        for k, shard in enumerate(shards, 1):
            for name, block, rows in _read_safetensors(path / shard, LOAD_CHUNK_BYTES):
                if name == "encoder.embed_tokens.weight":    # some exports also store the tied copy
                    continue
                if name not in params:
                    raise KeyError(f"{path / shard}: unexpected tensor {name}")
                target = params[name] if params[name].dim() else params[name].view(1)
                if tuple(block.shape[1:]) != tuple(target.shape[1:]) or rows.stop > target.shape[0]:
                    raise ValueError(f"{name}: checkpoint shape does not match the model's {tuple(target.shape)}")
                target[rows].copy_(block)
                if rows.stop == target.shape[0]:
                    missing.discard(name)
            total += (path / shard).stat().st_size
            took = time.monotonic() - t0
            print(f"text encoder: {k}/{len(shards)} shards, {total / 1e9:.1f} GB in {took:.0f} s "
                  f"({total / 1e6 / max(took, 1e-3):.0f} MB/s)", flush=True)
    missing.discard("encoder.embed_tokens.weight")
    if missing:
        raise KeyError(f"{path}: no weights for {sorted(missing)[:5]}")
    return encoder.eval()


_SAFETENSORS_DTYPES = {"F32": "float32", "F16": "float16", "BF16": "bfloat16", "F64": "float64"}


def _read_safetensors(file: Path, chunk_bytes: int):
    """Yield ``(name, block, rows)`` over a .safetensors file in file order:
    ``block`` holds rows ``rows`` (a slice of dim 0; a scalar is one row) of
    tensor ``name`` in at most ``chunk_bytes`` (or one row), and is only valid
    until the next item -- the buffer is reused. The format is an 8-byte
    little-endian header length, a JSON header, then the raw tensors."""
    import json
    import math
    import struct

    import torch

    buffer = bytearray(chunk_bytes)
    with open(file, "rb", buffering=0) as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        base = 8 + n
        for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if info["dtype"] not in _SAFETENSORS_DTYPES:
                raise ValueError(f"{file}: {name} has unsupported dtype {info['dtype']}")
            src = getattr(torch, _SAFETENSORS_DTYPES[info["dtype"]])
            itemsize = torch.empty((), dtype=src).element_size()
            shape = [int(d) for d in info["shape"]] or [1]        # a scalar is read as one row
            begin, end = info["data_offsets"]
            row_bytes = math.prod(shape[1:]) * itemsize
            if end - begin != shape[0] * row_bytes:
                raise ValueError(f"{file}: {name} spans {end - begin} bytes, not {shape} x {itemsize}")
            if row_bytes > len(buffer):
                buffer = bytearray(row_bytes)
            step = len(buffer) // row_bytes
            f.seek(base + begin)
            for i in range(0, shape[0], step):
                rows = min(step, shape[0] - i)
                view = memoryview(buffer)[:rows * row_bytes]
                got = 0
                while got < len(view):
                    r = f.readinto(view[got:])
                    if not r:
                        raise EOFError(f"{file}: truncated in {name}")
                    got += r
                block = torch.frombuffer(buffer, dtype=src, count=rows * row_bytes // itemsize)
                yield name, block.view(rows, *shape[1:]), slice(i, i + rows)


def write_caption_h5(path: Path, caption: str = "", embedding=None, dataset_name: str = "0000") -> Path:
    import h5py
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if embedding is None:
        raw = np.zeros((MAX_SEQUENCE_LENGTH, TEXT_EMBED_DIM), dtype=np.uint16)
    else:
        import torch

        tensor = embedding.detach().to("cpu")
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        if tensor.shape[0] < MAX_SEQUENCE_LENGTH:
            tensor = torch.cat([tensor, torch.zeros(MAX_SEQUENCE_LENGTH - tensor.shape[0],
                                                    tensor.shape[1], dtype=torch.bfloat16)], dim=0)
        elif tensor.shape[0] > MAX_SEQUENCE_LENGTH:
            tensor = tensor[:MAX_SEQUENCE_LENGTH]
        raw = tensor.view(torch.uint16).numpy()

    with h5py.File(path, "w") as f:
        dataset = f.create_dataset(dataset_name, data=raw)
        dataset.attrs["caption"] = caption
    return path


def resolve_prompt(cfg, out_dir: Path) -> tuple[Path, str]:
    """Return ``(caption.h5 path, caption text)``, creating one if needed.

    Priority: an existing ``prompt.caption_h5``; else ``prompt.text`` /
    ``prompt.text_file`` encoded here; else a zero prompt (unconditioned).
    """
    if cfg.prompt.caption_h5:
        return Path(cfg.prompt.caption_h5), ""

    text = cfg.prompt.text or ""
    if not text and cfg.prompt.text_file:
        text = Path(cfg.prompt.text_file).read_text(encoding="utf-8").strip()

    # Always (re)write: reusing an existing caption.h5 when no text is configured
    # would silently condition a "no prompt" run on whatever an earlier run left
    # in this output directory.
    target = Path(out_dir) / "caption.h5"
    if text:
        if not cfg.prompt.text_encoder:
            raise ValueError("prompt.text is set but prompt.text_encoder is not; "
                             "point it at a local Wan diffusers repo")
        embedding = encode_caption(text, str(cfg.prompt.text_encoder))
    else:
        embedding = None
    write_caption_h5(target, caption=text, embedding=embedding)
    return target, text
