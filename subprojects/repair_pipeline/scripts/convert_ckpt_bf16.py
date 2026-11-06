"""Convert a float32 transformer checkpoint to bfloat16, streaming.

Why bother: ArtiFixer's 14B checkpoint on disk is 63 GB of float32, but the
model is built and run entirely in bfloat16. Every load therefore pays twice --
once to stream 63 GB off a disk measured at 144 MB/s (7.5 min), and again to
convert 1455 tensors fp32 -> bf16 on the CPU (another ~7.5 min).

Worse, 63 GB does not fit in this container's 62 GB of RAM, so the page cache
can never hold the file and *every* load re-reads it from scratch. Halving the
file to ~31.5 GB fixes both: the conversion disappears and the file now fits in
memory, so the second and later loads come from cache.

The conversion itself has to be careful. Naively materialising the bf16 result
holds 31.5 GB of anonymous memory while the source's 63 GB streams through the
page cache underneath it, which is exactly the shape of allocation that has
OOM-killed several runs on the development machine. So this walks the state dict one
tensor at a time, drops its reference immediately, and periodically asks the
kernel to evict the source file's clean pages. It also refuses to start unless
there is genuine headroom, and aborts rather than risking the container.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import torch


def mem(key: str) -> float:
    with open("/sys/fs/cgroup/memory.stat") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0] == key:
                return int(parts[1]) / 1024**3
    return 0.0


def current_gib() -> float:
    return mem("anon") + mem("file") + mem("kernel")


def cap_gib() -> float:
    """The cgroup's real limit, so the guard adapts if the box is resized.

    This started as a hard-coded 58 (sized against a 62 GiB cap); when the
    instance was moved to 248 GiB that constant would have aborted the run
    immediately even though there was plenty of room.
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            raw = fh.read().strip()
        if raw.isdigit():
            return int(raw) / 1024**3
    except OSError:
        pass
    return 0.0


def convert(src: Path, dst: Path, limit_gib: float | None = None) -> None:
    if dst.exists():
        print(f"[convert] {dst} already exists, refusing to overwrite", flush=True)
        return

    cap = cap_gib()
    if limit_gib is None:
        # Leave headroom for the kernel and for the write that follows: the
        # finished bf16 dict is held in RAM while torch.save streams it out.
        limit_gib = cap * 0.85 if cap else 58.0

    print(f"[convert] {src} -> {dst}", flush=True)
    print(f"[convert] cgroup cap {cap:.0f} GiB, abort threshold {limit_gib:.0f} GiB", flush=True)
    t0 = time.time()
    sd = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
    keys = list(sd.keys())
    n_params = sum(int(sd[k].numel()) for k in keys)
    print(f"[convert] {len(keys)} tensors, {n_params / 1e9:.2f}B params", flush=True)
    print(f"[convert] before: current={current_gib():.1f} GiB "
          f"(anon={mem('anon'):.1f} file={mem('file'):.1f})", flush=True)

    fd = os.open(src, os.O_RDONLY)
    out: dict[str, torch.Tensor] = {}
    try:
        for i, k in enumerate(keys):
            v = sd[k]
            # .to() reads through the mmap, so this faults in that tensor's
            # pages; the bf16 copy is the only thing we keep.
            out[k] = v.to(torch.bfloat16).contiguous().clone()
            del v
            if i % 64 == 0:
                gc.collect()
                try:
                    # Ask for the source file's clean pages back. Pages still
                    # mapped by an in-flight read survive this, which is fine --
                    # the kernel reclaims them under pressure regardless.
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
                c = current_gib()
                if c > limit_gib:
                    raise MemoryError(
                        f"memory.current reached {c:.1f} GiB (threshold {limit_gib:.0f}); "
                        f"aborting before the kernel OOM-kills the container"
                    )
                if i % 256 == 0 or i == len(keys) - 1:
                    print(f"  [{i:4d}/{len(keys)}] current={c:5.1f} "
                          f"(anon={mem('anon'):5.1f} file={mem('file'):5.1f}) "
                          f"{time.time() - t0:6.1f}s", flush=True)
    finally:
        os.close(fd)

    gc.collect()
    print(f"[convert] converted in {time.time() - t0:.1f}s, writing {dst} ...", flush=True)
    torch.save(out, dst)
    size = dst.stat().st_size
    print(f"[convert] wrote {size / 1024**3:.1f} GiB in {time.time() - t0:.1f}s total", flush=True)

    # Round-trip check: the whole point is that ArtiFixer can load this.
    back = torch.load(dst, map_location="cpu", weights_only=True, mmap=True)
    dtypes = {back[k].dtype for k in list(back)[:200]}
    print(f"[convert] round-trip ok: {len(back)} tensors, dtypes={dtypes}", flush=True)
    missing = set(keys) - set(back)
    if missing:
        print(f"[convert] WARNING: {len(missing)} keys missing", flush=True)
    assert {int(back[k].numel()) for k in keys[:50]} == {int(sd[k].numel()) for k in keys[:50]}, \
        "shape changed during conversion"
    print("[convert] shapes preserved", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/convert_ckpt_bf16.py <artifixer-14b.pt> <artifixer-14b-bf16.pt>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    convert(src, dst)
