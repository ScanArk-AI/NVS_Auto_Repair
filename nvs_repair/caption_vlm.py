"""Describe the scene from a few rendered frames, then encode it for the model.

ArtiFixer is conditioned on a text prompt. Two kinds have been used:

* **scene** -- what the room *is*: layout, materials, colours, lighting. This is
  what a repair model should be told; the camera motion is already implied by
  the frames it is given.
* **scene_motion** -- the same plus one clause naming the camera move. It was
  used earlier because the prompt template asked for it, but naming a motion the
  model is not asked to change arguably just adds noise.

Where the description comes from, in order:

1. ``workflow.caption_text`` -- a literal description you write yourself; no
   network, no VLM.
2. A vision-language model behind an HTTP API (the only step in the workflow
   that needs network access), configured by environment variables:

   ``NVS_REPAIR_VLM_URL``    endpoint, or an API base such as
                             ``https://api.openai.com/v1`` (the path is added)
   ``NVS_REPAIR_VLM_KEY``    bearer token (omit for local servers)
   ``NVS_REPAIR_VLM_MODEL``  model name
   ``NVS_REPAIR_VLM_API``    ``chat`` (default, OpenAI Chat Completions: OpenAI,
                             vLLM, Ollama, LM Studio, Gemini's OpenAI endpoint)
                             or ``responses`` (OpenAI Responses API)

``--set workflow.auto_caption=false`` skips the caption entirely and repairs
with ArtiFixer's zero prompt -- deliberately; a failed caption stops the run.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

_PATHS = {"chat": "/chat/completions", "responses": "/responses"}

_COMMON = """Look at the frames and reply with ONE English paragraph of 40-70 words.
Describe only what the frames support; if unsure of an object, use a generic term
instead of inventing one. Output ONLY the paragraph -- no preamble, no bullets,
no quotation marks."""

_INTRO = {
    "photos": """You are writing a TEXT PROMPT for a video restoration model.

The model receives frames rendered from a 3D Gaussian Splatting reconstruction
of the place shown in these REAL PHOTOS. The renders contain artifacts:
floating geometry, stretched or melted surfaces, holes, flicker. Your prompt
conditions the model on WHAT THE REAL SCENE IS, so it can push each frame back
toward it -- describe the place as these photos show it.

""",
    "renders": """You are writing a TEXT PROMPT for a video restoration model.

The model receives frames rendered from a 3D Gaussian Splatting reconstruction.
Those renders contain artifacts: floating geometry, stretched or melted
surfaces, flicker, and lighting that shifts between frames. Your prompt
conditions the model on WHAT THE REAL SCENE IS, so it can push each frame back
toward it. Describe the scene, not the artifacts.

""",
}

_COVER = """
Cover: whether it is indoors or outdoors and what kind of place it is, its
layout, the main objects and structures, materials and colours, the ground or
floor, what is overhead (ceiling or sky) and in the background, and the
lighting."""

_ENDING = {
    "scene": " Do NOT describe the camera move.",
    "scene_motion": " End with one short clause naming the camera motion and stating that"
                    " lighting and colour stay constant.",
}


def instructions(kind: str, source: str) -> str:
    if kind not in _ENDING:
        raise ValueError(f"unknown caption kind {kind!r}; pick from {sorted(_ENDING)}")
    return _INTRO[source] + _COMMON + _COVER + _ENDING[kind]


# Kept for callers that only want the render-based prompts.
INSTRUCTIONS = {k: instructions(k, "renders") for k in _ENDING}


def _data_url(path: Path, max_side: int = 768) -> str:
    """JPEG data URL, downscaled -- full-size captures are megabytes each."""
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _endpoint() -> tuple[str, str, str | None, str]:
    """``(url, api, key, model)`` from the environment, or a clear error."""
    url = os.environ.get("NVS_REPAIR_VLM_URL", "").strip()
    model = os.environ.get("NVS_REPAIR_VLM_MODEL", "").strip()
    api = os.environ.get("NVS_REPAIR_VLM_API", "chat").strip() or "chat"
    if api not in _PATHS:
        raise RuntimeError(f"NVS_REPAIR_VLM_API={api!r}; expected one of {sorted(_PATHS)}")
    if not url or not model:
        raise RuntimeError(
            "no VLM configured for captioning: set NVS_REPAIR_VLM_URL and NVS_REPAIR_VLM_MODEL "
            "(and NVS_REPAIR_VLM_KEY if the server needs one), or write the description yourself "
            "with --set workflow.caption_text='...', or skip it with --set workflow.auto_caption=false")
    url = url.rstrip("/")
    if not url.endswith(tuple(_PATHS.values())):
        url += _PATHS[api]
    return url, api, os.environ.get("NVS_REPAIR_VLM_KEY") or None, model


def _request_body(api: str, model: str, prompt: str, frames: list[Path]) -> dict:
    images = [_data_url(f) for f in frames]
    if api == "responses":
        content = [{"type": "input_text", "text": prompt}]
        content += [{"type": "input_image", "image_url": u} for u in images]
        return {"model": model, "input": [{"role": "user", "content": content}], "max_output_tokens": 600}
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in images]
    return {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 600}


def _response_text(api: str, data: dict) -> str | None:
    if api == "responses":
        for item in data.get("output") or []:
            for c in item.get("content") or []:
                if c.get("type") == "output_text" and c.get("text"):
                    return c["text"]
        return None
    for choice in data.get("choices") or []:
        msg = (choice.get("message") or {}).get("content")
        if isinstance(msg, list):   # some servers return content parts
            msg = "".join(part.get("text", "") for part in msg if isinstance(part, dict))
        if msg:
            return msg
    return None


def describe(frames: list[Path], kind: str = "scene", timeout: int = 300, source: str = "renders") -> str:
    """Ask the configured VLM for one paragraph describing the scene in ``frames``.

    ``source`` says what the images are -- ``photos`` (real captures) or
    ``renders`` (3DGS renders) -- which changes how the VLM is told to read them.
    """
    prompt = instructions(kind, source)
    url, api, key, model = _endpoint()
    body = json.dumps(_request_body(api, model, prompt, frames)).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if data.get("error"):
        raise RuntimeError(data["error"])
    text = _response_text(api, data)
    if not text:
        raise RuntimeError(f"no text in the {api} response: {json.dumps(data)[:300]}")
    return text.strip()


def ensure_caption(cfg, out_dir: Path, kind: str = "scene", frames: int = 4) -> Path:
    """Write ``<out>/caption.h5`` for this scene, reusing it if already there."""
    target = Path(out_dir) / "caption.h5"
    if target.is_file():
        return target

    literal = (cfg.workflow.get("caption_text") or "").strip() if cfg.get("workflow") else ""
    if literal:
        return _write(cfg, target, literal)

    source = str(cfg.workflow.get("caption_source", "photos"))
    picks = _photos(cfg, frames) if source == "photos" else []
    if source == "photos" and not picks:
        print("caption: no photos found through the repair scene config; describing renders instead")
    if not picks:
        source, picks = "renders", _renders(cfg, frames)
    return _write(cfg, target, describe(picks, kind=kind, source=source))


def _photos(cfg, n: int) -> list[Path]:
    """``n`` captured photos spread over the capture, via the repair scene config."""
    try:
        from omegaconf import OmegaConf

        _repair_path()
        from repair_pipeline import colmap_io
        from repair_pipeline.config import _resolve_scene

        scene = OmegaConf.create({"scene": OmegaConf.load(str(cfg.repair.config)).scene})
        _resolve_scene(scene)
        _, ims = colmap_io.read_model(Path(scene.scene.sparse))
    except Exception as exc:  # noqa: BLE001 - fall back to renders
        print(f"caption: could not read the capture ({exc})")
        return []
    names = sorted(im.name for im in ims)
    paths = [Path(scene.scene.images) / names[int(round(i))]
             for i in np.linspace(0, len(names) - 1, max(1, n))] if names else []
    return [p for p in dict.fromkeys(paths) if p.is_file()]


def _renders(cfg, n: int) -> list[Path]:
    """The middle frame of up to ``n`` trajectories spread over the finder's list."""
    traj_root = Path(cfg.out) / "1_find" / "repair" / "trajectories"
    trajs = sorted(p for p in traj_root.iterdir() if p.is_dir()) if traj_root.is_dir() else []
    if not trajs:
        raise RuntimeError(f"no rendered trajectories under {traj_root}; run the find stage first")
    chosen = [trajs[int(round(i))] for i in np.linspace(0, len(trajs) - 1, min(max(1, n), len(trajs)))]
    picks = []
    for t in dict.fromkeys(chosen):
        frames = sorted((t / "frames").glob("*.png"))
        if frames:
            picks.append(frames[len(frames) // 2])
    if not picks:
        raise RuntimeError(f"no rendered frames under {traj_root}")
    return picks


def _repair_path() -> None:
    import sys

    sub = Path(__file__).resolve().parent.parent / "subprojects" / "repair_pipeline"
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))


# Encodings of descriptions already seen, keyed by the text and the encoder
# (see _cache_key). Loading the encoder reads 22.7 GB -- about two minutes
# from a hard disk -- while each run writes to a new output directory.
CAPTION_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "captions"


def _cache_key(caption: str, encoder_dir: Path) -> str:
    import hashlib

    h = hashlib.sha256(b"caption.h5 v1\0" + caption.encode("utf-8") + b"\0")
    h.update(str(encoder_dir.resolve()).encode("utf-8") + b"\0")
    for name in ("model.safetensors.index.json", "config.json"):   # a different encoder at the same path
        f = encoder_dir / "text_encoder" / name
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:32]


def _write(cfg, target: Path, caption: str) -> Path:
    """Encode ``caption`` with Wan's text encoder; write caption.h5 and caption.txt.
    The same description with the same encoder is encoded once (CAPTION_CACHE)."""
    import shutil

    encoder_dir = Path(str(cfg.checkpoints.text_encoder))
    if not encoder_dir.is_absolute():
        encoder_dir = Path(cfg.checkpoints_dir) / encoder_dir
    cached = CAPTION_CACHE / f"{_cache_key(caption, encoder_dir)}.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    if cached.is_file():
        print(f"caption: this description was encoded before; reusing {cached}", flush=True)
        shutil.copyfile(cached, target)
    else:
        _repair_path()
        from repair_pipeline.caption import encode_caption, write_caption_h5

        embedding = encode_caption(caption, str(encoder_dir))
        write_caption_h5(target, caption=caption, embedding=embedding)
        try:
            CAPTION_CACHE.mkdir(parents=True, exist_ok=True)
            partial = cached.with_suffix(f".{os.getpid()}.partial")
            shutil.copyfile(target, partial)
            os.replace(partial, cached)
        except OSError as exc:      # a read-only install still repairs; it just re-encodes next time
            print(f"caption: could not cache the encoding ({exc})", flush=True)
    target.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
    return target
