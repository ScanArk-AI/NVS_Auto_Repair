#!/bin/bash
# Build the environment from nothing: one venv holding both stages (GAVIS and
# three CUDA rasterisers cloned at pinned commits, see configs/workflow.yaml;
# the repair stack and gsplat), the ArtiFixer checkout, and the weights.
#
#   ./bootstrap.sh                 # into ~/nvs_repair_runtime, all weights
#   ./bootstrap.sh --variant 1.3b  # only what a 1.3B run needs (~100 GiB less)
#   ./bootstrap.sh --root /data/wf # elsewhere
#   ./bootstrap.sh --only finder   # just the finder's part (no weights, configs untouched)
#
# Everything is idempotent: a second run skips what already imports/downloads.
# The only outside prerequisites are python3.12, git, and a CUDA toolkit whose
# version matches the torch wheel.
#
# Networks that cannot reach GitHub / Hugging Face directly (all optional):
#   GH_PROXY=https://<github-mirror>/   prefix tried when a GitHub clone fails
#   HF_ENDPOINT=https://<hf-mirror>     download the weights from a mirror
#   HF_HUB_DISABLE_XET=1                if the mirror's Xet storage returns 401s
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
RUNTIME=${RUNTIME:-$HOME/nvs_repair_runtime}
VARIANT=all
SKIP_WEIGHTS=0
ONLY=all
PY=${PYTHON:-python3.12}

while [ $# -gt 0 ]; do
  case "$1" in
    --root) RUNTIME=$2; shift 2 ;;
    --root=*) RUNTIME=${1#*=}; shift ;;
    --variant) VARIANT=$2; shift 2 ;;
    --variant=*) VARIANT=${1#*=}; shift ;;
    --python) PY=$2; shift 2 ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --only) ONLY=$2; shift 2 ;;
    --only=*) ONLY=${1#*=}; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$ONLY" in all|finder|repair) ;; *) echo "--only must be all, finder or repair" >&2; exit 2 ;; esac
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

log() { echo -e "\n[bootstrap] $*"; }
cfg() { "$PY" -c "import yaml,sys;print(yaml.safe_load(open('$ROOT/configs/workflow.yaml'))$1)"; }

TORCH_VERSION=$(cfg "['torch_version']")
TORCH_INDEX=$(cfg "['torch_index']")
ARTIFIXER_COMMIT=$(cfg "['artifixer_commit']")
# One env for both stages: GAVIS's rasteriser only runs on torch 2.7, and the
# repair stack (diffusers, ArtiFixer, gsplat) works there too.
ENV="$RUNTIME/envs/nvs"
ARTIFIXER_DIR="$RUNTIME/ArtiFixer"
WEIGHTS="$ROOT/checkpoints/ArtiFixer"

mkdir -p "$RUNTIME/envs" "$ROOT/checkpoints"

# --- prerequisites ----------------------------------------------------------
log "prerequisites"
command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v "$PY" >/dev/null || { echo "$PY not found (use --python)" >&2; exit 1; }
echo "  python: $("$PY" -V)"
if command -v nvcc >/dev/null; then
  echo "  nvcc:   $(nvcc --version | tail -1)"
else
  echo "  WARNING: nvcc not on PATH -- the three rasterisers cannot compile." >&2
  echo "           Point CUDA_HOME/PATH at a CUDA toolkit matching torch $TORCH_VERSION." >&2
fi
echo "  runtime: $RUNTIME"

# --- venv -------------------------------------------------------------------
if [ -x "$ENV/bin/python" ]; then
  log "venv exists: $ENV"
else
  log "creating venv $ENV"
  "$PY" -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install -q --upgrade pip wheel

# A constraints file, not just a first install: requirements pin torch without
# an index, so a later `pip install -r requirements.txt` would happily replace
# the cu128 wheel with the default PyPI build -- which may lack kernels for the
# GPU. Pinning the local version makes that impossible.
CONSTRAINTS="$RUNTIME/constraints.txt"
printf 'torch==%s+cu128\n' "$TORCH_VERSION" > "$CONSTRAINTS"

install_torch() {
  local have
  have=$("$ENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null || true)
  if [ "$have" = "${TORCH_VERSION}+cu128" ]; then
    echo "  torch $have already present"
    return
  fi
  echo "  installing torch $TORCH_VERSION+cu128 from $TORCH_INDEX"
  "$ENV/bin/python" -m pip install -q -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
}

pip_reqs() {  # pip_reqs <requirements>
  "$ENV/bin/python" -m pip install -q -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX" -r "$1"
}

log "torch"
install_torch
# ninja before anything compiles: the three rasterisers build with it (much
# faster than the fallback), and gsplat JIT-compiles with it on first render --
# render_backend puts the venv's bin/ on PATH for that, since the venv is never
# activated, so it has to be in there.
if [ -x "$ENV/bin/ninja" ]; then
  echo "  ninja already present"
else
  echo "  installing ninja"
  "$ENV/bin/python" -m pip install -q ninja
fi

# --- third-party sources (finder) --------------------------------------------
# Cloned, not vendored: GAVIS has no licence file and the INRIA rasteriser is
# non-commercial, so neither is redistributed here. Each is pinned to a commit.
SRC="$RUNTIME/src"
GH_PROXY=${GH_PROXY:-}
gh_clone() {  # gh_clone <url> <dir>: direct first, then through $GH_PROXY if set
  git clone --quiet "$1" "$2" 2>/dev/null && return 0
  if [ -n "$GH_PROXY" ] && [[ $1 == https://github.com/* ]]; then
    echo "  direct clone failed, retrying through $GH_PROXY"
    git clone --quiet "${GH_PROXY%/}/$1" "$2" && return 0
  fi
  echo "  could not clone $1 (behind a firewall? set GH_PROXY to a GitHub mirror prefix)" >&2
  return 1
}
fetch_src() {  # fetch_src <key in workflow.yaml third_party> -> $SRC/<key>
  local key=$1 url commit dir
  url=$(cfg "['third_party']['$key']['url']")
  commit=$(cfg "['third_party']['$key']['commit']")
  dir="$SRC/$key"
  if [ ! -d "$dir/.git" ]; then
    echo "  cloning $url"
    gh_clone "$url" "$dir" || exit 1
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch --quiet origin
    # --force: patch_rasterizer_headers edits tracked files; it runs again after this.
    git -C "$dir" checkout --quiet --force "$commit"
  fi
  git -C "$dir" submodule update --quiet --init --recursive 2>/dev/null \
    || { [ -n "$GH_PROXY" ] && git -C "$dir" -c url."${GH_PROXY%/}/https://github.com/".insteadOf=https://github.com/ \
           submodule update --quiet --init --recursive; } \
    || { echo "  submodules of $key could not be fetched (set GH_PROXY?)" >&2; exit 1; }
  echo "  $key at ${commit:0:7}"
}

patch_rasterizer_headers() {
  # Declare integer types directly; do not rely on CUDA/STL transitive includes.
  "$ENV/bin/python" - "$SRC" <<'PY'
from pathlib import Path
import re
import sys

src = Path(sys.argv[1])
for key in ("diff_gaussian_rasterization", "gavis_rasterizer"):
    for name in ("rasterizer_impl.h", "forward.h", "backward.h"):
        header = src / key / "cuda_rasterizer" / name
        content = header.read_bytes()
        if re.search(rb"(?m)^[ \t]*#[ \t]*include[ \t]*<cstdint>", content):
            continue
        first_include = re.search(rb"(?m)^[ \t]*#[ \t]*include\b", content)
        if first_include is None:
            raise RuntimeError(f"No include anchor in {header}")
        newline = b"\r\n" if b"\r\n" in content else b"\n"
        offset = first_include.start()
        header.write_bytes(content[:offset] + b"#include <cstdint>" + newline + content[offset:])
        print(f"  patched {key}/cuda_rasterizer/{name}: added <cstdint>")
PY
}

# --- finder env -------------------------------------------------------------
if want finder; then
  log "finder dependencies"
  pip_reqs "$ROOT/subprojects/repair_view_finder/requirements.txt"
  mkdir -p "$SRC"
  for key in gavis diff_gaussian_rasterization gavis_rasterizer simple_knn; do
    fetch_src "$key"
  done
  patch_rasterizer_headers
  # GAVIS is pure python; its dependencies are already in requirements.txt.
  "$ENV/bin/python" -m pip install -q --no-deps --no-build-isolation "$SRC/gavis"
  # <source key> <module that proves it is built>
  for spec in "simple_knn simple_knn._C" \
              "diff_gaussian_rasterization diff_gaussian_rasterization._C" \
              "gavis_rasterizer gavis_rasterization._C"; do
    set -- $spec
    if "$ENV/bin/python" -c "import torch, $2" 2>/dev/null; then
      echo "  $1 already built"
    else
      log "  building $1"
      "$ENV/bin/python" -m pip install -q --no-build-isolation "$SRC/$1"
    fi
  done
fi

# --- repair stage -------------------------------------------------------------
if want repair; then
  log "repair dependencies"
  pip_reqs "$ROOT/subprojects/repair_pipeline/requirements.txt"

  # gsplat is pure python (it JIT-compiles kernels at first render). Install with
  # --no-deps so pip cannot decide to "helpfully" swap torch or numpy.
  if "$ENV/bin/python" -c 'import gsplat' 2>/dev/null; then
    echo "  gsplat already present"
  else
    echo "  installing gsplat and jaxtyping"
    "$ENV/bin/python" -m pip install -q --no-deps gsplat jaxtyping
  fi
fi

# --- torch still the CUDA build? -------------------------------------------
got=$("$ENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo missing)
if [ "$got" != "${TORCH_VERSION}+cu128" ]; then
  log "torch in $ENV is $got, expected ${TORCH_VERSION}+cu128 -- reinstalling"
  "$ENV/bin/python" -m pip install -q --force-reinstall --no-deps \
    -c "$CONSTRAINTS" --extra-index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
fi

if [ "$ONLY" = finder ]; then
  log "done (finder part only): $ENV/bin/python -- configs/envs.local.yaml left as it was"
  exit 0
fi

# --- ArtiFixer checkout -----------------------------------------------------
GH=https://github.com/nv-tlabs/ArtiFixer.git
if [ -d "$ARTIFIXER_DIR/.git" ]; then
  log "ArtiFixer already cloned"
else
  log "cloning ArtiFixer -> $ARTIFIXER_DIR"
  gh_clone "$GH" "$ARTIFIXER_DIR" || exit 1
fi
if git -C "$ARTIFIXER_DIR" rev-parse --verify --quiet "$ARTIFIXER_COMMIT" >/dev/null; then
  git -C "$ARTIFIXER_DIR" checkout --quiet "$ARTIFIXER_COMMIT"
  echo "  at $ARTIFIXER_COMMIT"
else
  echo "  WARNING: $ARTIFIXER_COMMIT not found; staying on $(git -C "$ARTIFIXER_DIR" rev-parse --short HEAD)" >&2
fi

# --- weights ----------------------------------------------------------------
if [ "$SKIP_WEIGHTS" = 1 ]; then
  log "skipping weights (--skip-weights)"
elif [ -f "$WEIGHTS/artifixer-1.3b.pt" ] && [ "$VARIANT" != "all" ]; then
  log "weights already present at $WEIGHTS"
else
  log "downloading weights ($VARIANT) -> $WEIGHTS"
  # HF_ENDPOINT / HF_HUB_DISABLE_XET pass through from the caller (see the top).
  PATH="$ENV/bin:$PATH" bash "$ROOT/subprojects/repair_pipeline/scripts/download_weights.sh" "$VARIANT" "$WEIGHTS"
fi

# --- record the result ------------------------------------------------------
cat > "$ROOT/configs/envs.local.yaml" <<EOF
# Generated by bootstrap.sh on $(date -Iseconds). Gitignored: absolute paths.
envs:                      # one env serves both stages
  finder: $ENV/bin/python
  repair: $ENV/bin/python
paths:
  artifixer_repo: $ARTIFIXER_DIR
EOF

log "wrote configs/envs.local.yaml"
cat <<EOF

[bootstrap] done.

  next:
    $ENV/bin/python run.py make-config --scene-root /path/to/scene
    $ENV/bin/python run.py doctor --config configs/scenes/generated/<scene>/workflow.yaml
    # wiring check without weights (renders pass through unchanged):
    $ENV/bin/python run.py all --config configs/scenes/generated/<scene>/workflow.yaml --limit 2 --set 'repair.set=[model.backend=passthrough]'

  weights:  $WEIGHTS
  checkout: $ARTIFIXER_DIR
EOF
