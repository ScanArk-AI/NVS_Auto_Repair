#!/bin/bash
# Fetch the HuggingFace repos this pipeline needs.
#
#   bash scripts/download_weights.sh [1.3b|14b|all] [target_dir]
#
# Default target is <nvs_repair>/checkpoints/ArtiFixer, which is where
# configs/method.yaml's relative weight paths point.
#
# Networks that cannot reach huggingface.co: point HF_ENDPOINT at a mirror,
# and if the Xet CAS backend is unreachable (401s), also
#   export HF_HUB_DISABLE_XET=1
# `hf download` also refuses files above MAX_HTTP_DOWNLOAD_SIZE, and
# artifixer-14b.pt is 67.6 GB -- the 14b branch falls back to curl for that one.
set -euo pipefail

VARIANT=${1:-all}
TARGET=${2:-$(cd "$(dirname "$0")/../../.." && pwd)/checkpoints/ArtiFixer}
BASE=${HF_ENDPOINT:-https://huggingface.co}
AF=nvidia/ArtiFixer
WAN13=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
WAN14=Wan-AI/Wan2.1-T2V-14B-Diffusers

log() { echo "[$(date +%H:%M:%S)] $*"; }

fetch() {  # fetch <repo> <local_dir> [hf args…]
  local repo=$1 dir=$2; shift 2
  log "download $repo -> $dir"
  mkdir -p "$dir"
  hf download "$repo" --local-dir "$dir" "$@" || { log "FAILED $repo"; return 1; }
  log "done $(du -sh "$dir" | cut -f1)"
}

fetch_curl() {  # fetch_curl <repo> <file> <local_dir>  -- for >50 GB files
  local repo=$1 file=$2 dir=$3
  mkdir -p "$dir"
  log "curl $repo/$file (above hf download's size cap)"
  curl -L -C - --retry 5 --retry-delay 5 \
    -o "$dir/$file" "$BASE/$repo/resolve/main/$file" || { log "FAILED $repo/$file"; return 1; }
  log "done $file $(du -h "$dir/$file" | cut -f1)"
}

do_13b() {
  fetch "$WAN13" "$TARGET/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
  fetch "$AF"    "$TARGET" "artifixer-1.3b.pt" "config.json" "README.md"
  for file in artifixer-1.3b.pt config.json README.md; do
    test -s "$TARGET/$file" || { log "FAILED missing $TARGET/$file"; return 1; }
  done
}

do_14b() {
  fetch "$WAN14" "$TARGET/Wan-AI/Wan2.1-T2V-14B-Diffusers"
  fetch "$AF"    "$TARGET" --exclude "artifixer-14b.pt"   # everything but the giant fp32 file
  fetch_curl "$AF" artifixer-14b.pt "$TARGET"
  log "converting fp32 -> bf16 (halves the file and removes a per-load cast)"
  python scripts/convert_ckpt_bf16.py \
    "$TARGET/artifixer-14b.pt" "$TARGET/artifixer-14b-bf16.pt"
}

case "$VARIANT" in
  1.3b) do_13b ;;
  14b)  do_14b ;;
  all)  do_13b; do_14b ;;
  *)    echo "usage: $0 [1.3b|14b|all] [target_dir]" >&2; exit 2 ;;
esac

log "weights under $TARGET:"
du -sh "$TARGET"/* 2>/dev/null
