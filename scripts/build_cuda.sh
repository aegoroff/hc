#!/usr/bin/env bash
# Build CUDA sources into a static library for linking from Zig.
# Usage: scripts/build_cuda.sh [out_dir] [arch]
# Default out_dir: zig-out/cuda, arch: sm_75
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$ROOT/zig-out/cuda}"
ARCH="${2:-sm_75}"
INC="$ROOT/src/zig/cuda_include"
SRC="$ROOT/src/hc"
NVCC_BIN="${NVCC:-nvcc}"

find_nvcc() {
  if [[ "$NVCC_BIN" != "nvcc" && -x "$NVCC_BIN" ]]; then
    echo "$NVCC_BIN"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi
  local candidate
  for candidate in \
      "${CUDA_PATH:+$CUDA_PATH/bin/nvcc}" \
      "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}" \
      /opt/cuda/bin/nvcc \
      /usr/local/cuda/bin/nvcc; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
    # Windows (Git Bash): accept nvcc.exe
    if [[ -n "$candidate" && -x "${candidate}.exe" ]]; then
      echo "${candidate}.exe"
      return 0
    fi
  done
  return 1
}

if ! NVCC="$(find_nvcc)"; then
  echo "nvcc not found (set CUDA_PATH / CUDA_HOME or put nvcc on PATH)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/obj"
OBJS=()
for cu in "$SRC"/*.cu; do
  base="$(basename "$cu" .cu)"
  obj="$OUT_DIR/obj/$base.o"
  echo "nvcc $base.cu -> $obj"
  "$NVCC" -c -o "$obj" "$cu" \
    -I"$INC" \
    -arch="$ARCH" \
    -std=c++17 \
    -O2 \
    --compiler-options -fPIC
  OBJS+=("$obj")
done

LIB="$OUT_DIR/libhc-cuda.a"
rm -f "$LIB"
ar rcs "$LIB" "${OBJS[@]}"
echo "Wrote $LIB"
