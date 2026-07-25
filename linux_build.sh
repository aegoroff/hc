#!/usr/bin/env bash
# Hybrid build: Zig app-layer (hc, l2h) + optional legacy CMake packaging.
# Usage: ./linux_build.sh [abi] [os] [arch]
#   abi: gnu|musl (default gnu)
#   os:  linux (default)
#   arch: x86_64 (default)
set -euo pipefail

BUILD_CONF=Release
ABI=${1:-gnu}
OS=${2:-linux}
ARCH=${3:-x86_64}
VERSION="${HC_VERSION:-5.5.0}"
ZIG_OPTIMIZE=ReleaseFast

TRIPLE="${ARCH}-${OS}-${ABI}"
OUT_DIR="zig-out"
BIN_DIR="bin"
mkdir -p "${BIN_DIR}"

# Single binary: CUDA is linked automatically when nvcc is on the machine;
# hashes without GPU / no device fall back to CPU at runtime.
echo "==> zig build -Dtarget=${TRIPLE} -Doptimize=${ZIG_OPTIMIZE} -Dversion=${VERSION}"
zig build \
  -Dtarget="${TRIPLE}" \
  -Doptimize="${ZIG_OPTIMIZE}" \
  -Dversion="${VERSION}"

# Install artefacts into bin/ for packaging (cpack-compatible layout)
cp -v "${OUT_DIR}/bin/hc" "${BIN_DIR}/hc"
cp -v "${OUT_DIR}/bin/l2h" "${BIN_DIR}/l2h" 2>/dev/null || true
cp -v LICENSE.txt "${BIN_DIR}/LICENSE.txt" 2>/dev/null || true

# Unit tests
echo "==> zig build test"
zig build test -Dtarget="${TRIPLE}"

# C# black-box regression against the freshly built hc
if [[ "${ARCH}" = "x86_64" ]] && [[ "${OS}" = "linux" ]]; then
  export PATH="$(pwd)/${OUT_DIR}/bin:${PATH}"
  if command -v dotnet >/dev/null 2>&1; then
    echo "==> dotnet test"
    # Point tests at zig-out/bin/hc when the suite supports HC_BIN;
    # otherwise rely on PATH.
    HC_BIN="$(pwd)/${OUT_DIR}/bin/hc" dotnet test -c "${BUILD_CONF}" src || \
      PATH="$(pwd)/${OUT_DIR}/bin:$PATH" dotnet test -c "${BUILD_CONF}" src
  else
    echo "dotnet not found; skipping C# tests"
  fi
fi

# TGZ packaging (replaces cpack for the Zig binary)
PKG_NAME="hc-${VERSION}-x86_64-unknown-linux-${ABI}"
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/${PKG_NAME}"
cp -v "${OUT_DIR}/bin/hc" "${STAGE}/${PKG_NAME}/"
[[ -f "${OUT_DIR}/bin/l2h" ]] && cp -v "${OUT_DIR}/bin/l2h" "${STAGE}/${PKG_NAME}/"
[[ -f LICENSE.txt ]] && cp -v LICENSE.txt "${STAGE}/${PKG_NAME}/"
tar -C "${STAGE}" -czvf "${BIN_DIR}/${PKG_NAME}.tar.gz" "${PKG_NAME}"
echo "Package: ${BIN_DIR}/${PKG_NAME}.tar.gz"
