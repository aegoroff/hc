#!/usr/bin/env bash
# Hybrid build under `zig build`: cross-compiles hc/l2h for a target
# triple, runs unit tests + C# black-box regression (gnu), and produces a
# cpack-equivalent TGZ artefact.
#
# C dependencies the Zig build cannot yet build itself (OpenSSL libcrypto)
# are provisioned by scripts/build_external_libs.sh on first run and
# cached afterwards (mirrors the Windows job's c:/external_lib strategy).
#
# Usage: ./linux_build.sh [abi] [os] [arch]
#   abi:  gnu|musl|none (default gnu; use none for macos)
#   os:   linux|macos   (default linux)
#   arch: x86_64|aarch64 (default x86_64)
set -euo pipefail

ABI=${1:-gnu}
OS=${2:-linux}
ARCH=${3:-x86_64}
VERSION="${HC_VERSION:-6.0.0}"
BUILD_CONF=Release
ZIG_OPTIMIZE=ReleaseFast

TRIPLE="${ARCH}-${OS}-${ABI}"
OUT_DIR="zig-out"
BIN_DIR="bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
# ArchLinux.cs resolves hc via PROJECT_BASE_PATH/build-x86_64-linux-gnu-Release/hc
# when set; default to the repo root so local runs match CI.
export PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-${SCRIPT_DIR}}"

mkdir -p "${BIN_DIR}"

# 1. Provision OpenSSL libcrypto for this triple
#    (external_lib/lib/openssl-${arch}-${os}-${abi}/).
"${SCRIPT_DIR}/scripts/build_external_libs.sh" "${ARCH}" "${OS}" "${ABI}"

# 2. CUDA only for native Linux gnu (host arch). musl / macOS / cross-arch use
#    the GPU stub — nvcc objects and libcudart match the host toolkit.
HOST_ARCH="$(uname -m)"
case "${HOST_ARCH}" in
  arm64) HOST_ARCH=aarch64 ;;
esac
CUDA_FLAG=""
if [[ "${OS}" != "linux" ]] || [[ "${ABI}" != "gnu" ]] || [[ "${ARCH}" != "${HOST_ARCH}" ]]; then
  CUDA_FLAG="-Dcuda=false"
fi

# 3. zig build (cross-target via -Dtarget; pinned glibc 2.17 for gnu in build.zig).
echo "==> zig build -Dtarget=${TRIPLE} -Doptimize=${ZIG_OPTIMIZE} -Dversion=${VERSION} ${CUDA_FLAG}"
zig build \
  -Dtarget="${TRIPLE}" \
  -Doptimize="${ZIG_OPTIMIZE}" \
  -Dversion="${VERSION}" \
  ${CUDA_FLAG}

# Expose artefacts under bin/ for the legacy packaging layout.
cp -v "${OUT_DIR}/bin/hc" "${BIN_DIR}/hc"
cp -v "${OUT_DIR}/bin/l2h" "${BIN_DIR}/l2h" 2>/dev/null || true
cp -v LICENSE.txt "${BIN_DIR}/LICENSE.txt" 2>/dev/null || true

# 4. Unit tests. Musl test binaries are static and run on the gnu host.
if [[ "${ARCH}" = "x86_64" ]] && [[ "${OS}" = "linux" ]]; then
  echo "==> zig build test -Dtarget=${TRIPLE} ${CUDA_FLAG}"
  zig build test -Dtarget="${TRIPLE}" ${CUDA_FLAG} --summary new
  zig build test-l2h -Dtarget="${TRIPLE}" ${CUDA_FLAG} --summary new
fi

# 5. C# black-box regression (develop parity). ArchLinux.cs looks for
#    ${PROJECT_BASE_PATH}/build-x86_64-linux-gnu-Release/hc — point that at the
#    zig-built binary. Only gnu/x86_64/linux: musl is an artefact, not the test host.
if [[ "${ARCH}" = "x86_64" ]] && [[ "${OS}" = "linux" ]] && [[ "${ABI}" = "gnu" ]]; then
  COMPAT_DIR="build-x86_64-linux-gnu-${BUILD_CONF}"
  mkdir -p "${COMPAT_DIR}"
  ln -sfn "${SCRIPT_DIR}/${OUT_DIR}/bin/hc" "${COMPAT_DIR}/hc"
  ln -sfn "${SCRIPT_DIR}/${OUT_DIR}/bin/l2h" "${COMPAT_DIR}/l2h"
  echo "==> dotnet test -c ${BUILD_CONF} src/_tst.net  (hc -> ${COMPAT_DIR}/hc -> ${OUT_DIR}/bin/hc)"
  dotnet test -c "${BUILD_CONF}" src/_tst.net/_tst.net.csproj
fi

# 6. TGZ packaging (replaces cpack TGZ: hc + l2h + LICENSE per triple).
PKG_NAME="hc-${VERSION}-${ARCH}-unknown-${OS}-${ABI}"
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/${PKG_NAME}"
cp -v "${OUT_DIR}/bin/hc" "${STAGE}/${PKG_NAME}/"
[[ -f "${OUT_DIR}/bin/l2h" ]] && cp -v "${OUT_DIR}/bin/l2h" "${STAGE}/${PKG_NAME}/"
[[ -f LICENSE.txt ]] && cp -v LICENSE.txt "${STAGE}/${PKG_NAME}/"
tar -C "${STAGE}" -czvf "${BIN_DIR}/${PKG_NAME}.tar.gz" "${PKG_NAME}"
echo "Package: ${BIN_DIR}/${PKG_NAME}.tar.gz"
