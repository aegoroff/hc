#!/usr/bin/env bash
# Hybrid build under `zig build`: cross-compiles hc/l2h for a target
# triple, runs unit tests + pytest black-box regression (gnu), and produces a
# TGZ artefact with both binaries (hc + l2h + LICENSE).
#
# C dependencies the Zig build cannot yet build itself (OpenSSL libcrypto)
# are provisioned by scripts/build_external_libs.sh into workspace
# external_lib/. On CI (or when HC_EXTERNAL_LIB_CACHE is set) a persistent
# agent cache is seeded/written so checkout cleans do not force a full
# OpenSSL rebuild (Windows: C:\external_lib / HC_EXTERNAL_LIB_CACHE).
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
# pytest runner resolves hc via PROJECT_BASE_PATH/build-${ARCH}-linux-${ABI}-Release/hc
# when set; default to the repo root so local runs match CI.
export PROJECT_BASE_PATH="${PROJECT_BASE_PATH:-${SCRIPT_DIR}}"
export HC_TEST_ABI="${HC_TEST_ABI:-${ABI}}"
export HC_TEST_ARCH="${HC_TEST_ARCH:-${ARCH}}"

mkdir -p "${BIN_DIR}"
TEST_RESULTS_DIR="${SCRIPT_DIR}/test-results"
mkdir -p "${TEST_RESULTS_DIR}"

# Append zig --summary output to the GitHub Actions job summary when running in CI.
append_zig_summary() {
  local title="$1"
  local log_file="$2"
  [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] || return 0
  {
    echo "## ${title}"
    echo
    echo '```'
    if grep -q '^Build Summary:' "${log_file}" 2>/dev/null; then
      awk '/^Build Summary:/{found=1} found' "${log_file}" | tail -n 80
    else
      tail -n 40 "${log_file}"
    fi
    echo '```'
    echo
  } >> "${GITHUB_STEP_SUMMARY}"
}

# Run `zig build <args> --summary new`, tee to test-results/, publish summary.
run_zig_tests() {
  local log_name="$1"
  shift
  local log_file="${TEST_RESULTS_DIR}/${log_name}.log"
  echo "==> zig build $* --summary new"
  set +e
  zig build "$@" --summary new 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  append_zig_summary "Zig: ${log_name} (${TRIPLE})" "${log_file}"
  return "${status}"
}

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

# 4. Unit tests — native Linux only (x86_64 or aarch64 host). Musl test
#    binaries are static and run on the gnu host. `test` already pulls in l2h
#    (`test-l2h` remains for local focused runs). Logs + Job Summary in CI.
if [[ "${OS}" = "linux" ]] && [[ "${ARCH}" = "${HOST_ARCH}" ]]; then
  zig_test_args=(test "-Dtarget=${TRIPLE}" "-Doptimize=${ZIG_OPTIMIZE}")
  if [[ -n "${CUDA_FLAG}" ]]; then
    zig_test_args+=("${CUDA_FLAG}")
  fi
  run_zig_tests "zig-test" "${zig_test_args[@]}"
fi

# 5. pytest black-box regression. runner.py looks for
#    ${PROJECT_BASE_PATH}/build-${ARCH}-linux-${ABI}-Release/hc — point that at
#    the zig-built binary. Native Linux only (musl artefact is runnable on the
#    gnu host). JUnit XML lands in test-results/ for dorny/test-reporter in CI.
if [[ "${OS}" = "linux" ]] && [[ "${ARCH}" = "${HOST_ARCH}" ]]; then
  COMPAT_DIR="build-${ARCH}-linux-${ABI}-${BUILD_CONF}"
  mkdir -p "${COMPAT_DIR}"
  ln -sfn "${SCRIPT_DIR}/${OUT_DIR}/bin/hc" "${COMPAT_DIR}/hc"
  ln -sfn "${SCRIPT_DIR}/${OUT_DIR}/bin/l2h" "${COMPAT_DIR}/l2h"
  echo "==> pytest src/_tst.py  (hc -> ${COMPAT_DIR}/hc -> ${OUT_DIR}/bin/hc)"
  PY=""
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  elif command -v python >/dev/null 2>&1; then
    PY=python
  else
    echo "error: python3 not found on PATH (needed for src/_tst.py black-box)" >&2
    exit 1
  fi
  if [[ ! -d "${SCRIPT_DIR}/.venv-tst" ]]; then
    "${PY}" -m venv "${SCRIPT_DIR}/.venv-tst"
  fi
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv-tst/bin/activate"
  python -m pip install -q -r src/_tst.py/requirements.txt
  export HC_TEST_DIR="${TEST_RESULTS_DIR}/_tst.py-workdir"
  export HC_TEST_ABI="${ABI}"
  export HC_TEST_ARCH="${ARCH}"
  # Parallel via xdist; file → group "file", crack → group "crack" (GPU VRAM),
  # each group serial on one worker (--dist loadgroup).
  python -m pytest src/_tst.py \
    -n auto --dist loadgroup \
    --junitxml="${TEST_RESULTS_DIR}/pytest-linux-${ABI}.xml"
fi

# 6. TGZ packaging: one archive with hc + l2h + LICENSE.
# Flat layout (binaries + LICENSE at archive root) matches historical releases
# and AUR/scoop expectations; both tools ship in the same package.
PKG_NAME="hc-${VERSION}-${ARCH}-unknown-${OS}-${ABI}"
STAGE=$(mktemp -d)
trap 'rm -rf "${STAGE}"' EXIT
members=()
cp -v "${OUT_DIR}/bin/hc" "${STAGE}/"
members+=(hc)
if [[ -f "${OUT_DIR}/bin/l2h" ]]; then
  cp -v "${OUT_DIR}/bin/l2h" "${STAGE}/"
  members+=(l2h)
fi
if [[ -f LICENSE.txt ]]; then
  cp -v LICENSE.txt "${STAGE}/"
  members+=(LICENSE.txt)
fi
tar -C "${STAGE}" -czvf "${BIN_DIR}/${PKG_NAME}.tar.gz" "${members[@]}"
echo "Package: ${BIN_DIR}/${PKG_NAME}.tar.gz"

# 7. Packages via nfpm: gnu → .deb/.rpm; musl → .apk (Alpine).
if [[ "${OS}" = "linux" && "${ABI}" = "gnu" ]]; then
  chmod +x "${SCRIPT_DIR}/scripts/package_linux.sh"
  "${SCRIPT_DIR}/scripts/package_linux.sh" \
    "${VERSION}" "${ARCH}" \
    "${BIN_DIR}/${PKG_NAME}.tar.gz" \
    "${BIN_DIR}" \
    "deb,rpm"
elif [[ "${OS}" = "linux" && "${ABI}" = "musl" ]]; then
  chmod +x "${SCRIPT_DIR}/scripts/package_linux.sh"
  "${SCRIPT_DIR}/scripts/package_linux.sh" \
    "${VERSION}" "${ARCH}" \
    "${BIN_DIR}/${PKG_NAME}.tar.gz" \
    "${BIN_DIR}" \
    "apk"
fi
