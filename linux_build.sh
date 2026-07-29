#!/usr/bin/env bash
# Hybrid build under `zig build`: cross-compiles hc/l2h for a target
# triple, runs unit tests + C# black-box regression (gnu), and produces
# separate TGZ artefacts per binary (hc-*.tar.gz and l2h-*.tar.gz).
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

# 4. Unit tests. Musl test binaries are static and run on the gnu host.
#    `test` already pulls in l2h; `test-l2h` is run explicitly for a clear
#    failure surface (same as windows_build.ps1). Logs + Job Summary in CI.
if [[ "${ARCH}" = "x86_64" ]] && [[ "${OS}" = "linux" ]]; then
  zig_test_args=(test "-Dtarget=${TRIPLE}")
  zig_l2h_args=(test-l2h "-Dtarget=${TRIPLE}")
  if [[ -n "${CUDA_FLAG}" ]]; then
    zig_test_args+=("${CUDA_FLAG}")
    zig_l2h_args+=("${CUDA_FLAG}")
  fi
  run_zig_tests "zig-test" "${zig_test_args[@]}"
  run_zig_tests "zig-test-l2h" "${zig_l2h_args[@]}"
fi

# 5. C# black-box regression (develop parity). ArchLinux.cs looks for
#    ${PROJECT_BASE_PATH}/build-x86_64-linux-gnu-Release/hc — point that at the
#    zig-built binary. Only gnu/x86_64/linux: musl is an artefact, not the test host.
#    TRX lands in test-results/ for dorny/test-reporter in CI.
if [[ "${ARCH}" = "x86_64" ]] && [[ "${OS}" = "linux" ]] && [[ "${ABI}" = "gnu" ]]; then
  COMPAT_DIR="build-x86_64-linux-gnu-${BUILD_CONF}"
  mkdir -p "${COMPAT_DIR}"
  ln -sfn "${SCRIPT_DIR}/${OUT_DIR}/bin/hc" "${COMPAT_DIR}/hc"
  ln -sfn "${SCRIPT_DIR}/${OUT_DIR}/bin/l2h" "${COMPAT_DIR}/l2h"
  echo "==> dotnet test -c ${BUILD_CONF} src/_tst.net  (hc -> ${COMPAT_DIR}/hc -> ${OUT_DIR}/bin/hc)"
  dotnet test -c "${BUILD_CONF}" src/_tst.net/_tst.net.csproj \
    --logger "trx;LogFileName=csharp-linux-gnu.trx" \
    --results-directory "${TEST_RESULTS_DIR}"
fi

# 6. TGZ packaging: one archive per binary (hc and l2h separately).
# Flat layout (binary + LICENSE at archive root) matches historical releases
# and AUR/scoop expectations for hc. l2h is GitHub Releases only — not AUR/scoop.
pack_tgz() {
  local pkg_name="$1"
  local bin_path="$2"
  local stage members=()
  stage=$(mktemp -d)
  cp -v "${bin_path}" "${stage}/"
  members+=("$(basename "${bin_path}")")
  if [[ -f LICENSE.txt ]]; then
    cp -v LICENSE.txt "${stage}/"
    members+=(LICENSE.txt)
  fi
  tar -C "${stage}" -czvf "${BIN_DIR}/${pkg_name}.tar.gz" "${members[@]}"
  rm -rf "${stage}"
  echo "Package: ${BIN_DIR}/${pkg_name}.tar.gz"
}

pack_tgz "hc-${VERSION}-${ARCH}-unknown-${OS}-${ABI}" "${OUT_DIR}/bin/hc"
if [[ -f "${OUT_DIR}/bin/l2h" ]]; then
  pack_tgz "l2h-${VERSION}-${ARCH}-unknown-${OS}-${ABI}" "${OUT_DIR}/bin/l2h"
fi
