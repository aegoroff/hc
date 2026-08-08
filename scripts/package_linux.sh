#!/usr/bin/env bash
# Build .deb and .rpm from a linux-gnu release tarball using nfpm.
#
# Usage: scripts/package_linux.sh <version> <cpu_arch> <tarball> <outdir>
#   cpu_arch: x86_64 | aarch64
#
# Requires network on first run unless `nfpm` is already on PATH.
# Override NFPM_VER to pin the nfpm release (default 2.47.0).
set -euo pipefail

VERSION="${1:?version required}"
CPU_ARCH="${2:?cpu arch required (x86_64|aarch64)}"
TARBALL="${3:?tarball path required}"
OUT_DIR="${4:?output directory required}"
NFPM_VER="${NFPM_VER:-2.47.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${REPO_ROOT}/nfpm.yaml"

case "${CPU_ARCH}" in
  x86_64)
    DEB_ARCH=amd64
    RPM_ARCH=x86_64
    ;;
  aarch64)
    DEB_ARCH=arm64
    RPM_ARCH=aarch64
    ;;
  *)
    echo "unsupported cpu arch: ${CPU_ARCH} (want x86_64 or aarch64)" >&2
    exit 1
    ;;
esac

if [[ ! -f "${TARBALL}" ]]; then
  echo "tarball not found: ${TARBALL}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "nfpm config not found: ${CONFIG}" >&2
  exit 1
fi

resolve_nfpm() {
  if command -v nfpm >/dev/null 2>&1; then
    command -v nfpm
    return
  fi
  local host_arch
  host_arch="$(uname -m)"
  case "${host_arch}" in
    x86_64 | amd64) host_arch=x86_64 ;;
    aarch64 | arm64) host_arch=arm64 ;;
    *)
      echo "cannot download nfpm for host arch ${host_arch}" >&2
      exit 1
      ;;
  esac
  local cache_dir="${REPO_ROOT}/.zig-cache/nfpm-${NFPM_VER}-${host_arch}"
  local bin="${cache_dir}/nfpm"
  if [[ ! -x "${bin}" ]]; then
    mkdir -p "${cache_dir}"
    local url="https://github.com/goreleaser/nfpm/releases/download/v${NFPM_VER}/nfpm_${NFPM_VER}_Linux_${host_arch}.tar.gz"
    echo "==> downloading nfpm ${NFPM_VER} (${host_arch})" >&2
    curl -fsSL "${url}" | tar --no-same-owner -xz -C "${cache_dir}" nfpm
    chmod +x "${bin}"
  fi
  printf '%s\n' "${bin}"
}

NFPM_BIN="$(resolve_nfpm)"

STAGE="${REPO_ROOT}/pkg-staging"
EXTRACT="$(mktemp -d)"
rm -rf "${STAGE}"
mkdir -p "${STAGE}" "${OUT_DIR}"
trap 'rm -rf "${STAGE}" "${EXTRACT}"' EXIT

# Flat layout from linux_build.sh: hc, l2h, LICENSE.txt at tarball root.
tar --no-same-owner -xzf "${TARBALL}" -C "${EXTRACT}"

HC_BIN="${EXTRACT}/hc"
L2H_BIN="${EXTRACT}/l2h"
LICENSE="${EXTRACT}/LICENSE.txt"
if [[ ! -f "${HC_BIN}" || ! -f "${L2H_BIN}" || ! -f "${LICENSE}" ]]; then
  echo "tarball missing hc, l2h, or LICENSE.txt: ${TARBALL}" >&2
  find "${EXTRACT}" -type f >&2
  exit 1
fi

cp -f "${HC_BIN}" "${STAGE}/hc"
cp -f "${L2H_BIN}" "${STAGE}/l2h"
cp -f "${LICENSE}" "${STAGE}/LICENSE.txt"
chmod 755 "${STAGE}/hc" "${STAGE}/l2h"

package_one() {
  local packager="$1"
  local arch="$2"
  local target="$3"
  echo "==> nfpm package ${target}"
  (
    cd "${REPO_ROOT}"
    export VERSION ARCH="${arch}"
    "${NFPM_BIN}" package --config "${CONFIG}" --packager "${packager}" --target "${target}"
  )
  echo "Package: ${target}"
}

package_one deb "${DEB_ARCH}" "${OUT_DIR}/hash-calculator_${VERSION}_${DEB_ARCH}.deb"
package_one rpm "${RPM_ARCH}" "${OUT_DIR}/hash-calculator-${VERSION}-1.${RPM_ARCH}.rpm"
