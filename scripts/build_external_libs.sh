#!/usr/bin/env bash
# Provisions C deps the Zig build cannot yet build itself:
#   - OpenSSL (static libcrypto + headers) for MD5/SHA*/RIPEMD160/WHIRLPOOL
# googletest/pcre/expat/argtable3/APR/apr-util are not needed by the Zig port
# (pcre2 is a Zig package; bf is pool-free).
#
# Workspace install (what build.zig links):
#   external_lib/lib/openssl-${arch}-${os}-${abi}/
#
# Persistent agent cache (CI / HC_EXTERNAL_LIB_CACHE only; local stays in
# workspace external_lib/ with no write-back):
#   ${HC_EXTERNAL_LIB_CACHE:-/opt/actions-runner/hc-external-lib}/
#     openssl-${ver}-${arch}-${os}-${abi}-{asm|noasm}/
# On k3s self-hosted runners /opt/actions-runner is a PVC that survives
# pod redeploys; fall back to XDG/HOME cache only when that path is absent.
# Rebuild only when the cached tree is missing or its version stamp differs.
#
# Built with `zig cc -target ${arch}-${os}-${abi}` and an explicit OpenSSL
# Configure target so cross builds (Linux→macOS, x86_64→aarch64) produce the
# correct object format.
#
# Platform asm (AES-NI / ARMv8 Crypto Extensions, …) is enabled only for a
# native build (target arch+os == host). Cross builds pass `no-asm` so the
# same script works on both without assembler failures.
#
# Usage: ./scripts/build_external_libs.sh [arch] [os] [abi]
set -euo pipefail

ARCH=${1:-x86_64}
OS=${2:-linux}
ABI=${3:-gnu}
HOST_TRIPLE="${ARCH}-${OS}-${ABI}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_INSTALL_SRC="${ROOT}/external_lib/src"
LIB_INSTALL_PREFIX="${ROOT}/external_lib/lib"

OPENSSL_VER=4.0.1
OPENSSL_SRC="openssl-${OPENSSL_VER}"
OPENSSL_PREFIX="${LIB_INSTALL_PREFIX}/openssl-${HOST_TRIPLE}"
# Force a stable libdir across Linux (default lib64) and Darwin (default lib).
OPENSSL_LIBDIR="${OPENSSL_PREFIX}/lib"
OPENSSL_MARKER="${OPENSSL_LIBDIR}/libcrypto.a"
OPENSSL_VERSION_STAMP="${OPENSSL_PREFIX}/.hc-openssl-version"

# OpenSSL Configure target for the requested arch/OS (not host auto-detect).
openssl_configure_target() {
  case "${ARCH}-${OS}" in
    x86_64-linux)   echo "linux-x86_64" ;;
    aarch64-linux)  echo "linux-aarch64" ;;
    arm-linux)      echo "linux-armv4" ;;
    x86_64-macos)   echo "darwin64-x86_64" ;;
    aarch64-macos)  echo "darwin64-arm64" ;;
    *)
      echo "error: unsupported OpenSSL target ${ARCH}-${OS} (abi=${ABI})" >&2
      echo "supported: x86_64|aarch64|arm × linux, x86_64|aarch64 × macos" >&2
      return 1
      ;;
  esac
}

# Match CMake/linux_build.sh -march=haswell for x86_64; leave ARM to the
# toolchain defaults (OpenSSL perlasm selects ARMv8 crypto itself).
openssl_cflags() {
  case "${ARCH}" in
    x86_64) echo "-Ofast -march=haswell -mtune=haswell" ;;
    *)      echo "-Ofast" ;;
  esac
}

# True when Configure target matches this machine (arch + OS).
openssl_is_native_build() {
  local host_arch host_os
  host_arch="$(uname -m)"
  case "${host_arch}" in
    arm64) host_arch=aarch64 ;;
  esac
  case "$(uname -s)" in
    Linux)  host_os=linux ;;
    Darwin) host_os=macos ;;
    *)      host_os="" ;;
  esac
  [[ -n "${host_os}" ]] && [[ "${ARCH}" = "${host_arch}" ]] && [[ "${OS}" = "${host_os}" ]]
}

# After Configure on a native build: refuse a no-asm tree.
assert_openssl_asm_enabled() {
  local src_dir="$1"
  (
    cd "${src_dir}"
    perl -I. -Mconfigdata -e '
      die "OpenSSL asm disabled on native build; digests need platform asm\n"
        if $disabled{asm};
      my $arch = $target{asm_arch} // "";
      die "OpenSSL Configure left asm_arch empty\n" if $arch eq "";
      print "OpenSSL asm enabled (asm_arch=${arch})\n";
    '
  )
}

# Copy a provisioned OpenSSL tree into the workspace (or cache).
seed_openssl_tree() {
  local from="$1"
  local to="$2"
  rm -rf "${to}"
  mkdir -p "$(dirname "${to}")"
  cp -a "${from}" "${to}"
}

openssl_workspace_ready() {
  [[ -f "${OPENSSL_MARKER}" ]] && [[ -f "${OPENSSL_VERSION_STAMP}" ]] &&
    [[ "$(cat "${OPENSSL_VERSION_STAMP}")" = "${OPENSSL_VER}" ]]
}

OPENSSL_TARGET="$(openssl_configure_target)"
CFLAGS="$(openssl_cflags)"

# Native → platform asm; cross → portable C (no-asm).
OPENSSL_ASM_ARGS=()
if openssl_is_native_build; then
  OPENSSL_NATIVE=1
  OPENSSL_ASM_TAG=asm
else
  OPENSSL_NATIVE=0
  OPENSSL_ASM_TAG=noasm
  OPENSSL_ASM_ARGS+=(no-asm)
fi

# Persistent agent cache: only when HC_EXTERNAL_LIB_CACHE is set, or under CI
# (GITHUB_ACTIONS/CI). Local developer builds stay workspace-only under
# external_lib/ — no write to a host cache.
USE_AGENT_CACHE=0
CACHE_ROOT=""
if [[ -n "${HC_EXTERNAL_LIB_CACHE:-}" ]]; then
  USE_AGENT_CACHE=1
  CACHE_ROOT="${HC_EXTERNAL_LIB_CACHE}"
elif [[ "${GITHUB_ACTIONS:-}" = "true" ]] || [[ "${CI:-}" = "true" ]]; then
  USE_AGENT_CACHE=1
  # Prefer the runner PVC (/opt/actions-runner) so the OpenSSL tree survives
  # k3s redeploys; ephemeral HOME/.cache is wiped with the pod.
  if [[ -d /opt/actions-runner ]]; then
    CACHE_ROOT="/opt/actions-runner/hc-external-lib"
  elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    CACHE_ROOT="${XDG_CACHE_HOME}/hc-external-lib"
  else
    CACHE_ROOT="${HOME}/.cache/hc-external-lib"
  fi
fi

CACHE_KEY="openssl-${OPENSSL_VER}-${HOST_TRIPLE}-${OPENSSL_ASM_TAG}"
CACHE_DIR="${CACHE_ROOT}/${CACHE_KEY}"
CACHE_MARKER="${CACHE_DIR}/lib/libcrypto.a"
CACHE_VERSION_STAMP="${CACHE_DIR}/.hc-openssl-version"
CACHE_SRC_DIR="${CACHE_ROOT}/src"
CACHE_TARBALL="${CACHE_SRC_DIR}/${OPENSSL_SRC}.tar.gz"

CC_FLAGS="zig cc -target ${HOST_TRIPLE}"
AR_FLAGS="zig ar"
RANLIB_FLAGS="zig ranlib"

mkdir -p "${LIB_INSTALL_SRC}" "${LIB_INSTALL_PREFIX}"

# 1. Workspace already has the right version → done.
if openssl_workspace_ready; then
  echo "==> external_lib already provisioned (OpenSSL ${OPENSSL_VER} for ${HOST_TRIPLE})"
  exit 0
fi

# 2. Persistent cache hit (CI / HC_EXTERNAL_LIB_CACHE) → seed workspace.
if [[ "${USE_AGENT_CACHE}" -eq 1 ]] &&
  [[ -f "${CACHE_MARKER}" ]] && [[ -f "${CACHE_VERSION_STAMP}" ]] &&
  [[ "$(cat "${CACHE_VERSION_STAMP}")" = "${OPENSSL_VER}" ]]; then
  echo "==> seeding OpenSSL ${OPENSSL_VER} from cache ${CACHE_DIR} -> ${OPENSSL_PREFIX}"
  seed_openssl_tree "${CACHE_DIR}" "${OPENSSL_PREFIX}"
  if openssl_workspace_ready; then
    echo "==> external_lib OpenSSL ready (${OPENSSL_MARKER})"
    exit 0
  fi
  echo "warning: cache seed incomplete; rebuilding" >&2
fi

# Stale workspace tree (wrong/missing version) → clear before rebuild.
rm -rf "${OPENSSL_PREFIX}"

echo "==> provisioning external_lib OpenSSL (libcrypto) for ${HOST_TRIPLE}"
echo "    version: ${OPENSSL_VER}"
echo "    Configure target: ${OPENSSL_TARGET}"
echo "    prefix: ${OPENSSL_PREFIX}"
if [[ "${USE_AGENT_CACHE}" -eq 1 ]]; then
  echo "    cache: ${CACHE_DIR}"
else
  echo "    cache: disabled (local workspace-only)"
fi
if [[ "${OPENSSL_NATIVE}" -eq 1 ]]; then
  echo "    asm: enabled (native ${ARCH}-${OS})"
else
  echo "    asm: disabled via no-asm (cross-compile)"
fi

# Tarball: prefer agent cache when enabled, else workspace external_lib/src.
TARBALL_DEST="${LIB_INSTALL_SRC}/${OPENSSL_SRC}.tar.gz"
if [[ "${USE_AGENT_CACHE}" -eq 1 ]]; then
  mkdir -p "${CACHE_SRC_DIR}"
  if [[ ! -f "${CACHE_TARBALL}" ]]; then
    echo "==> downloading ${OPENSSL_SRC}.tar.gz"
    wget -q -O "${CACHE_TARBALL}" \
      "https://github.com/openssl/openssl/releases/download/${OPENSSL_SRC}/${OPENSSL_SRC}.tar.gz"
  fi
  cp -f "${CACHE_TARBALL}" "${TARBALL_DEST}"
elif [[ ! -f "${TARBALL_DEST}" ]]; then
  echo "==> downloading ${OPENSSL_SRC}.tar.gz"
  wget -q -O "${TARBALL_DEST}" \
    "https://github.com/openssl/openssl/releases/download/${OPENSSL_SRC}/${OPENSSL_SRC}.tar.gz"
fi

rm -rf "${LIB_INSTALL_SRC}/${OPENSSL_SRC}"
(cd "${LIB_INSTALL_SRC}" && tar -xzf "${OPENSSL_SRC}.tar.gz")

(cd "${LIB_INSTALL_SRC}/${OPENSSL_SRC}" && \
  AR="${AR_FLAGS}" RANLIB="${RANLIB_FLAGS}" CC="${CC_FLAGS}" \
  CFLAGS="${CFLAGS}" CXXFLAGS="${CFLAGS}" \
  ./Configure "${OPENSSL_TARGET}" -static no-apps \
    "${OPENSSL_ASM_ARGS[@]}" \
    --prefix="${OPENSSL_PREFIX}" --libdir=lib && \
  if [[ "${OPENSSL_NATIVE}" -eq 1 ]]; then
    assert_openssl_asm_enabled "${LIB_INSTALL_SRC}/${OPENSSL_SRC}"
  fi && \
  make -j"$(nproc)" && make install_sw)

if [[ ! -f "${OPENSSL_MARKER}" ]]; then
  echo "error: OpenSSL install did not produce ${OPENSSL_MARKER}" >&2
  exit 1
fi
printf '%s\n' "${OPENSSL_VER}" > "${OPENSSL_VERSION_STAMP}"

# Write-back only on CI / when HC_EXTERNAL_LIB_CACHE is set.
if [[ "${USE_AGENT_CACHE}" -eq 1 ]]; then
  echo "==> caching OpenSSL ${OPENSSL_VER} -> ${CACHE_DIR}"
  seed_openssl_tree "${OPENSSL_PREFIX}" "${CACHE_DIR}"
  printf '%s\n' "${OPENSSL_VER}" > "${CACHE_VERSION_STAMP}"
fi

echo "==> external_lib provisioning complete (${OPENSSL_MARKER})"
