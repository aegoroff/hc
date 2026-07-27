#!/usr/bin/env bash
# Provisions C deps the Zig build cannot yet build itself:
#   - OpenSSL (static libcrypto + headers) for MD5/SHA*/RIPEMD160/WHIRLPOOL
# googletest/pcre/expat/argtable3/APR/apr-util are not needed by the Zig port
# (pcre2 is a Zig package; bf is pool-free).
#
# Idempotent: if libcrypto.a already exists for this triple, nothing is rebuilt.
# Each arch/os/abi combination gets its own install prefix:
#   external_lib/lib/openssl-${arch}-${os}-${abi}/
# Examples:
#   openssl-x86_64-linux-gnu
#   openssl-x86_64-linux-musl
#   openssl-aarch64-linux-gnu
#   openssl-x86_64-macos-none
#   openssl-aarch64-macos-none
#
# Built with `zig cc -target ${arch}-${os}-${abi}` and an explicit OpenSSL
# Configure target so cross builds (Linux→macOS, x86_64→aarch64) produce the
# correct object format.
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

OPENSSL_SRC=openssl-4.0.0
OPENSSL_PREFIX="${LIB_INSTALL_PREFIX}/openssl-${HOST_TRIPLE}"
# Force a stable libdir across Linux (default lib64) and Darwin (default lib).
OPENSSL_LIBDIR="${OPENSSL_PREFIX}/lib"
OPENSSL_MARKER="${OPENSSL_LIBDIR}/libcrypto.a"

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
# toolchain defaults (OpenSSL asm selects its own baseline).
openssl_cflags() {
  case "${ARCH}" in
    x86_64) echo "-Ofast -march=haswell -mtune=haswell" ;;
    *)      echo "-Ofast" ;;
  esac
}

OPENSSL_TARGET="$(openssl_configure_target)"
CFLAGS="$(openssl_cflags)"

CC_FLAGS="zig cc -target ${HOST_TRIPLE}"
AR_FLAGS="zig ar"
RANLIB_FLAGS="zig ranlib"

mkdir -p "${LIB_INSTALL_SRC}" "${LIB_INSTALL_PREFIX}"

# Already provisioned? Skip entirely (cache hit).
if [[ -f "${OPENSSL_MARKER}" ]]; then
  echo "==> external_lib already provisioned (libcrypto present for ${HOST_TRIPLE})"
  exit 0
fi

echo "==> provisioning external_lib OpenSSL (libcrypto) for ${HOST_TRIPLE}"
echo "    Configure target: ${OPENSSL_TARGET}"
echo "    prefix: ${OPENSSL_PREFIX}"

rm -rf "${LIB_INSTALL_SRC}/${OPENSSL_SRC}"
(cd "${LIB_INSTALL_SRC}" && {
  [[ -f "${OPENSSL_SRC}.tar.gz" ]] || wget -q "https://github.com/openssl/openssl/releases/download/${OPENSSL_SRC}/${OPENSSL_SRC}.tar.gz"
  tar -xzf "${OPENSSL_SRC}.tar.gz"
})
(cd "${LIB_INSTALL_SRC}/${OPENSSL_SRC}" && \
  AR="${AR_FLAGS}" RANLIB="${RANLIB_FLAGS}" CC="${CC_FLAGS}" \
  CFLAGS="${CFLAGS}" CXXFLAGS="${CFLAGS}" \
  ./Configure "${OPENSSL_TARGET}" -static no-apps \
    --prefix="${OPENSSL_PREFIX}" --libdir=lib && \
  make -j"$(nproc)" && make install_sw)

if [[ ! -f "${OPENSSL_MARKER}" ]]; then
  echo "error: OpenSSL install did not produce ${OPENSSL_MARKER}" >&2
  exit 1
fi

echo "==> external_lib provisioning complete (${OPENSSL_MARKER})"
