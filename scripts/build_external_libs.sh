#!/usr/bin/env bash
# Provisions C deps the Zig build cannot yet build itself:
#   - OpenSSL (static libcrypto + headers) for MD5/SHA*/RIPEMD160/WHIRLPOOL
# googletest/pcre/expat/argtable3/APR/apr-util are not needed by the Zig port
# (pcre2 is a Zig package; bf is pool-free).
#
# Idempotent: if libcrypto.a already exists for this ABI, nothing is rebuilt.
# Install prefixes are ABI-split so gnu and musl do not overwrite each other:
#   gnu  -> external_lib/lib/openssl
#   musl -> external_lib/lib/openssl-musl
#
# Built with `zig cc -target ${arch}-${os}-${abi}`.
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
if [[ "${ABI}" = "musl" ]]; then
  OPENSSL_PREFIX="${LIB_INSTALL_PREFIX}/openssl-musl"
else
  OPENSSL_PREFIX="${LIB_INSTALL_PREFIX}/openssl"
fi
OPENSSL_MARKER="${OPENSSL_PREFIX}/lib64/libcrypto.a"

CC_FLAGS="zig cc -target ${HOST_TRIPLE}"
AR_FLAGS="zig ar"
RANLIB_FLAGS="zig ranlib"
# Match CMake/linux_build.sh -march=haswell for x86_64.
CFLAGS="-Ofast -march=haswell -mtune=haswell"

mkdir -p "${LIB_INSTALL_SRC}" "${LIB_INSTALL_PREFIX}"

# Already provisioned? Skip entirely (cache hit).
if [[ -f "${OPENSSL_MARKER}" ]]; then
  echo "==> external_lib already provisioned (libcrypto present for ${HOST_TRIPLE})"
  exit 0
fi

echo "==> provisioning external_lib OpenSSL (libcrypto) for ${HOST_TRIPLE}"

rm -rf "${LIB_INSTALL_SRC}/${OPENSSL_SRC}"
(cd "${LIB_INSTALL_SRC}" && {
  [[ -f "${OPENSSL_SRC}.tar.gz" ]] || wget -q https://github.com/openssl/openssl/releases/download/${OPENSSL_SRC}/${OPENSSL_SRC}.tar.gz
  tar -xzf ${OPENSSL_SRC}.tar.gz
})
(cd "${LIB_INSTALL_SRC}/${OPENSSL_SRC}" && \
  AR="${AR_FLAGS}" RANLIB="${RANLIB_FLAGS}" CC="${CC_FLAGS}" \
  CFLAGS="${CFLAGS}" CXXFLAGS="${CFLAGS}" \
  ./Configure -static no-apps --prefix="${OPENSSL_PREFIX}" && \
  make -j"$(nproc)" && make install_sw)

if [[ ! -f "${OPENSSL_MARKER}" ]]; then
  echo "error: OpenSSL install did not produce ${OPENSSL_MARKER}" >&2
  exit 1
fi

echo "==> external_lib provisioning complete (${OPENSSL_MARKER})"
