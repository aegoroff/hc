#!/usr/bin/env bash
# Provisions the C dependencies the Zig build (build.zig) cannot yet build itself:
#   - APR      : libapr-1.a is linked as an object file into hc/bf (hard dependency)
#   - OpenSSL  : only headers are consumed (whrlpool compiled from vendored src)
# googletest/pcre/expat/argtable3/apr-util are NOT needed any more (pcre2 is a
# Zig package; the rest were dropped by the Zig port).
#
# Mirrors the Windows job's c:/external_lib caching strategy: idempotent — if the
# apr archive and openssl headers already exist, nothing is rebuilt. On a cached
# self-hosted runner this is a no-op.
#
# Built for the host triple (x86_64-linux-gnu) with `zig cc`. The musl
# cross-target links the same archive (symbol-compatible; verified) — a clean
# per-ABI rebuild is tracked as a TODO.
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

APR_SRC=apr-1.7.6
OPENSSL_SRC=openssl-4.0.0
APR_PREFIX="${LIB_INSTALL_PREFIX}/apr"
OPENSSL_PREFIX="${LIB_INSTALL_PREFIX}/openssl"

CC_FLAGS="zig cc -target ${HOST_TRIPLE}"
AR_FLAGS="zig ar"
RANLIB_FLAGS="zig ranlib"
# Match CMake/linux_build.sh -march=haswell for x86_64.
CFLAGS="-Ofast -march=haswell -mtune=haswell"

mkdir -p "${LIB_INSTALL_SRC}" "${LIB_INSTALL_PREFIX}"

# Already provisioned? Skip entirely (cache hit).
if [[ -f "${APR_PREFIX}/lib/libapr-1.a" ]] && [[ -f "${OPENSSL_PREFIX}/include/openssl/whrlpool.h" ]]; then
  echo "==> external_lib already provisioned (apr + openssl headers present)"
  exit 0
fi

echo "==> provisioning external_lib for ${HOST_TRIPLE}"

# ---- OpenSSL (headers consumed by crypto lib + vendored whirlpool) ----
if [[ ! -f "${OPENSSL_PREFIX}/include/openssl/whrlpool.h" ]]; then
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
fi

# ---- APR (libapr-1.a linked into hc/bf) ----
if [[ ! -f "${APR_PREFIX}/lib/libapr-1.a" ]]; then
  rm -rf "${LIB_INSTALL_SRC}/${APR_SRC}"
  (cd "${LIB_INSTALL_SRC}" && {
    [[ -f "${APR_SRC}.tar.gz" ]] || wget -q https://dlcdn.apache.org/apr/${APR_SRC}.tar.gz
    tar -xzf ${APR_SRC}.tar.gz
  })
  (cd "${LIB_INSTALL_SRC}/${APR_SRC}" && \
    AR="${AR_FLAGS}" RANLIB="${RANLIB_FLAGS}" CC="${CC_FLAGS}" \
    CFLAGS="${CFLAGS} -Wno-implicit-function-declaration -Wno-int-conversion" \
    ./configure \
      ac_cv_file__dev_zero=yes apr_cv_process_shared_works=yes \
      apr_cv_mutex_robust_shared=yes apr_cv_tcp_nodelay_with_cork=yes \
      --host=x86_64-linux --enable-shared=no --prefix="${APR_PREFIX}" && \
    make -j"$(nproc)" && make install)
fi

echo "==> external_lib provisioning complete"
