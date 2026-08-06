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

OPENSSL_SRC=openssl-4.0.1
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

OPENSSL_TARGET="$(openssl_configure_target)"
CFLAGS="$(openssl_cflags)"

# Native → platform asm; cross → portable C (no-asm).
OPENSSL_ASM_ARGS=()
if openssl_is_native_build; then
  OPENSSL_NATIVE=1
else
  OPENSSL_NATIVE=0
  OPENSSL_ASM_ARGS+=(no-asm)
fi

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
if [[ "${OPENSSL_NATIVE}" -eq 1 ]]; then
  echo "    asm: enabled (native ${ARCH}-${OS})"
else
  echo "    asm: disabled via no-asm (cross-compile)"
fi

rm -rf "${LIB_INSTALL_SRC}/${OPENSSL_SRC}"
(cd "${LIB_INSTALL_SRC}" && {
  [[ -f "${OPENSSL_SRC}.tar.gz" ]] || wget -q "https://github.com/openssl/openssl/releases/download/${OPENSSL_SRC}/${OPENSSL_SRC}.tar.gz"
  tar -xzf "${OPENSSL_SRC}.tar.gz"
})
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

echo "==> external_lib provisioning complete (${OPENSSL_MARKER})"
