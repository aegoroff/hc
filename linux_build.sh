BUILD_CONF=Release
ABI=$1
OS=$2
ARCH=$3
[[ -n "${ABI}" ]] || ABI=musl
[[ -n "${OS}" ]] || OS=linux
[[ -n "${ARCH}" ]] || ARCH=x86_64
BUILD_DIR=build-${ARCH}-${OS}-${ABI}-${BUILD_CONF}
LIB_INSTALL_SRC=./external_lib/src
LIB_INSTALL_PREFIX=./external_lib/lib
CC_FLAGS="zig cc -target ${ARCH}-${OS}-${ABI}"
APR_SRC=apr-1.7.4
APR_UTIL_SRC=apr-util-1.6.3
EXPAT_SRC=expat-2.5.0
OPENSSL_SRC=openssl-3.2.1

[[ -d "${LIB_INSTALL_SRC}" ]] || mkdir -p ${LIB_INSTALL_SRC}
#[[ -d "${LIB_INSTALL_PREFIX}" ]] && rm -rf ${LIB_INSTALL_PREFIX}
[[ -d "${LIB_INSTALL_PREFIX}" ]] || mkdir -p ${LIB_INSTALL_PREFIX}
rm -rf ${BUILD_DIR}
rm -rf ${LIB_INSTALL_SRC}/${EXPAT_SRC}
rm -rf ${LIB_INSTALL_SRC}/${APR_SRC}
rm -rf ${LIB_INSTALL_SRC}/${APR_UTIL_SRC}
rm -rf ${LIB_INSTALL_SRC}/${OPENSSL_SRC}

EXTERNAL_PREFIX=$(realpath ${LIB_INSTALL_PREFIX})
EXPAT_PREFIX=${EXTERNAL_PREFIX}/expat
APR_PREFIX=${EXTERNAL_PREFIX}/apr
OPENSSL_PREFIX=${EXTERNAL_PREFIX}/openssl

if [[ "${ARCH}" = "x86_64" ]]; then
	CFLAGS="-Ofast -march=haswell -mtune=haswell"
	if [[ "${ABI}" = "musl" ]] && [[ "${OS}" = "linux" ]]; then
		TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=$(realpath cmake/zig-toolchain-x86_64-linux-musl.cmake)"
	fi
	if [[ "${ABI}" = "gnu" ]] && [[ "${OS}" = "linux" ]]; then
		TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=$(realpath cmake/zig-toolchain-x86_64-linux-gnu.cmake)"
	fi
fi

if [[ ! -d "${OPENSSL_PREFIX}" ]]; then
	(cd ${LIB_INSTALL_SRC} && [[ -f "${OPENSSL_SRC}.tar.gz" ]] || wget https://github.com/openssl/openssl/releases/download/${OPENSSL_SRC}/${OPENSSL_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC} && tar -xzf ${OPENSSL_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC}/${OPENSSL_SRC} && AR="zig ar" RANLIB="zig ranlib" CC="${CC_FLAGS}" CFLAGS="${CFLAGS}" CXXFLAGS="${CFLAGS}" ./Configure -static --prefix=${OPENSSL_PREFIX} && make -j $(nproc) && make install_sw)
fi

if [[ ! -d "${EXPAT_PREFIX}" ]]; then
	(cd ${LIB_INSTALL_SRC} && [[ -f "${EXPAT_SRC}.tar.gz" ]] || wget https://github.com/libexpat/libexpat/releases/download/R_2_5_0/${EXPAT_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC} && tar -xzf ${EXPAT_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC}/${EXPAT_SRC} && AR="zig ar" RANLIB="zig ranlib" CC="${CC_FLAGS}" CFLAGS="${CFLAGS}" CXXFLAGS="${CFLAGS}" ./configure --host=x86_64-linux --enable-shared=no --prefix=${EXPAT_PREFIX} && make -j $(nproc) && make install)
fi

if [[ ! -d "${APR_PREFIX}" ]]; then
	(cd ${LIB_INSTALL_SRC} && [[ -f "${APR_SRC}.tar.gz" ]] || wget https://dlcdn.apache.org/apr/${APR_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC} && tar -xzf ${APR_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC}/${APR_SRC} && AR="zig ar" RANLIB="zig ranlib" CC="${CC_FLAGS}" CFLAGS="${CFLAGS} -Wno-implicit-function-declaration -Wno-int-conversion" ./configure ac_cv_file__dev_zero=yes apr_cv_process_shared_works=yes apr_cv_mutex_robust_shared=yes apr_cv_tcp_nodelay_with_cork=yes --host=x86_64-linux --enable-shared=no --prefix=${APR_PREFIX} && make -j $(nproc) && make install)

	(cd ${LIB_INSTALL_SRC} && [[ -f "${APR_UTIL_SRC}.tar.gz" ]] || wget https://dlcdn.apache.org/apr/${APR_UTIL_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC} && tar -xzf ${APR_UTIL_SRC}.tar.gz)
	(cd ${LIB_INSTALL_SRC}/${APR_UTIL_SRC} && AR="zig ar" RANLIB="zig ranlib" CC="${CC_FLAGS}" CFLAGS="${CFLAGS}" ./configure --host=x86_64-linux --enable-shared=no --prefix=${APR_PREFIX} --with-apr=${APR_PREFIX} --with-expat=${EXPAT_PREFIX} && make -j $(nproc) && make install)
fi

cmake -DCMAKE_BUILD_TYPE=${BUILD_CONF} -B ${BUILD_DIR} ${TOOLCHAIN}
cmake --build ${BUILD_DIR} --verbose --parallel $(nproc)

if [[ "${ARCH}" = "x86_64" ]] && [[ "${OS}" = "linux" ]]; then
	ctest --test-dir ${BUILD_DIR} -VV
fi

(cd ${BUILD_DIR} && cpack --config CPackConfig.cmake)
