cmake_minimum_required(VERSION 3.21)
project(hc LANGUAGES C CXX CUDA ASM)

if(NOT DEFINED _DISABLE_INSTALLATION)
  # This variable is responsible for installation disabling.
  set(_DISABLE_INSTALLATION FALSE)

  # Replace install() with conditional installation.
  macro(install)
    if (NOT _DISABLE_INSTALLATION)
      _install(${ARGN})
    endif()
  endmacro()
endif()

set(SRC src)
set(GEN generated)
set(HC hc)
set(L2H l2h)
set(TOMCRYPT libtomcrypt)
set(TST _tst)
set(LIB srclib)
set(PCRE pcre)
set(ARGTABLE3 argtable3)
set(APR apr)
set(APR_UTIL apr-util)
set(LIBD Debug)
set(LIBR Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BLAKE3_CFLAGS_AVX2 "-mavx2" CACHE STRING "the compiler flags to enable AVX2")
set(BLAKE3_SIMD_X86_INTRINSICS ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Store old value of flag for the case the external project already disables installation of current one.
set(_DISABLE_INSTALLATION_OLD ${_DISABLE_INSTALLATION})

set(_DISABLE_INSTALLATION TRUE)
FetchContent_MakeAvailable(googletest)
# Restore original install() behavior.
set(_DISABLE_INSTALLATION ${_DISABLE_INSTALLATION_OLD})

enable_testing()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(APR_X64 "$ENV{APR_HOME16}/${APR}/x64")
    set(APR_UTIL_X64 "$ENV{APR_HOME16}/${APR_UTIL}/x64")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
else ()
    set(APR_X64 "$ENV{APR_HOME}/APR")
    set(APR_UTIL_X64 "$ENV{APR_HOME}APR-UTIL")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif ()

set(compiler_flags)
set(debug_compiler_flags)
set(release_compiler_flags)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND compiler_flags
        /MP
        /Zc:wchar_t
        /Zc:inline
        /Zc:preprocessor
        /utf-8
        /validate-charset
        /sdl
        /std:c11
        /W3
    )
    list(APPEND debug_compiler_flags
        /MTd
        /Od
        /Zi
        /RTC1
    )
    list(APPEND release_compiler_flags
        /MT
        /Ox
        /Ob2
        /Oy
        /Oi
        /Ot
        /GL
        /Qpar
        /D
        NDEBUG
    )
    foreach(compiler_flag ${compiler_flags})
        set(FLAGS "${FLAGS} ${compiler_flag}")
    endforeach()
    foreach(compiler_flag ${release_compiler_flags})
        set(RELEASE_FLAGS "${RELEASE_FLAGS} ${compiler_flag}")
    endforeach()
    foreach(compiler_flag ${debug_compiler_flags})
        set(DEBUG_FLAGS "${DEBUG_FLAGS} ${compiler_flag}")
    endforeach()
    set(CMAKE_CXX_FLAGS_RELEASE "${FLAGS} ${RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${FLAGS} ${DEBUG_FLAGS}")
    set(CMAKE_C_FLAGS_RELEASE "${FLAGS} ${RELEASE_FLAGS} /TC")
    set(CMAKE_C_FLAGS_DEBUG "${FLAGS} ${DEBUG_FLAGS} /TC")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    else ()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG")
    endif ()
else ()
    list(APPEND compiler_flags
        -Wall
        -pthread
    )
    if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
        list(APPEND compiler_flags
            -march=haswell
            -mtune=haswell
            -mavx2
        )
    endif ()
    list(APPEND debug_compiler_flags
        -Og
    )
    list(APPEND release_compiler_flags
        -Ofast
        -s
    )
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        list(APPEND debug_compiler_flags
            -gfull
            -glldb
        )
    else ()
        list(APPEND debug_compiler_flags
            -ggdb
        )
    endif ()
    foreach(compiler_flag ${compiler_flags})
        set(FLAGS "${FLAGS} ${compiler_flag}")
    endforeach()
    foreach(compiler_flag ${debug_compiler_flags})
        set(DEBUG_FLAGS "${DEBUG_FLAGS} ${compiler_flag}")
    endforeach()
    foreach(compiler_flag ${release_compiler_flags})
        set(RELEASE_FLAGS "${RELEASE_FLAGS} ${compiler_flag}")
    endforeach()
    set(CMAKE_CXX_FLAGS_RELEASE "${FLAGS} ${RELEASE_FLAGS} -std=c++17")
    set(CMAKE_CXX_FLAGS_DEBUG "${FLAGS} ${DEBUG_FLAGS} -std=c++17")
    set(CMAKE_C_FLAGS_RELEASE "${FLAGS} ${RELEASE_FLAGS}")
    set(CMAKE_C_FLAGS_DEBUG "${FLAGS} ${DEBUG_FLAGS}")
endif ()

include_directories(${SRC}/${LIB})
include_directories(${SRC}/${L2H})
include_directories(${SRC}/${TOMCRYPT}/src/headers)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    include_directories($ENV{APR_HOME16}/${APR}/include)
    include_directories($ENV{APR_HOME16}/${APR_UTIL}/include)
else ()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/apr/include/apr-1)
endif ()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/${PCRE}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/openssl/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external_lib//lib/${ARGTABLE3})

add_definitions(-DAPR_DECLARE_STATIC)
add_definitions(-DAPU_DECLARE_STATIC)
add_definitions(-DUSE_KECCAK)
add_definitions(-DBLAKE3_NO_AVX512)
add_definitions(-DYYERROR_VERBOSE)
add_definitions(-DYY_USE_CONST)
add_definitions(-DSPH_64)
add_definitions(-DARCH="${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif ()

if ("$ENV{HC_VERSION}" STREQUAL "")
    add_definitions(-DPRODUCT_VERSION="5.3.0")
    set(CPACK_PACKAGE_VERSION "5.3.0")
else ()
    if ("$ENV{HC_VERSION}" STREQUAL "master")
        add_definitions(-DPRODUCT_VERSION="5.3.0-$ENV{CI_BUILD_NUMBER}")
        set(CPACK_PACKAGE_VERSION "5.3.0-$ENV{CI_BUILD_NUMBER}")
    else ()
        add_definitions(-DPRODUCT_VERSION="$ENV{HC_VERSION}")
        set(CPACK_PACKAGE_VERSION "$ENV{HC_VERSION}")
    endif ()
endif ()

SET(HC_SOURCE_FILES
    "${SRC}/${HC}/builtin.c"
    "${SRC}/${HC}/configuration.c"
    "${SRC}/${L2H}/hashes.c"
    "${SRC}/${HC}/dir.c"
    "${SRC}/${HC}/file.c"
    "${SRC}/${HC}/hash.c"
    "${SRC}/${HC}/str.c"
    "${SRC}/${HC}/hc.c"
    "${SRC}/${LIB}/dbg_helpers.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/${ARGTABLE3}/argtable3.c"
)

SET(L2H_SOURCE_FILES
    "${SRC}/${L2H}/backend.c"
    "${SRC}/${L2H}/configuration.c"
    "${SRC}/${L2H}/frontend.c"
    "${SRC}/${L2H}/hashes.c"
    "${SRC}/${L2H}/l2h.c"
    "${SRC}/${L2H}/processor.c"
    "${SRC}/${L2H}/treeutil.c"
    "${SRC}/${HC}/builtin.c"
    "${SRC}/${HC}/dir.c"
    "${SRC}/${HC}/file.c"
    "${SRC}/${HC}/hash.c"
    "${SRC}/${HC}/str.c"
    "${SRC}/${LIB}/dbg_helpers.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/${ARGTABLE3}/argtable3.c"
)

SET(CUDA_SOURCE_FILES
    "${SRC}/${HC}/crc32.cu"
    "${SRC}/${HC}/gpu.cu"
    "${SRC}/${HC}/md2.cu"
    "${SRC}/${HC}/md4.cu"
    "${SRC}/${HC}/md5.cu"
    "${SRC}/${HC}/rmd160.cu"
    "${SRC}/${HC}/sha1.cu"
    "${SRC}/${HC}/sha224.cu"
    "${SRC}/${HC}/sha256.cu"
    "${SRC}/${HC}/sha384.cu"
    "${SRC}/${HC}/sha512.cu"
    "${SRC}/${HC}/whirlpool.cu"
)

SET(TST_SOURCE_FILES
    "${SRC}/${TST}/_tst.cpp"
    "${SRC}/${TST}/BruteForceTest.cpp"
    "${SRC}/${TST}/BufferedTest.cpp"
    "${SRC}/${TST}/displayError.cpp"
    "${SRC}/${TST}/EncodingTest.cpp"
    "${SRC}/${TST}/FrontendTest.cpp"
    "${SRC}/${TST}/HashTest.cpp"
    "${SRC}/${TST}/ProcessorTest.cpp"
    "${SRC}/${TST}/SizeToString.cpp"
    "${SRC}/${TST}/ToStringTime.cpp"
    "${SRC}/${TST}/TreeTest.cpp"
    "${SRC}/${HC}/builtin.c"
    "${SRC}/${HC}/dir.c"
    "${SRC}/${HC}/file.c"
    "${SRC}/${HC}/hash.c"
    "${SRC}/${HC}/str.c"

    "${SRC}/${L2H}/backend.c"
    "${SRC}/${L2H}/frontend.c"
    "${SRC}/${L2H}/hashes.c"
    "${SRC}/${L2H}/processor.c"
    "${SRC}/${L2H}/treeutil.c"
)

SET(LICENSE_FILE "LICENSE.txt")

SET(LIB_SOURCE_FILES
    ${SRC}/${LIB}/b64.c
    ${SRC}/${LIB}/bf.c
    ${SRC}/${LIB}/blake3.c
    ${SRC}/${LIB}/blake3_dispatch.c
    ${SRC}/${LIB}/blake3_portable.c
    ${SRC}/${LIB}/blake3_avx2_x86-64_unix.S
    ${SRC}/${LIB}/blake3_avx512_x86-64_unix.S
    ${SRC}/${LIB}/blake3_sse2_x86-64_unix.S
    ${SRC}/${LIB}/blake3_sse41_x86-64_unix.S
    ${SRC}/${LIB}/byte_order.c
    ${SRC}/${LIB}/crc32.c
    ${SRC}/${LIB}/edonr.c
    ${SRC}/${LIB}/encoding.c
    ${SRC}/${LIB}/filehash.c
    ${SRC}/${LIB}/gost.c
    ${SRC}/${LIB}/haval.c
    ${SRC}/${LIB}/lib.c
    ${SRC}/${LIB}/md2.c
    ${SRC}/${LIB}/md4.c
    ${SRC}/${LIB}/output.c
    ${SRC}/${LIB}/rhash_tiger.c
    ${SRC}/${LIB}/ripemd.c
    ${SRC}/${LIB}/sha3.c
    ${SRC}/${LIB}/snefru.c
    ${SRC}/${LIB}/tiger.c
    ${SRC}/${LIB}/tiger_sbox.c
    ${SRC}/${LIB}/traverse.c
    ${SRC}/${LIB}/tth.c
)

SET(TOMCTYPT_SOURCE_FILES
    "${SRC}/${TOMCRYPT}/src/hashes/rmd128.c"
    "${SRC}/${TOMCRYPT}/src/hashes/rmd160.c"
    "${SRC}/${TOMCRYPT}/src/hashes/rmd256.c"
    "${SRC}/${TOMCRYPT}/src/hashes/rmd320.c"
    "${SRC}/${TOMCRYPT}/src/hashes/blake2b.c"
    "${SRC}/${TOMCRYPT}/src/hashes/blake2s.c"
    "${SRC}/${TOMCRYPT}/src/misc/crypt/crypt_argchk.c"
    "${SRC}/${TOMCRYPT}/src/misc/zeromem.c"
)

add_library(tomcypt STATIC ${TOMCTYPT_SOURCE_FILES})

link_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/openssl/lib64/"
    "${CMAKE_CURRENT_SOURCE_DIR}/external_lib/lib/apr/lib/"
    "${CMAKE_CURRENT_SOURCE_DIR}/external_lib/${PCRE}/lib/"
)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(FLEX_EXE win_flex.exe)
    set(BISON_EXE win_bison.exe)
else ()
    set(FLEX_EXE flex)
    set(BISON_EXE bison)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(FLEX_OPT --wincompat)
endif ()

set(FLEX ${FLEX_EXE} --fast ${FLEX_OPT} --outfile="${GEN}/l2h.flex.c" l2h.lex)
set(BISON ${BISON_EXE} --output="${GEN}/l2h.tab.c" -dy l2h.y)

add_custom_target(generated_directory ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}/${L2H}/${GEN}")

add_custom_target(generate_parser
        COMMAND ${FLEX}
        COMMAND ${BISON}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${SRC}/${L2H})

SET(L2H_GENERATED_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}/${L2H}/${GEN}/l2h.flex.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC}/${L2H}/${GEN}/l2h.tab.c")

set_source_files_properties(${L2H_GENERATED_FILES}
        PROPERTIES GENERATED TRUE)

add_library(libhc STATIC ${LIB_SOURCE_FILES})

target_include_directories(libhc PRIVATE
    ${SRC}/${HC}
)

add_executable(hc ${HC_SOURCE_FILES} ${CUDA_SOURCE_FILES})
set_property(TARGET hc 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(hc PRIVATE cxx_std_17)

target_include_directories(hc PRIVATE
    ${SRC}/${HC}
    ${SRC}/${HC}/include
)

add_executable(l2h ${L2H_GENERATED_FILES} ${L2H_SOURCE_FILES} ${CUDA_SOURCE_FILES})
set_property(TARGET l2h 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(l2h PRIVATE cxx_std_17)

target_include_directories(l2h PRIVATE
    ${SRC}/${L2H}/${GEN}
    ${SRC}/${L2H}/include
)

add_executable(_tst ${L2H_GENERATED_FILES} ${TST_SOURCE_FILES} ${CUDA_SOURCE_FILES})
target_compile_features(_tst PRIVATE cxx_std_17)
set_property(TARGET _tst 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(_tst PRIVATE
    ${SRC}/${HC}
    ${SRC}/${L2H}/${GEN}
    ${SRC}/${TST}
)

include(GoogleTest)

target_compile_definitions(_tst PRIVATE GTEST)

add_dependencies(generate_parser generated_directory)
add_dependencies(l2h generate_parser)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_link_libraries(libhc 
        PRIVATE ws2_32 
        PRIVATE rpcrt4 
        PRIVATE apr-1 
        PRIVATE aprutil-1 
        PRIVATE strsafe
        PRIVATE libcrypto)
    target_link_libraries(l2h PRIVATE libhc PRIVATE tomcypt PRIVATE pcre2-8-static)
    target_link_libraries(_tst PRIVATE libhc PRIVATE tomcypt PRIVATE pcre2-8-static PRIVATE GTest::gtest_main)
else ()
    target_link_libraries(libhc INTERFACE m INTERFACE libapr-1.a INTERFACE libaprutil-1.a INTERFACE crypto)
    target_link_libraries(l2h PRIVATE libhc PRIVATE tomcypt PRIVATE libpcre2-8.a)
    target_link_libraries(_tst PRIVATE libhc PRIVATE tomcypt PRIVATE libpcre2-8.a PRIVATE GTest::gtest_main)
    target_link_options(libhc PRIVATE "-static")
    target_link_options(hc PRIVATE "-static-libgcc" "-static-libstdc++")
    target_link_options(l2h PRIVATE "-static-libgcc" "-static-libstdc++")
    target_link_options(_tst PUBLIC "-static-libgcc")
    SET_TARGET_PROPERTIES(_tst PROPERTIES LINK_SEARCH_START_STATIC 1)
endif ()
target_link_libraries(hc PRIVATE libhc PRIVATE tomcypt)

add_test(NAME RunUnitTests COMMAND _tst)

install(TARGETS hc DESTINATION . COMPONENT application)
install(FILES ${LICENSE_FILE} DESTINATION . COMPONENT license)

set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    set(CPACK_SYSTEM_NAME  x86_64-pc-windows-msvc)
    set(CPACK_GENERATOR     TGZ)
elseif ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "zig-toolchain-x86_64-linux-musl")
    set(CPACK_SYSTEM_NAME  x86_64-unknown-linux-musl)
    set(CPACK_GENERATOR     TGZ)
elseif ("${CMAKE_TOOLCHAIN_FILE}" MATCHES "zig-toolchain-x86_64-linux-gnu")
    set(CPACK_SYSTEM_NAME  x86_64-unknown-linux-gnu)
    set(CPACK_GENERATOR     TGZ)
else()
    set(CPACK_SYSTEM_NAME  x86_64-unknown-linux-gnu)
    set(CPACK_GENERATOR     TGZ)
endif()

include(CPack)

cpack_add_component(application)
cpack_add_component(license)