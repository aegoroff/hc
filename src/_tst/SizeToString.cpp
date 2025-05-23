/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
 */

#include <memory>

#ifdef _MSC_VER
#include <tchar.h>
#include <Windows.h>
#else
#include <limits.h>
#define MAXUINT64 ULLONG_MAX
#endif

#include "SizeToString.h"

extern "C" {
#include <lib.h>
}

const size_t kBufferSize = 128;

size_t TSizeToString::GetBufferSize() const {
    return kBufferSize;
}

TEST_F(TSizeToString, KBytesBoundary) {
    const uint64_t size = 1024;
    lib_size_to_string(size, GetBuffer());
    EXPECT_STREQ("1.00 Kb (1024 bytes)", GetBuffer());
}

TEST_F(TSizeToString, KBytes) {
    const uint64_t size = BINARY_THOUSAND * 2 + 10;
    lib_size_to_string(size, GetBuffer());
    EXPECT_STREQ("2.01 Kb (2058 bytes)", GetBuffer());
}

TEST_F(TSizeToString, Bytes) {
    const uint64_t size = 20;
    lib_size_to_string(size, GetBuffer());
    EXPECT_STREQ("20 bytes", GetBuffer());
}

TEST_F(TSizeToString, BytesZero) {
    const uint64_t size = 0;
    lib_size_to_string(size, GetBuffer());
    EXPECT_STREQ("0 bytes", GetBuffer());
}

TEST_F(TSizeToString, MaxValue) {
    const uint64_t size = MAXUINT64;
    lib_size_to_string(size, GetBuffer());
    EXPECT_STREQ("16.00 Eb (18446744073709551615 bytes)", GetBuffer());
}
