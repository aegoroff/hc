/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#include <memory>
#include "SizeToString.h"
#include "lib.h"

const size_t kBufferSize = 128;

size_t TSizeToString::GetBufferSize() const
{
    return kBufferSize;
}

TEST_F(TSizeToString, KBytesBoundary) {
    uint64_t size = 1024;
    FileSize result = NormalizeSize(size);
    SizeToString(size, kBufferSize, GetBuffer());
    EXPECT_STREQ("1.00 Kb (1024 bytes)", GetBuffer());
}

TEST_F(TSizeToString, KBytes) {
    uint64_t size = BINARY_THOUSAND * 2 + 10;
    FileSize result = NormalizeSize(size);
    SizeToString(size, kBufferSize, GetBuffer());
    EXPECT_STREQ("2.01 Kb (2058 bytes)", GetBuffer());
}

TEST_F(TSizeToString, Bytes) {
    uint64_t size = 20;
    FileSize result = NormalizeSize(size);
    SizeToString(size, kBufferSize, GetBuffer());
    EXPECT_STREQ("20 bytes", GetBuffer());
}

TEST_F(TSizeToString, BytesZero) {
    uint64_t size = 0;
    FileSize result = NormalizeSize(size);
    SizeToString(size, kBufferSize, GetBuffer());
    EXPECT_STREQ("0 bytes", GetBuffer());
}

TEST_F(TSizeToString, MaxValue) {
    uint64_t size = MAXUINT64;
    FileSize result = NormalizeSize(size);
    SizeToString(size, kBufferSize, GetBuffer());
    EXPECT_STREQ("16.00 Eb (18446744073709551615 bytes)", GetBuffer());
}
