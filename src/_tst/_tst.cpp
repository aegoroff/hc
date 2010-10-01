/*!
 * \brief   The file contains solution's unit tests
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2010
 */

#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include <memory>

#include "gtest.h"
#include "lib.h"

TEST(Htoi, 1SymbolByte) {
    EXPECT_EQ(5, htoi("5", 1));
}

TEST(Htoi, 2SymbolByte) {
    EXPECT_EQ(255, htoi("FF", 2));
}

TEST(Htoi, 2Bytes) {
    EXPECT_EQ(65518, htoi("FFEE", 4));
}

TEST(Htoi, 2BytesPartString) {
    EXPECT_EQ(255, htoi("FFFF", 2));
}

TEST(Htoi, NullString) {
    EXPECT_EQ(0, htoi(NULL, 2));
}

TEST(NormalizeSize, ZeroBytes) {
    uint64_t size = 0;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitBytes);
    EXPECT_EQ(result.value.sizeInBytes, size);
}

TEST(NormalizeSize, Bytes) {
    uint64_t size = 1023;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitBytes);
    EXPECT_EQ(result.value.sizeInBytes, size);
}

TEST(NormalizeSize, KBytesBoundary) {
    uint64_t size = 1024;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitKBytes);
    EXPECT_EQ(result.value.size, 1.0);
}

TEST(NormalizeSize, KBytes) {
    uint64_t size = BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitKBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, MBytes) {
    uint64_t size = BINARY_THOUSAND * BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitMBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, GBytes) {
    uint64_t size = BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND *
                              (uint64_t)4;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitGBytes);
    EXPECT_EQ(result.value.size, 4.0);
}

TEST(NormalizeSize, TBytes) {
    uint64_t size = (uint64_t)BINARY_THOUSAND * BINARY_THOUSAND *
                              BINARY_THOUSAND * BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitTBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, PBytes) {
    uint64_t size = (uint64_t)BINARY_THOUSAND * BINARY_THOUSAND *
                              BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitPBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, EBytes) {
    uint64_t size = (uint64_t)BINARY_THOUSAND * BINARY_THOUSAND *
                              BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND *
                              BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(SizeUnitEBytes, result.unit);
    EXPECT_EQ(2.0, result.value.size);
}

TEST(NormalizeTime, Hours) {
    double time = 7000.0;

    Time result = NormalizeTime(time);

    EXPECT_EQ(1, result.hours);
    EXPECT_EQ(56, result.minutes);
    EXPECT_FLOAT_EQ(40.00, result.seconds);
}

TEST(NormalizeTime, HoursFractial) {
    double time = 7000.51;

    Time result = NormalizeTime(time);

    EXPECT_EQ(1, result.hours);
    EXPECT_EQ(56, result.minutes);
    EXPECT_FLOAT_EQ(40.51, result.seconds);
}

TEST(NormalizeTime, Minutes) {
    double time = 200.0;

    Time result = NormalizeTime(time);

    EXPECT_EQ(0, result.hours);
    EXPECT_EQ(3, result.minutes);
    EXPECT_FLOAT_EQ(20.00, result.seconds);
}

TEST(NormalizeTime, Seconds) {
    double time = 50.0;

    Time result = NormalizeTime(time);

    EXPECT_EQ(0, result.hours);
    EXPECT_EQ(0, result.minutes);
    EXPECT_FLOAT_EQ(50.00, result.seconds);
}

TEST(NormalizeTime, BigValue) {
    double time = 500001.0;

    Time result = NormalizeTime(time);

    EXPECT_EQ(138, result.hours);
    EXPECT_EQ(53, result.minutes);
    EXPECT_FLOAT_EQ(21.00, result.seconds);
}

TEST(ToStringTime, BigValue) {
    double time = 500001.0;

    Time result = NormalizeTime(time);

    size_t sz = 64;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    TimeToString(result, sz, buffer.get());
    EXPECT_STREQ("138 h 53 min 21.000 sec", buffer.get());
}

TEST(ToStringTime, Hours) {
    double time = 7000.0;

    Time result = NormalizeTime(time);

    size_t sz = 64;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    TimeToString(result, sz, buffer.get());
    EXPECT_STREQ("1 h 56 min 40.000 sec", buffer.get());
}

TEST(ToStringTime, Minutes) {
    double time = 200.0;

    Time result = NormalizeTime(time);

    size_t sz = 64;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    TimeToString(result, sz, buffer.get());
    EXPECT_STREQ("3 min 20.000 sec", buffer.get());
}

TEST(ToStringTime, Seconds) {
    double time = 20.0;

    Time result = NormalizeTime(time);

    size_t sz = 64;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    TimeToString(result, sz, buffer.get());
    EXPECT_STREQ("20.000 sec", buffer.get());
}

TEST(ToStringTime, ZeroSize) {
    double time = 20.0;

    Time result = NormalizeTime(time);

    size_t sz = 64;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    TimeToString(result, 0, buffer.get());
    EXPECT_STREQ("", buffer.get());
}

TEST(ToStringTime, NullString) {
    double time = 20.0;
    Time result = NormalizeTime(time);
    TimeToString(result, 10, NULL);
}

TEST(SizeToSting, KBytesBoundary) {
    uint64_t size = 1024;

    FileSize result = NormalizeSize(size);

    size_t sz = 128;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    SizeToString(size, sz, buffer.get());
    EXPECT_STREQ("1.00 Kb (1024 bytes)", buffer.get());
}

TEST(SizeToSting, KBytes) {
    uint64_t size = BINARY_THOUSAND * 2 + 10;

    FileSize result = NormalizeSize(size);

    size_t sz = 128;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    SizeToString(size, sz, buffer.get());
    EXPECT_STREQ("2.01 Kb (2058 bytes)", buffer.get());
}

TEST(SizeToSting, Bytes) {
    uint64_t size = 20;

    FileSize result = NormalizeSize(size);

    size_t sz = 128;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    SizeToString(size, sz, buffer.get());
    EXPECT_STREQ("20 bytes", buffer.get());
}

TEST(SizeToSting, BytesZero) {
    uint64_t size = 0;

    FileSize result = NormalizeSize(size);

    size_t sz = 128;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    SizeToString(size, sz, buffer.get());
    EXPECT_STREQ("0 bytes", buffer.get());
}

TEST(SizeToSting, MaxValue) {
    uint64_t size = MAXUINT64;

    FileSize result = NormalizeSize(size);

    size_t sz = 128;
    std::auto_ptr<char> buffer(new char[sz]);
    memset(buffer.get(), 0, sz);
    SizeToString(size, sz, buffer.get());
    EXPECT_STREQ("16.00 Eb (18446744073709551615 bytes)", buffer.get());
}


int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    // Print test time
    testing::GTEST_FLAG(print_time) = true;
    return RUN_ALL_TESTS();
}
