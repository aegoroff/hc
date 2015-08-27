/*!
 * \brief   The file contains solution's unit tests
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-03-05
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#include <stdio.h>
#include <tchar.h>
#include <windows.h>
#include <memory>

#include "gtest.h"

extern "C" {
    #include "lib.h"
}

TEST(Htoi, 1SymbolByte) {
    EXPECT_EQ(5, htoi("5", 1));
}

TEST(Htoi, 2SymbolByte) {
    EXPECT_EQ(255, htoi("FF", 2));
}

TEST(Htoi, ZeroSize) {
    EXPECT_EQ(0, htoi("FF", 0));
}

TEST(Htoi, NegativeSize) {
    EXPECT_EQ(0, htoi("FF", -1));
}

TEST(Htoi, 2Bytes) {
    EXPECT_EQ(65518, htoi("FFEE", 4));
}

TEST(Htoi, TrimTest) {
    EXPECT_EQ(65518, htoi("  FFEE", 6));
}

TEST(Htoi, OnlyWhiteSpaces) {
    EXPECT_EQ(0, htoi(" \t", 2));
}

TEST(Htoi, TrimTestOfPartString) {
    EXPECT_EQ(255, htoi("  FFEE", 4));
}

TEST(Htoi, 2BytesPartString) {
    EXPECT_EQ(255, htoi("FFFF", 2));
}

TEST(Htoi, NullString) {
    EXPECT_EQ(0, htoi(NULL, 2));
}

TEST(Htoi, IncorrectStringAll) {
    EXPECT_EQ(0, htoi("RR", 2));
}

TEST(Htoi, IncorrectStringPart) {
    EXPECT_EQ(15, htoi("FR", 2));
}

TEST(NormalizeSize, ZeroBytes) {
    uint64_t size = 0;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitBytes);
    EXPECT_EQ(result.value.sizeInBytes, size);
}

TEST(NormalizeSize, Bytes) {
    uint64_t size = 1023;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitBytes);
    EXPECT_EQ(result.value.sizeInBytes, size);
}

TEST(NormalizeSize, KBytesBoundary) {
    uint64_t size = 1024;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitKBytes);
    EXPECT_EQ(result.value.size, 1.0);
}

TEST(NormalizeSize, KBytes) {
    uint64_t size = BINARY_THOUSAND * 2;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitKBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, MBytes) {
    uint64_t size = BINARY_THOUSAND * BINARY_THOUSAND * 2;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitMBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, GBytes) {
    uint64_t size = BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND *
            static_cast<uint64_t>(4);

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitGBytes);
    EXPECT_EQ(result.value.size, 4.0);
}

TEST(NormalizeSize, TBytes) {
    auto size = static_cast<uint64_t>(BINARY_THOUSAND) * BINARY_THOUSAND *
            BINARY_THOUSAND * BINARY_THOUSAND * 2;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitTBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, PBytes) {
    auto size = static_cast<uint64_t>(BINARY_THOUSAND) * BINARY_THOUSAND *
            BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * 2;

    auto result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitPBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, EBytes) {
    auto size = static_cast<uint64_t>(BINARY_THOUSAND) * BINARY_THOUSAND *
            BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND *
            BINARY_THOUSAND * 2;

    auto result = NormalizeSize(size);

    EXPECT_EQ(SizeUnitEBytes, result.unit);
    EXPECT_EQ(2.0, result.value.size);
}

TEST(NormalizeTime, Hours) {
    auto time = 7000.0;

    auto result = NormalizeTime(time);

    EXPECT_EQ(1, result.hours);
    EXPECT_EQ(56, result.minutes);
    EXPECT_FLOAT_EQ(40.00, result.seconds);
}

TEST(NormalizeTime, HoursFractial) {
    auto time = 7000.51;

    auto result = NormalizeTime(time);

    EXPECT_EQ(1, result.hours);
    EXPECT_EQ(56, result.minutes);
    EXPECT_FLOAT_EQ(40.51, result.seconds);
}

TEST(NormalizeTime, Minutes) {
    auto time = 200.0;

    auto result = NormalizeTime(time);

    EXPECT_EQ(0, result.hours);
    EXPECT_EQ(3, result.minutes);
    EXPECT_FLOAT_EQ(20.00, result.seconds);
}

TEST(NormalizeTime, Seconds) {
    auto time = 50.0;

    auto result = NormalizeTime(time);

    EXPECT_EQ(0, result.hours);
    EXPECT_EQ(0, result.minutes);
    EXPECT_FLOAT_EQ(50.00, result.seconds);
}

TEST(NormalizeTime, BigValue) {
    auto time = 500001.0;

    auto result = NormalizeTime(time);

    EXPECT_EQ(5, result.days);
    EXPECT_EQ(18, result.hours);
    EXPECT_EQ(53, result.minutes);
    EXPECT_FLOAT_EQ(21.00, result.seconds);
}

TEST(CountDigits, Zero) {
    EXPECT_EQ(1, CountDigitsIn(0.0));
}

TEST(CountDigits, One) {
    EXPECT_EQ(1, CountDigitsIn(1.0));
}

TEST(CountDigits, Ten) {
    EXPECT_EQ(2, CountDigitsIn(10.0));
}

TEST(CountDigits, N100) {
    EXPECT_EQ(3, CountDigitsIn(100.0));
}

TEST(CountDigits, N100F) {
    EXPECT_EQ(3, CountDigitsIn(100.23423));
}

TEST(CountDigits, N1000) {
    EXPECT_EQ(4, CountDigitsIn(1000.0));
}

TEST(CountDigits, N10000) {
    EXPECT_EQ(5, CountDigitsIn(10000.0));
}

TEST(CountDigits, N100000) {
    EXPECT_EQ(6, CountDigitsIn(100000.0));
}

TEST(CountDigits, N1000000) {
    EXPECT_EQ(7, CountDigitsIn(1000000.0));
}

TEST(CountDigits, N10000000) {
    EXPECT_EQ(8, CountDigitsIn(10000000.0));
}

TEST(CountDigits, N100000000) {
    EXPECT_EQ(9, CountDigitsIn(100000000.0));
}

TEST(CountDigits, N1000000000) {
    EXPECT_EQ(10, CountDigitsIn(1000000000.0));
}

TEST(CountDigits, N10000000000) {
    EXPECT_EQ(11, CountDigitsIn(10000000000.0));
}

TEST(CountDigits, N100000000000) {
    EXPECT_EQ(12, CountDigitsIn(100000000000.0));
}

TEST(GetFileName, Full) {
    ASSERT_STREQ("file.txt", GetFileName("c:\\path\\file.txt"));
}

TEST(GetFileName, OnlyFile) {
    ASSERT_STREQ("file.txt", GetFileName("file.txt"));
}

TEST(GetFileName, Null) {
    ASSERT_STREQ(NULL, GetFileName(NULL));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    // Print test time
    testing::GTEST_FLAG(print_time) = true;
    return RUN_ALL_TESTS();
}