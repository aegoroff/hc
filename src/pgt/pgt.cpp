/*
 * Copyright 2009 Alexander Egorov
 */

#include <stdio.h>
#include <tchar.h>
#include <windows.h>

#include "gtest.h"
#include "pglib.h"

TEST(CalcMemorySize, Less1000) {
	size_t num = 100;
	EXPECT_EQ(num, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less65K) {
	size_t num = 10000;
	EXPECT_EQ(num / 5, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less500K) {
	size_t num = 100000;
	EXPECT_EQ(num / 10, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less9M600K) {
	size_t num = 600000;
	EXPECT_EQ(num / 12, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less71M) {
	size_t num = 60000000;
	EXPECT_EQ(num / 15, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less200M) {
	size_t num = 100000000;
	EXPECT_EQ(num / 17, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less800M) {
	size_t num = 300000000;
	EXPECT_EQ(num / 18, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less2B) {
	size_t num = 900000000;
	EXPECT_EQ(num / 19, CalculateMemorySize(num));
}

TEST(CalcMemorySize, Less4B) {
	size_t num = 3000000000;
	EXPECT_EQ(num / 20, CalculateMemorySize(num));
}

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

TEST(Permutation, Test) {
	int n = 2;
	int p[3];
	p[0] = 0;
	p[1] = 1;
	p[2] = 2;
	EXPECT_EQ(0, NextPermutation(n, p));
	EXPECT_EQ(0, p[0]);
	EXPECT_EQ(2, p[1]);
	EXPECT_EQ(1, p[2]);
	EXPECT_EQ(1, NextPermutation(n, p));
}

TEST(Permutation, Big) {
	int n = 9;
	int count = 1;
	int p[10];
	p[0] = 0;
	p[1] = 1;
	p[2] = 2;
	p[3] = 3;
	p[4] = 4;
	p[5] = 5;
	p[6] = 6;
	p[7] = 7;
	p[8] = 8;
	p[9] = 9;

	while (!NextPermutation(n, p)) {
		++count;
	}
	EXPECT_EQ(0, p[0]);
	EXPECT_EQ(362880, count);
}

TEST(Reverse, Normal) {
	char str[] = { 'a', 'b', 'c', 0 };
	ReverseString(str, 0, 2);
	EXPECT_STREQ("cba", str);
}

TEST(Reverse, RightOutOfRange) {
	char str[] = { 'a', 'b', 'c', 0 };
	ReverseString(str, 0, 3);
	EXPECT_STREQ("abc", str);
}

TEST(Reverse, LeftBiggerThenRight) {
	char str[] = { 'a', 'b', 'c', 0 };
	ReverseString(str, 2, 1);
	EXPECT_STREQ("abc", str);
}

TEST(Reverse, ShiftLeftToOne) {
	char str[] = { 'a', 'b', 'c', 0 };
	ReverseString(str, 1, 3 - 1);
	ReverseString(str, 0, 3 - 1);
	EXPECT_STREQ("bca", str);
}

TEST(Reverse, ShiftLeftToCustom) {
	unsigned int shiftSize = 2;
	char str[] = { 'a', 'b', 'c', 0 };
	ReverseString(str, 0, shiftSize - 1);
	ReverseString(str, shiftSize, strlen(str) - 1);
	ReverseString(str, 0, strlen(str) - 1);
	EXPECT_STREQ("cab", str);
}

TEST(NormalizeSize, ZeroBytes) {
	unsigned long long size = 0;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitBytes);
    EXPECT_EQ(result.value.sizeInBytes, size);
}

TEST(NormalizeSize, Bytes) {
	unsigned long long size = 1023;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitBytes);
    EXPECT_EQ(result.value.sizeInBytes, size);
}

TEST(NormalizeSize, KBytesBoundary) {
	unsigned long long size = 1024;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitKBytes);
    EXPECT_EQ(result.value.size, 1.0);
}

TEST(NormalizeSize, KBytes) {
	unsigned long long size = BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitKBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, MBytes) {
	unsigned long long size = BINARY_THOUSAND * BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitMBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, GBytes) {
	unsigned long long size = BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * (unsigned long long)4;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitGBytes);
    EXPECT_EQ(result.value.size, 4.0);
}

TEST(NormalizeSize, TBytes) {
	unsigned long long size = (unsigned long long)BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitTBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, PBytes) {
	unsigned long long size = (unsigned long long)BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * 2;

    FileSize result = NormalizeSize(size);

    EXPECT_EQ(result.unit, SizeUnitPBytes);
    EXPECT_EQ(result.value.size, 2.0);
}

TEST(NormalizeSize, EBytes) {
	unsigned long long size = (unsigned long long)BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * BINARY_THOUSAND * 2;

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

    PrintTime(time);
    CrtPrintf("\n");
}

TEST(NormalizeTime, Minutes) {
	double time = 200.0;

    Time result = NormalizeTime(time);

    EXPECT_EQ(0, result.hours);
    EXPECT_EQ(3, result.minutes);
    EXPECT_FLOAT_EQ(20.00, result.seconds);
    PrintTime(time);
    CrtPrintf("\n");
}

TEST(NormalizeTime, Seconds) {
	double time = 50.0;

    Time result = NormalizeTime(time);

    EXPECT_EQ(0, result.hours);
    EXPECT_EQ(0, result.minutes);
    EXPECT_FLOAT_EQ(50.00, result.seconds);
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	// Print test time
	testing::GTEST_FLAG(print_time) = true;
	return RUN_ALL_TESTS();
}
