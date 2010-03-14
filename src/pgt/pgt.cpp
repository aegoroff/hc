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

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	// Print test time
	testing::GTEST_FLAG(print_time) = true;
	return RUN_ALL_TESTS();
}
