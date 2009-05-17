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

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	// ѕринудительно печатаем врем€ работы тестов.
	testing::GTEST_FLAG(print_time) = true;
	return RUN_ALL_TESTS();
}