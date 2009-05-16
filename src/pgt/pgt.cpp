#include <stdio.h>
#include <tchar.h>
#include <windows.h>

#include "gtest.h"

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	// ѕринудительно печатаем врем€ работы тестов.
	testing::GTEST_FLAG(print_time) = true;
	return RUN_ALL_TESTS();
}