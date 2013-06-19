#pragma once

#include "gtest.h"
#include <stdio.h>
#include <tchar.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

class TestUsingVectors : public ::testing::Test {
    private:
        vector<string> split(const char *str, char c = ' ')
        {
            vector<string> result;
            do
            {
                const char *begin = str;

                while(*str != c && *str)
                    str++;

                result.push_back(string(begin, str));
            } while (0 != *str++);

            return result;
        }
    protected:
        static void TearDownTestCase()
        {
        }

        static void SetUpTestCase()
        {
            ifstream infile("thefile.txt");
        }
};


#ifdef __cplusplus
}
#endif