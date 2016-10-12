/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include "ToStringTime.h"

extern "C" {
    #include "lib.h"
}

const size_t kBufferSize = 64;

size_t ToStringTime::GetBufferSize() const
{
    return kBufferSize;
}

TEST_F(ToStringTime, BigValueYears) {
    double time = 50000001.0;
    lib_time_t result = lib_normalize_time(time);
    lib_time_to_string(result, GetBuffer());
    EXPECT_STREQ("1 years 213 days 16 hr 53 min 21.000 sec", GetBuffer());
}

TEST_F(ToStringTime, BigValue) {
    double time = 500001.0;
    lib_time_t result = lib_normalize_time(time);
    lib_time_to_string(result, GetBuffer());
    EXPECT_STREQ("5 days 18 hr 53 min 21.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Hours) {
    double time = 7000.0;
    lib_time_t result = lib_normalize_time(time);
    lib_time_to_string(result, GetBuffer());
    EXPECT_STREQ("1 hr 56 min 40.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Minutes) {
    double time = 200.0;
    lib_time_t result = lib_normalize_time(time);
    lib_time_to_string(result, GetBuffer());
    EXPECT_STREQ("3 min 20.000 sec", GetBuffer());
    EXPECT_EQ(time, result.total_seconds);
}

TEST_F(ToStringTime, Seconds) {
    double time = 20.0;

    lib_time_t result = lib_normalize_time(time);
    lib_time_to_string(result, GetBuffer());
    EXPECT_STREQ("20.000 sec", GetBuffer());
}

TEST_F(ToStringTime, NullString) {
    double time = 20.0;
    lib_time_t result = lib_normalize_time(time);
    lib_time_to_string(result, nullptr);
}
