/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2010-10-02
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#include "ToStringTime.h"

extern "C" {
#include "lib.h"
}

const size_t kBufferSize = 64;

size_t ToStringTime::GetBufferSize() const {
    return kBufferSize;
}

TEST_F(ToStringTime, BigValueYears) {
    const auto time = 50000001.0;
    auto result = lib_normalize_time(time);
    lib_time_to_string(&result, GetBuffer());
    EXPECT_STREQ("1 years 213 days 16 hr 53 min 21.000 sec", GetBuffer());
}

TEST_F(ToStringTime, BigValue) {
    const auto time = 500001.0;
    auto result = lib_normalize_time(time);
    lib_time_to_string(&result, GetBuffer());
    EXPECT_STREQ("5 days 18 hr 53 min 21.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Hours) {
    const auto time = 7000.0;
    auto result = lib_normalize_time(time);
    lib_time_to_string(&result, GetBuffer());
    EXPECT_STREQ("1 hr 56 min 40.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Minutes) {
    auto time = 200.0;
    auto result = lib_normalize_time(time);
    lib_time_to_string(&result, GetBuffer());
    EXPECT_STREQ("3 min 20.000 sec", GetBuffer());
    EXPECT_EQ(time, result.total_seconds);
}

TEST_F(ToStringTime, Seconds) {
    const auto time = 20.0;

    auto result = lib_normalize_time(time);
    lib_time_to_string(&result, GetBuffer());
    EXPECT_STREQ("20.000 sec", GetBuffer());
}

TEST_F(ToStringTime, NullString) {
    const auto time = 20.0;
    auto result = lib_normalize_time(time);
    lib_time_to_string(&result, nullptr);
}
