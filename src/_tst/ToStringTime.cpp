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

#include "ToStringTime.h"
#include "lib.h"

const size_t kBufferSize = 64;

size_t ToStringTime::GetBufferSize() const
{
    return kBufferSize;
}

TEST_F(ToStringTime, BigValue) {
    double time = 500001.0;
    Time result = NormalizeTime(time);
    TimeToString(result, kBufferSize, GetBuffer());
    EXPECT_STREQ("138 h 53 min 21.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Hours) {
    double time = 7000.0;
    Time result = NormalizeTime(time);
    TimeToString(result, kBufferSize, GetBuffer());
    EXPECT_STREQ("1 h 56 min 40.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Minutes) {
    double time = 200.0;
    Time result = NormalizeTime(time);
    TimeToString(result, kBufferSize, GetBuffer());
    EXPECT_STREQ("3 min 20.000 sec", GetBuffer());
}

TEST_F(ToStringTime, Seconds) {
    double time = 20.0;

    Time result = NormalizeTime(time);
    TimeToString(result, kBufferSize, GetBuffer());
    EXPECT_STREQ("20.000 sec", GetBuffer());
}

TEST_F(ToStringTime, ZeroSize) {
    double time = 20.0;
    Time result = NormalizeTime(time);
    TimeToString(result, 0, GetBuffer());
    EXPECT_STREQ("", GetBuffer());
}

TEST_F(ToStringTime, NullString) {
    double time = 20.0;
    Time result = NormalizeTime(time);
    TimeToString(result, 10, NULL);
}
