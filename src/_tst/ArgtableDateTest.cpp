/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains argtable test interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-14
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "ArgtableDateTest.h"
#include "argtable3.h"

void ArgtableDateTest::SetUp() {
    a = arg_date1(nullptr, nullptr, "%H:%M", nullptr, "time 23:59");
    b = arg_date0("b", nullptr, "%Y-%m-%d", nullptr, "date YYYY-MM-DD");
    c = arg_daten(nullptr, "date", "%D", nullptr, 1, 2, "MM/DD/YY");
    auto end = arg_end(20);
    argtable = static_cast<void**>(malloc(3 * sizeof(arg_dbl *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = b;
    argtable[2] = c;
    argtable[3] = end;
}

void ArgtableDateTest::TearDown() {
    arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_001) {
    // Arrange
    char* argv[] = { "program", "23:59", "--date", "12/31/04", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_EQ(a->tmval->tm_sec, 0);
    EXPECT_EQ(a->tmval->tm_min, 59);
    EXPECT_EQ(a->tmval->tm_hour, 23);
    EXPECT_EQ(a->tmval->tm_mday, 0);
    EXPECT_EQ(a->tmval->tm_mon, 0);
    EXPECT_EQ(a->tmval->tm_year, 0);
    EXPECT_EQ(a->tmval->tm_wday, 0);
    EXPECT_EQ(a->tmval->tm_yday, 0);
    EXPECT_EQ(a->tmval->tm_isdst, 0);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(1, c->count);
    EXPECT_EQ(c->tmval->tm_sec, 0);
    EXPECT_EQ(c->tmval->tm_min, 0);
    EXPECT_EQ(c->tmval->tm_hour, 0);
    EXPECT_EQ(c->tmval->tm_mday, 31);
    EXPECT_EQ(c->tmval->tm_mon, 11);
    EXPECT_EQ(c->tmval->tm_year, 104);
    EXPECT_EQ(c->tmval->tm_wday, 0);
    EXPECT_EQ(c->tmval->tm_yday, 0);
    EXPECT_EQ(c->tmval->tm_isdst, 0);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_002) {
    // Arrange
    char* argv[] = { "program", "--date", "12/31/04", "20:15", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_EQ(a->tmval->tm_sec, 0);
    EXPECT_EQ(a->tmval->tm_min, 15);
    EXPECT_EQ(a->tmval->tm_hour, 20);
    EXPECT_EQ(a->tmval->tm_mday, 0);
    EXPECT_EQ(a->tmval->tm_mon, 0);
    EXPECT_EQ(a->tmval->tm_year, 0);
    EXPECT_EQ(a->tmval->tm_wday, 0);
    EXPECT_EQ(a->tmval->tm_yday, 0);
    EXPECT_EQ(a->tmval->tm_isdst, 0);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(1, c->count);
    EXPECT_EQ(c->tmval->tm_sec, 0);
    EXPECT_EQ(c->tmval->tm_min, 0);
    EXPECT_EQ(c->tmval->tm_hour, 0);
    EXPECT_EQ(c->tmval->tm_mday, 31);
    EXPECT_EQ(c->tmval->tm_mon, 11);
    EXPECT_EQ(c->tmval->tm_year, 104);
    EXPECT_EQ(c->tmval->tm_wday, 0);
    EXPECT_EQ(c->tmval->tm_yday, 0);
    EXPECT_EQ(c->tmval->tm_isdst, 0);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_003) {
    // Arrange
    char* argv[] = { "program", "--date", "12/31/04", "20:15", "--date", "06/07/84", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_EQ(a->tmval->tm_sec, 0);
    EXPECT_EQ(a->tmval->tm_min, 15);
    EXPECT_EQ(a->tmval->tm_hour, 20);
    EXPECT_EQ(a->tmval->tm_mday, 0);
    EXPECT_EQ(a->tmval->tm_mon, 0);
    EXPECT_EQ(a->tmval->tm_year, 0);
    EXPECT_EQ(a->tmval->tm_wday, 0);
    EXPECT_EQ(a->tmval->tm_yday, 0);
    EXPECT_EQ(a->tmval->tm_isdst, 0);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(2, c->count);
    EXPECT_EQ(c->tmval->tm_sec, 0);
    EXPECT_EQ(c->tmval->tm_min, 0);
    EXPECT_EQ(c->tmval->tm_hour, 0);
    EXPECT_EQ(c->tmval->tm_mday, 31);
    EXPECT_EQ(c->tmval->tm_mon, 11);
    EXPECT_EQ(c->tmval->tm_year, 104);
    EXPECT_EQ(c->tmval->tm_wday, 0);
    EXPECT_EQ(c->tmval->tm_yday, 0);
    EXPECT_EQ(c->tmval->tm_isdst, 0);
    EXPECT_EQ((c->tmval + 1)->tm_sec, 0);
    EXPECT_EQ((c->tmval + 1)->tm_min, 0);
    EXPECT_EQ((c->tmval + 1)->tm_hour, 0);
    EXPECT_EQ((c->tmval + 1)->tm_mday, 7);
    EXPECT_EQ((c->tmval + 1)->tm_mon, 5);
    EXPECT_EQ((c->tmval + 1)->tm_year, 84);
    EXPECT_EQ((c->tmval + 1)->tm_wday, 0);
    EXPECT_EQ((c->tmval + 1)->tm_yday, 0);
    EXPECT_EQ((c->tmval + 1)->tm_isdst, 0);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_004) {
    // Arrange
    char* argv[] = { "program", "--date", "12/31/04", "20:15", "-b", "1982-11-28", "--date", "06/07/84", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_EQ(a->tmval->tm_sec, 0);
    EXPECT_EQ(a->tmval->tm_min, 15);
    EXPECT_EQ(a->tmval->tm_hour, 20);
    EXPECT_EQ(a->tmval->tm_mday, 0);
    EXPECT_EQ(a->tmval->tm_mon, 0);
    EXPECT_EQ(a->tmval->tm_year, 0);
    EXPECT_EQ(a->tmval->tm_wday, 0);
    EXPECT_EQ(a->tmval->tm_yday, 0);
    EXPECT_EQ(a->tmval->tm_isdst, 0);
    EXPECT_EQ(1, b->count);
    EXPECT_EQ(b->tmval->tm_sec, 0);
    EXPECT_EQ(b->tmval->tm_min, 0);
    EXPECT_EQ(b->tmval->tm_hour, 0);
    EXPECT_EQ(b->tmval->tm_mday, 28);
    EXPECT_EQ(b->tmval->tm_mon, 10);
    EXPECT_EQ(b->tmval->tm_year, 82);
    EXPECT_EQ(b->tmval->tm_wday, 0);
    EXPECT_EQ(b->tmval->tm_yday, 0);
    EXPECT_EQ(b->tmval->tm_isdst, 0);
    EXPECT_EQ(2, c->count);
    EXPECT_EQ(c->tmval->tm_sec, 0);
    EXPECT_EQ(c->tmval->tm_min, 0);
    EXPECT_EQ(c->tmval->tm_hour, 0);
    EXPECT_EQ(c->tmval->tm_mday, 31);
    EXPECT_EQ(c->tmval->tm_mon, 11);
    EXPECT_EQ(c->tmval->tm_year, 104);
    EXPECT_EQ(c->tmval->tm_wday, 0);
    EXPECT_EQ(c->tmval->tm_yday, 0);
    EXPECT_EQ(c->tmval->tm_isdst, 0);
    EXPECT_EQ((c->tmval + 1)->tm_sec, 0);
    EXPECT_EQ((c->tmval + 1)->tm_min, 0);
    EXPECT_EQ((c->tmval + 1)->tm_hour, 0);
    EXPECT_EQ((c->tmval + 1)->tm_mday, 7);
    EXPECT_EQ((c->tmval + 1)->tm_mon, 5);
    EXPECT_EQ((c->tmval + 1)->tm_year, 84);
    EXPECT_EQ((c->tmval + 1)->tm_wday, 0);
    EXPECT_EQ((c->tmval + 1)->tm_yday, 0);
    EXPECT_EQ((c->tmval + 1)->tm_isdst, 0);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_005) {
    // Arrange
    char* argv[] = { "program", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(2, nerrors);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_006) {
    // Arrange
    char* argv[] = { "program", "25:59", "--date", "12/31/04", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_007) {
    // Arrange
    char* argv[] = { "program", "23:59", "--date", "12/32/04", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_008) {
    // Arrange
    char* argv[] = { "program", "23:59", "--date", "12/31/04", "22:58", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_009) {
    // Arrange
    char* argv[] = { "program", "--date", "12/31/04", "20:15", "--date", "26/07/84", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDateTest, arg_parse_argdate_basic_010) {
    // Arrange
    char* argv[] = { "program", "-b", "1982-11-28", "-b", "1976-11-11", "--date", "12/07/84", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}
