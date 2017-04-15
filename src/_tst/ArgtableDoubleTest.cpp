/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains argtable test implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-04-13
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "ArgtableDoubleTest.h"
#include "argtable3.h"

void ArgtableDoubleTest::SetUp() {
    a = arg_dbl1(nullptr, nullptr, "a", "a is <double>");
    b = arg_dbl0(nullptr, nullptr, "b", "b is <double>");
    c = arg_dbl0(nullptr, nullptr, "c", "c is <double>");
    d = arg_dbln("dD", "delta", "<double>", 0, 3, "d can occur 0..3 times");
    e = arg_dbl0(nullptr, "eps,eqn", "<double>", "eps is optional");
    auto end = arg_end(20);
    n = 5;
    argtable = static_cast<void**>(malloc(n * sizeof(arg_dbl *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = b;
    argtable[2] = c;
    argtable[3] = d;
    argtable[4] = e;
    argtable[5] = end;
}

void ArgtableDoubleTest::TearDown() {
    arg_freetable(argtable, n + 1);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_001) {
    // Arrange
    char* argv[] = { "program", "0", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(0, a->dval[0]);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(0, c->count);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(0, e->count);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_002) {
    // Arrange
    char* argv[] = { "program", "1.234", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.234, a->dval[0]);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(0, c->count);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(0, e->count);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_003) {
    // Arrange
    char* argv[] = { "program", "1.8", "2.3", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.8, a->dval[0]);
    EXPECT_EQ(1, b->count);
    ASSERT_DOUBLE_EQ(2.3, b->dval[0]);
    EXPECT_EQ(0, c->count);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(0, e->count);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_004) {
    // Arrange
    char* argv[] = { "program", "5", "7", "9", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(5, a->dval[0]);
    EXPECT_EQ(1, b->count);
    ASSERT_DOUBLE_EQ(7, b->dval[0]);
    EXPECT_EQ(1, c->count);
    ASSERT_DOUBLE_EQ(9, c->dval[0]);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(0, e->count);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_005) {
    // Arrange
    char* argv[] = { "program", "1.9998", "-d", "13e-1", "-D", "17e-1", "--delta", "36e-1", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.9998, a->dval[0]);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(0, c->count);
    EXPECT_EQ(3, d->count);
    ASSERT_DOUBLE_EQ(13e-1, d->dval[0]);
    ASSERT_DOUBLE_EQ(17e-1, d->dval[1]);
    ASSERT_DOUBLE_EQ(36e-1, d->dval[2]);
    EXPECT_EQ(0, e->count);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_006) {
    // Arrange
    char* argv[] = { "program", "1.2", "2.3", "4.5", "--eps", "8.3456789", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.2, a->dval[0]);
    EXPECT_EQ(1, b->count);
    ASSERT_DOUBLE_EQ(2.3, b->dval[0]);
    EXPECT_EQ(1, c->count);
    ASSERT_DOUBLE_EQ(4.5, c->dval[0]);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(1, e->count);
    ASSERT_DOUBLE_EQ(8.3456789, e->dval[0]);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_007) {
    // Arrange
    char* argv[] = { "program", "1.2", "2.3", "4.5", "--eqn", "8.3456789", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.2, a->dval[0]);
    EXPECT_EQ(1, b->count);
    ASSERT_DOUBLE_EQ(2.3, b->dval[0]);
    EXPECT_EQ(1, c->count);
    ASSERT_DOUBLE_EQ(4.5, c->dval[0]);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(1, e->count);
    ASSERT_DOUBLE_EQ(8.3456789, e->dval[0]);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_008) {
    // Arrange
    char* argv[] = { "program", "1.2", "2.3", "4.5", "--eqn", "8.345", "-D", "0.234", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.2, a->dval[0]);
    EXPECT_EQ(1, b->count);
    ASSERT_DOUBLE_EQ(2.3, b->dval[0]);
    EXPECT_EQ(1, c->count);
    ASSERT_DOUBLE_EQ(4.5, c->dval[0]);
    EXPECT_EQ(1, d->count);
    ASSERT_DOUBLE_EQ(0.234, d->dval[0]);
    EXPECT_EQ(1, e->count);
    ASSERT_DOUBLE_EQ(8.345, e->dval[0]);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_009) {
    // Arrange
    char* argv[] = { "program", "1", "2", "3", "4", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_011) {
    // Arrange
    char* argv[] = { "program", "1", "2", "3", "-d1", "-d2", "-d3", "-d4", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_012) {
    // Arrange
    char* argv[] = { "program", "1", "2", "3", "--eps", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_013) {
    // Arrange
    char* argv[] = { "program", "1", "2", "3", "--eps", "3", "--eqn", "6", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_014) {
    // Arrange
    char* argv[] = { "program", "hello", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}

TEST_F(ArgtableDoubleTest, arg_parse_argdbl_basic_015) {
    // Arrange
    char* argv[] = { "program", "4", "hello", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(1, nerrors);
}
