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

#include "ArgtableTest.h"
#include "argtable3.h"

void ArgtableTest::SetUp() {
}

void ArgtableTest::TearDown() {
    arg_freetable(argtable_, sizeof(argtable_) / sizeof(argtable_[0]));
}

TEST_F(ArgtableTest, arg_parse_argdbl_basic_001) {
    // Arrange
    auto a = arg_dbl1(nullptr, nullptr, "a", "a is <double>");
    auto b = arg_dbl0(nullptr, nullptr, "b", "b is <double>");
    auto c = arg_dbl0(nullptr, nullptr, "c", "c is <double>");
    auto d = arg_dbln("dD", "delta", "<double>", 0, 3, "d can occur 0..3 times");
    auto e = arg_dbl0(nullptr, "eps,eqn", "<double>", "eps is optional");
    auto end = arg_end(20);
    void* argtable[] = { a, b, c, d, e, end };
    argtable_ = argtable;
    int nerrors;

    char* argv[] = { "program", "0", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    nerrors = arg_parse(argc, argv, argtable_);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(0, a->dval[0]);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(0, c->count);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(0, e->count);
}

TEST_F(ArgtableTest, arg_parse_argdbl_basic_002) {
    // Arrange
    auto a = arg_dbl1(nullptr, nullptr, "a", "a is <double>");
    auto b = arg_dbl0(nullptr, nullptr, "b", "b is <double>");
    auto c = arg_dbl0(nullptr, nullptr, "c", "c is <double>");
    auto d = arg_dbln("dD", "delta", "<double>", 0, 3, "d can occur 0..3 times");
    auto e = arg_dbl0(nullptr, "eps,eqn", "<double>", "eps is optional");
    auto end = arg_end(20);
    void* argtable[] = { a, b, c, d, e, end };

    argtable_ = argtable;
    int nerrors;

    char* argv[] = { "program", "1.234", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    nerrors = arg_parse(argc, argv, argtable_);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    ASSERT_DOUBLE_EQ(1.234, a->dval[0]);
    EXPECT_EQ(0, b->count);
    EXPECT_EQ(0, c->count);
    EXPECT_EQ(0, d->count);
    EXPECT_EQ(0, e->count);
}
