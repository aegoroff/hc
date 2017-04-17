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
            Creation date: 2017-04-17
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "ArgtableRexTest.h"
#include "argtable3.h"

void ArgtableRexTest::SetUp() {
    a = arg_rex0("a", nullptr, "hello", nullptr, 0, "blah blah");
    b = arg_rex1(nullptr, "beta", "[Ww]orld", nullptr, 0, "blah blah");
    c = arg_rexn(nullptr, nullptr, "goodbye", nullptr, 1, 5, ARG_REX_ICASE, "blah blah");
    d = arg_rex0(nullptr, nullptr, "any.*", nullptr, ARG_REX_ICASE, "blah blah");
    auto end = arg_end(20);
    argtable = static_cast<void**>(malloc(GetOptionsCount() * sizeof(arg_rex *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = b;
    argtable[2] = c;
    argtable[3] = d;
    argtable[4] = end;
}

size_t ArgtableRexTest::GetOptionsCount() {
    return 4;
}

TEST_F(ArgtableRexTest, arg_parse_argrex_basic_001) {
    // Arrange
    char* argv[] = { "program", "--beta", "world", "goodbye", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(0, a->count);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(b->sval[0], "world");
    EXPECT_EQ(1, c->count);
    EXPECT_STREQ(c->sval[0], "goodbye");
    EXPECT_EQ(0, d->count);
}
