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
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include "ArgtableLitTest.h"
#include "argtable3.h"

void ArgtableLitTest::SetUp() {
    a = arg_lit0(nullptr, "hello,world", "either --hello or --world or none");
    b = arg_lit0("bB", nullptr, "either -b or -B or none");
    c = arg_lit1("cC", nullptr, "either -c or -C");
    d = arg_litn("dD", "delta", 2, 4, "-d|-D|--delta 2..4 occurences");
    help = arg_lit0(nullptr, "help", "print this help and exit");
    auto end = arg_end(20);
    argtable = static_cast<void**>(malloc(GetOptionsCount() * sizeof(arg_lit *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = b;
    argtable[2] = c;
    argtable[3] = d;
    argtable[4] = help;
    argtable[5] = end;
}

size_t ArgtableLitTest::GetOptionsCount() {
    return 5;
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_001) {
    // Arrange
    char* argv[] = { "program", "--help", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 2);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 0);
    EXPECT_EQ(d->count, 0);
    EXPECT_EQ(help->count, 1);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_002) {
    // Arrange
    char* argv[] = { "program", "-cDd", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 2);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_003) {
    // Arrange
    char* argv[] = { "program", "-cdDd", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 3);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_004) {
    // Arrange
    char* argv[] = { "program", "-CDd", "--delta", "--delta", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 4);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_005) {
    // Arrange
    char* argv[] = { "program", "--delta", "-cD", "-b", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 1);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 2);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_006) {
    // Arrange
    char* argv[] = { "program", "-D", "-B", "--delta", "-C", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 1);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 2);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_007) {
    // Arrange
    char* argv[] = { "program", "-D", "-B", "--delta", "-C", "--hello", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 1);
    EXPECT_EQ(b->count, 1);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 2);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_008) {
    // Arrange
    char* argv[] = { "program", "-D", "-B", "--delta", "-C", "--world", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 0);
    EXPECT_EQ(a->count, 1);
    EXPECT_EQ(b->count, 1);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 2);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_009) {
    // Arrange
    char* argv[] = { "program", "-c", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 0);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_010) {
    // Arrange
    char* argv[] = { "program", "-D", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 2);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 0);
    EXPECT_EQ(d->count, 1);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_011) {
    // Arrange
    char* argv[] = { "program", "-CD", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 1);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_012) {
    // Arrange
    char* argv[] = { "program", "-Dd", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 0);
    EXPECT_EQ(d->count, 2);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_013) {
    // Arrange
    char* argv[] = { "program", "-cddddd", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 4);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_014) {
    // Arrange
    char* argv[] = { "program", "-ccddd", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 3);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_015) {
    // Arrange
    char* argv[] = { "program", "-C", "-d", "-D", "--delta", "-b", "-B", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 0);
    EXPECT_EQ(b->count, 1);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 3);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_016) {
    // Arrange
    char* argv[] = { "program", "-C", "-d", "-D", "--delta", "--hello", "--world", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 1);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 3);
    EXPECT_EQ(help->count, 0);
}

TEST_F(ArgtableLitTest, arg_parse_arglit_basic_017) {
    // Arrange
    char* argv[] = { "program", "-C", "-d", "-D", "--delta", "--hello", "X", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(nerrors, 1);
    EXPECT_EQ(a->count, 1);
    EXPECT_EQ(b->count, 0);
    EXPECT_EQ(c->count, 1);
    EXPECT_EQ(d->count, 3);
    EXPECT_EQ(help->count, 0);
}
