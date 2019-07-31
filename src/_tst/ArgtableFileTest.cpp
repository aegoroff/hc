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
            Creation date: 2017-04-15
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#include "ArgtableFileTest.h"
#include "argtable3.h"

void ArgtableFileTest::SetUp() {
    a = arg_file1(nullptr, nullptr, "<file>", "filename to test");
    auto end = arg_end(20);
    argtable = static_cast<void**>(malloc(GetOptionsCount() * sizeof(arg_file *) + sizeof(struct arg_end *)));
    argtable[0] = a;
    argtable[1] = end;
}

size_t ArgtableFileTest::GetOptionsCount() {
    return 1;
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_001) {
    // Arrange
    char* argv[] = { "program", "foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_002) {
    // Arrange
    char* argv[] = { "program", "/foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_003) {
    // Arrange
    char* argv[] = { "program", "./foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_004) {
    // Arrange
    char* argv[] = { "program", "././foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "././foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_005) {
    // Arrange
    char* argv[] = { "program", "./././foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./././foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_006) {
    // Arrange
    char* argv[] = { "program", "../foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_007) {
    // Arrange
    char* argv[] = { "program", "../../foo.bar", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../../foo.bar");
    EXPECT_STREQ(a->basename[0], "foo.bar");
    EXPECT_STREQ(a->extension[0], ".bar");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_008) {
    // Arrange
    char* argv[] = { "program", "foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_009) {
    // Arrange
    char* argv[] = { "program", "/foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_010) {
    // Arrange
    char* argv[] = { "program", "./foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_011) {
    // Arrange
    char* argv[] = { "program", "././foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "././foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_012) {
    // Arrange
    char* argv[] = { "program", "./././foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./././foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_013) {
    // Arrange
    char* argv[] = { "program", "../foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_014) {
    // Arrange
    char* argv[] = { "program", "../../foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../../foo");
    EXPECT_STREQ(a->basename[0], "foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_015) {
    // Arrange
    char* argv[] = { "program", ".foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], ".foo");
    EXPECT_STREQ(a->basename[0], ".foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_016) {
    // Arrange
    char* argv[] = { "program", "/.foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/.foo");
    EXPECT_STREQ(a->basename[0], ".foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_017) {
    // Arrange
    char* argv[] = { "program", "./.foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./.foo");
    EXPECT_STREQ(a->basename[0], ".foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_018) {
    // Arrange
    char* argv[] = { "program", "../.foo", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../.foo");
    EXPECT_STREQ(a->basename[0], ".foo");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_019) {
    // Arrange
    char* argv[] = { "program", "foo.", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "foo.");
    EXPECT_STREQ(a->basename[0], "foo.");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_020) {
    // Arrange
    char* argv[] = { "program", "/foo.", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/foo.");
    EXPECT_STREQ(a->basename[0], "foo.");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_021) {
    // Arrange
    char* argv[] = { "program", "./foo.", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./foo.");
    EXPECT_STREQ(a->basename[0], "foo.");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_022) {
    // Arrange
    char* argv[] = { "program", "../foo.", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../foo.");
    EXPECT_STREQ(a->basename[0], "foo.");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_023) {
    // Arrange
    char* argv[] = { "program", "/.foo.", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/.foo.");
    EXPECT_STREQ(a->basename[0], ".foo.");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_024) {
    // Arrange
    char* argv[] = { "program", "/.foo.c", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/.foo.c");
    EXPECT_STREQ(a->basename[0], ".foo.c");
    EXPECT_STREQ(a->extension[0], ".c");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_025) {
    // Arrange
    char* argv[] = { "program", "/.foo..b.c", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/.foo..b.c");
    EXPECT_STREQ(a->basename[0], ".foo..b.c");
    EXPECT_STREQ(a->extension[0], ".c");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_026) {
    // Arrange
    char* argv[] = { "program", "/", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_027) {
    // Arrange
    char* argv[] = { "program", ".", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], ".");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_028) {
    // Arrange
    char* argv[] = { "program", "..", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "..");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_029) {
    // Arrange
    char* argv[] = { "program", "/.", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/.");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_030) {
    // Arrange
    char* argv[] = { "program", "/..", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "/..");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_031) {
    // Arrange
    char* argv[] = { "program", "./", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "./");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}

TEST_F(ArgtableFileTest, arg_parse_argfile_basic_032) {
    // Arrange
    char* argv[] = { "program", "../", nullptr };
    int argc = sizeof(argv) / sizeof(char *) - 1;

    // Act
    auto nerrors = arg_parse(argc, argv, argtable);

    // Assert
    EXPECT_EQ(0, nerrors);
    EXPECT_EQ(1, a->count);
    EXPECT_STREQ(a->filename[0], "../");
    EXPECT_STREQ(a->basename[0], "");
    EXPECT_STREQ(a->extension[0], "");
}
