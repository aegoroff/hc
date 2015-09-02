/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
 */

#include <iostream>
#include <stdio.h>
#include "FrontendTest.h"

extern "C" {
    #include <encoding.h>    
    #include "linq2hash.tab.h"
    struct yy_buffer_state* yy_scan_string(char *yy_str);
    extern int yylineno;
}

#define COMPILE_SUCCESS(q) ASSERT_TRUE(Compile((q)))
#define COMPILE_FAIL(q) ASSERT_FALSE(Compile((q)))

using namespace std;

void ftest_on_each_query_callback(fend_node_t* ast) {
    
}

void FrontendTest::SetUp()
{
    cout_stream_buffer_ = cout.rdbuf(oss_.rdbuf());
    fend_translation_unit_init(&ftest_on_each_query_callback);
}

void FrontendTest::TearDown()
{
    __try {
        cout.rdbuf(cout_stream_buffer_);
        fend_translation_unit_cleanup();
    } __finally {
        // TODO
    }
}

bool FrontendTest::Compile(const char* q) const {
    auto utf8 = enc_from_ansi_to_utf8(q, pool_);
    yy_scan_string(utf8);
    return !yyparse();
}

TEST_F(FrontendTest, SynErrNoSemicolon) {
    COMPILE_FAIL("from file x in 'dfg' select x.md5");
}

TEST_F(FrontendTest, SynErrUnclosedString) {
    COMPILE_FAIL("from file x in 'dfg select x.md5;");
}

TEST_F(FrontendTest, SynErrSeveralLineQ) {
    COMPILE_FAIL("from file x in\n 'dfg'\n select x.md5");
    ASSERT_EQ(3, yylineno);
}

TEST_F(FrontendTest, SynErrInvalidStart) {
    COMPILE_FAIL("select x.md4 from file x in 'dfg' select x.md5;");
}

TEST_F(FrontendTest, SelectSingleProp) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.md5;");
}

TEST_F(FrontendTest, SelectManyProp) {
    COMPILE_SUCCESS("from file x in 'dfg' select { x.md5, x.md2 };");
}

TEST_F(FrontendTest, SelectSingleMethodNoParams) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.m();");
}

TEST_F(FrontendTest, SelectSingleMethodOneParams) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.m(1);");
}

TEST_F(FrontendTest, SelectSingleMethodManyParams) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.m(1, '123');");
}

TEST_F(FrontendTest, SelectInto) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.md5 into x select x.crc32;");
}

TEST_F(FrontendTest, Join) {
    COMPILE_SUCCESS("from a in x join y in z on a.i equals y.i into gr select a.md5;");
}