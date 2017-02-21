// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <iostream>
#include <stdio.h>
#include "FrontendTest.h"

extern "C" {
    #include <encoding.h>    
    #include "linq2hash.tab.h"
    struct yy_buffer_state* yy_scan_string(char *yy_str);
    extern int yylineno;
    extern int fend_error_count;
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
    fend_error_count = 0;
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

bool FrontendTest::Compile(const char* q) {
    auto utf8 = enc_from_ansi_to_utf8(q, pool_);
    yy_scan_string(utf8);
    return !yyparse() && fend_error_count == 0;
}

TEST_F(FrontendTest, SynErr_NoSemicolon_Fail) {
    COMPILE_FAIL("from file x in 'dfg' select x.md5");
}

TEST_F(FrontendTest, SynErr_UnclosedString_Fail) {
    COMPILE_FAIL("from file x in 'dfg select x.md5;");
}

TEST_F(FrontendTest, SynErr_SeveralLineQWithoutSemicolon_Fail) {
    COMPILE_FAIL("from file x in\n 'dfg'\n select x.md5");
    ASSERT_EQ(3, yylineno);
}

TEST_F(FrontendTest, SynErr_InvalidStart_Fail) {
    COMPILE_FAIL("select x.md4 from file x in 'dfg' select x.md5;");
}

TEST_F(FrontendTest, SynErr_UndefinedVariable_Fail) {
    COMPILE_FAIL("from file x in 'dfg' select y.md5;");
}

TEST_F(FrontendTest, Select_SingleObjectProp_Success) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.md5;");
}

TEST_F(FrontendTest, Select_ManyStringQuery_Success) {
    COMPILE_SUCCESS("from file x in \n'dfg' \nselect x.md5;");
}

TEST_F(FrontendTest, Select_ManyPropInNewDynamicType_Success) {
    COMPILE_SUCCESS("from file x in 'dfg' select { x.md5, x.md2 };");
}

TEST_F(FrontendTest, Select_MethodWithoutParamsInSelectClause_Success) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.m();");
}

TEST_F(FrontendTest, Select_MethodOneParamInSelectClause_Success) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.m(1);");
}

TEST_F(FrontendTest, Select_MethodManyParamsInSelectClause_Success) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.m(1, '123');");
}

TEST_F(FrontendTest, SelectInto_CorrectSyntax_Success) {
    COMPILE_SUCCESS("from file x in 'dfg' select x.md5 into x select x.crc32;");
}

TEST_F(FrontendTest, Join_CorrectSyntax_Success) {
    COMPILE_SUCCESS("from a in x join y in z on a.i equals y.i into gr select a.md5;");
}

TEST_F(FrontendTest, RestoreString_FromHash_Success) {
    COMPILE_SUCCESS("from md5 x in '202CB962AC59075B964B07152D234B70' select x.string;");
}

TEST_F(FrontendTest, CreateHash_FromString_Success) {
    COMPILE_SUCCESS("from string x in '123' select x.md5;");
}

TEST_F(FrontendTest, Comment_CommentAndQuerString_Success) {
    COMPILE_SUCCESS("# test\r\nfrom string x in '123' select x.md5;");
}

TEST_F(FrontendTest, Comment_OnlyComment_Success) {
    COMPILE_SUCCESS("# test");
}
