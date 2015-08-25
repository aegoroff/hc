/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-11-18
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
}

using namespace std;

void onEachQueryCallback(Node_t* ast) {
}

void FrontendTest::SetUp()
{
    cout_stream_buffer_ = cout.rdbuf(oss_.rdbuf());
    TranslationUnitInit(&onEachQueryCallback);
}

void FrontendTest::TearDown()
{
    __try {
        cout.rdbuf(cout_stream_buffer_);
        TranslationUnitCleanup();
    } __finally {
        // TODO
    }
}

void FrontendTest::Run(const char* q, BOOL dontRunActions)
{
    char* utf8 = FromAnsiToUtf8(q, pool_);
    yy_scan_string(utf8);
}

void FrontendTest::ValidateNoError()
{
    ASSERT_FALSE(yyparse());
}

void FrontendTest::ValidateError()
{
    ASSERT_TRUE(yyparse());
}

TEST_F(FrontendTest, CalcFileHash) {
    Run("from file x in 'dfg' select x.md5;");
    ValidateNoError();
}

TEST_F(FrontendTest, SynErrNoSemicolon) {
    Run("from file x in 'dfg' select x.md5");
    ValidateError();
}

TEST_F(FrontendTest, SynErrInvalidStart) {
    Run("select x.md4 from file x in 'dfg' select x.md5;");
    ValidateError();
}