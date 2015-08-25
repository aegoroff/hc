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
#include <fstream>
#include <stdio.h>
#include "FrontendTest.h"
using namespace std;

void FrontendTest::SetUp()
{
    cout_stream_buffer_ = cout.rdbuf(oss_.rdbuf());
}

void FrontendTest::TearDown()
{
    __try {
        cout.rdbuf(cout_stream_buffer_);
        
    } __finally {
        // TODO
    }
}

void FrontendTest::Run(const char* q, BOOL dontRunActions)
{
    const char* utf8 = FromAnsiToUtf8(q, pool_);
    
}

void FrontendTest::ValidateNoError()
{
    ASSERT_STREQ("", oss_.str().c_str());
}

void FrontendTest::ValidateError()
{
    ASSERT_TRUE(oss_.str().length() > 0);
}

TEST_F(FrontendTest, OnlyComment) {
    Run("# Comment\n");
    ValidateNoError();
}