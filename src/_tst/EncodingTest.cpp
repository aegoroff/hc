/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains encoding test implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2020-04-12
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */


#include "EncodingTest.h"

extern "C" {
#include <encoding.h>
}

void EncodingTest::SetUp() {
    apr_pool_create(&testPool_, pool_);
}


void EncodingTest::TearDown() {
    apr_pool_destroy(testPool_);
}

TEST_F(EncodingTest, FromUnicodeToAnsi) {
    // Arrange
    const char* utf8 = "тест";
    const wchar_t* unicode = L"тест";

    // Act
    char* result = enc_from_unicode_to_ansi(unicode, testPool_);

    // Assert
    ASSERT_STREQ(result, utf8);
}

TEST_F(EncodingTest, FromAnsiToUnicode) {
    // Arrange
    const char* ansi = "тест";
    const wchar_t* unicode = L"тест";

    // Act
    wchar_t* result = enc_from_ansi_to_unicode(ansi, testPool_);

    // Assert
    ASSERT_STREQ(result, unicode);
}