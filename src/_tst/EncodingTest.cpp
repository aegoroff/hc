/*!
 * \brief   The file contains encoding test implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2020-04-12
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */


#include "EncodingTest.h"

extern "C" {
#include "encoding.h"
#ifndef _MSC_VER
#include "types.h"
#endif
}

void EncodingTest::SetUp() {
    apr_pool_create(&testPool_, pool_);
}


void EncodingTest::TearDown() {
    apr_pool_destroy(testPool_);
}

TEST_F(EncodingTest, FromUnicodeToAnsi) {
    // Arrange

    // Act
    char* result = enc_from_unicode_to_ansi(unicode_, testPool_);

    // Assert
    ASSERT_STREQ(result, ansi_);
}

TEST_F(EncodingTest, FromUnicodeToUtf8) {
    // Arrange

    // Act
    char* result = enc_from_unicode_to_utf8(unicode_, testPool_);

    // Assert
    ASSERT_STREQ(result, utf8_);
    ASSERT_TRUE(enc_is_valid_utf8(result));
}

TEST_F(EncodingTest, FromUtf8ToUnicode) {
    // Arrange

    // Act
    const wchar_t* result = enc_from_utf8_to_unicode(utf8_, testPool_);

    // Assert
    ASSERT_STREQ(unicode_, result);
}

TEST_F(EncodingTest, FromAnsiToUnicode) {
    // Arrange

    // Act
    wchar_t* result = enc_from_ansi_to_unicode(ansi_, testPool_);

    // Assert
    ASSERT_STREQ(result, unicode_);
}

TEST_F(EncodingTest, FromAnsiToUtf8) {
    // Arrange

    // Act
    char* result = enc_from_ansi_to_utf8(ansi_, testPool_);

    // Assert
    ASSERT_STREQ(result, utf8_);
}

TEST_F(EncodingTest, FromUtf8ToAnsi) {
    // Arrange

    // Act
    char* result = enc_from_utf8_to_ansi(utf8_, testPool_);

    // Assert
    ASSERT_STREQ(result, ansi_);
}

TEST_F(EncodingTest, IsValidUtf8Success) {
    // Arrange

    // Act
    BOOL result = enc_is_valid_utf8(utf8_);

    // Assert
    ASSERT_TRUE(result);
}

TEST_F(EncodingTest, IsValidUtf8Fail) {
    // Arrange

    // Act
    BOOL result = enc_is_valid_utf8(ansi_);

    // Assert
    ASSERT_FALSE(result);
}

TEST_F(EncodingTest, DetectBomUtf8) {
    // Arrange
    const char* buffer = "\xEF\xBB\xBF\xd1\x82\xd0\xb5\xd1\x81\xd1\x82";
    size_t offset = 0;

    // Act
    bom_t result = enc_detect_bom_memory(buffer, 5, &offset);

    // Assert
    ASSERT_EQ(result, bom_utf8);
    ASSERT_EQ(offset, 3);
}

TEST_F(EncodingTest, DetectBomUtf16le) {
    // Arrange
    const char* buffer = "\xFF\xFE\x00\x00\x00\x00\x00\xd1\x81\xd1\x82";
    size_t offset = 0;

    // Act
    bom_t result = enc_detect_bom_memory(buffer, 5, &offset);

    // Assert
    ASSERT_EQ(result, bom_utf16le);
    ASSERT_EQ(offset, 2);
}

TEST_F(EncodingTest, DetectBomUtf16be) {
    // Arrange
    const char* buffer = "\xFE\xFF\x00\x00\x00\x00\x00\xd1\x81\xd1\x82";
    size_t offset = 0;

    // Act
    bom_t result = enc_detect_bom_memory(buffer, 5, &offset);

    // Assert
    ASSERT_EQ(result, bom_utf16be);
    ASSERT_EQ(offset, 2);
}

TEST_F(EncodingTest, DetectBomUtf32be) {
    // Arrange
    const char* buffer = "\x00\x00\xFE\xFF\x00\x00\x00\xd1\x81\xd1\x82";
    size_t offset = 0;

    // Act
    bom_t result = enc_detect_bom_memory(buffer, 5, &offset);

    // Assert
    ASSERT_EQ(result, bom_utf32be);
    ASSERT_EQ(offset, 4);
}

TEST_F(EncodingTest, DetectBomNoBom) {
    // Arrange
    const char* buffer = "\xd1\x82\xd0\xb5\xd1\x81\xd1\x82";
    size_t offset = 0;

    // Act
    bom_t result = enc_detect_bom_memory(buffer, 5, &offset);

    // Assert
    ASSERT_EQ(result, bom_unknown);
    ASSERT_EQ(offset, 0);
}