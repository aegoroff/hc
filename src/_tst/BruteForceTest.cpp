/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2020
 */

#include "BruteForceTest.h"
#include "output.h"
#include "encoding.h"
#include "bf.h"

void BruteForceTest::SetUp() {
}

void BruteForceTest::TearDown() {
}

hash_definition_t* hdef;

void* bf_create_digest(const char* s, apr_pool_t* pool) {
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool, sizeof(apr_byte_t) * hdef->hash_length_));
    lib_hex_str_2_byte_array(s, digest, hdef->hash_length_);
    return digest;
}

int bft_compare_digests(apr_byte_t* digest1, apr_byte_t* digest2) {
    return memcmp(digest1, digest2, hdef->hash_length_) == 0;
}

int bf_compare_hash_attempt(void* hash, const void* pass, const uint32_t length) {
    apr_byte_t attempt[SZ_SHA512]; // hack to improve performance
    hdef->pfn_digest_(attempt, pass, static_cast<apr_size_t>(length));
    return bft_compare_digests(attempt, static_cast<apr_byte_t*>(hash));
}

int bf_compare_hash(apr_byte_t* digest, const char* check_sum) {
    const auto bytes = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    lib_hex_str_2_byte_array(check_sum, digest, hdef->hash_length_);
    return bft_compare_digests(bytes, digest);
}

TEST_P(BruteForceTest, BruteForce_CrackHash_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 4, "12345", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashWithBase64TransformStep_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str_base64 = out_hash_to_base64_string(digest, hdef->hash_length_, pool_);
    const auto hash_str = hsh_from_base64(hash_str_base64, pool_);

    std::cerr << "Base 64: " << hash_str_base64 << std::endl;
    std::cerr << "Raw: " << hash_str << std::endl;

    // Act
    const auto result = bf_brute_force(1, 4, "12345", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashDigitsDictAsTemplate_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 3, "0-9", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashDigitsDictAsTemplateAndCustomChars_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 3, "0-9+-.#~&*", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashDigitsAndLowCaseDictAsTemplate_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 3, "0-9a-z", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashAllDictClassesAsTemplate_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 3, "0-9a-zA-Z", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashAsciiDictAsTemplate_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 3, "ASCII", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashManyThreads_RestoredStringAsSpecified) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 2;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 4, "12345", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(t, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashTooSmallMaxLength_RestoredStringNull) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 2, "12345", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(NULL, result);
}

TEST_P(BruteForceTest, BruteForce_CrackHashDictionaryWithoutNecessaryChars_RestoredStringNull) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    const auto digest = static_cast<apr_byte_t*>(apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_));
    const auto t = "123";
    const uint32_t num_of_threads = 1;

    if(hdef->use_wide_string_) {
        const auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    } else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    const char* hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    const auto result = bf_brute_force(1, 3, "345", hash_str, bf_create_digest, num_of_threads, hdef->use_wide_string_, hdef->has_gpu_implementation_, hdef->gpu_context_, pool_);

    // Assert
    ASSERT_STREQ(NULL, result);
}

INSTANTIATE_TEST_CASE_P(All,
    BruteForceTest,
    ::testing::Values("crc32", "crc32c", "edonr256", "edonr512", "gost", "haval-128-3", "haval-128-4", "haval-128-5", "haval-160-3", "haval-160-4", "haval-160-5", "haval-192-3", "haval-192-4", "haval-192-5", "haval-224-3", "haval-224-4", "haval-224-5", "haval-256-3", "haval-256-4", "haval-256-5", "md2", "md4", "md5", "ntlm", "ripemd128", "ripemd160", "ripemd256", "ripemd320", "sha-3-224", "sha-3-256", "sha-3-384", "sha-3-512", "sha-3k-224", "sha-3k-256", "sha-3k-384", "sha-3k-512", "sha1", "sha224", "sha256", "sha384", "sha512", "snefru128", "snefru256", "tiger", "tiger2", "tth", "whirlpool"));
