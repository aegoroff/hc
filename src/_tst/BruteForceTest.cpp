/*!
 * \brief   The file contains class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2015-08-25
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include "BruteForceTest.h"
#include "output.h"
#include "encoding.h"

extern "C" {
    #include "bf.h"
}

void BruteForceTest::SetUp()
{
}

void BruteForceTest::TearDown()
{
}

hash_definition_t* hdef;

void* bf_create_digest(const char* s, apr_pool_t* pool)
{
    auto digest = (apr_byte_t*)apr_pcalloc(pool, sizeof(apr_byte_t) * hdef->hash_length_);
    lib_hex_str_2_byte_array(s, digest, hdef->hash_length_);
    return digest;
}

extern "C" int CompareDigests(apr_byte_t* digest1, apr_byte_t* digest2) {
    return memcmp(digest1, digest2, hdef->hash_length_) == 0;
}

int bf_compare_hash_attempt(void* hash, const void* pass, const uint32_t length)
{
    apr_byte_t attempt[SZ_SHA512]; // hack to improve performance
    hdef->pfn_digest_(attempt, pass, (apr_size_t)length);
    return CompareDigests(attempt, (apr_byte_t*)hash);
}

int bf_compare_hash(apr_byte_t* digest, const char* checkSum)
{
    apr_byte_t* bytes = (apr_byte_t*)apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_);
    lib_hex_str_2_byte_array(checkSum, digest, hdef->hash_length_);
    return CompareDigests(bytes, digest);
}

TEST_P(BruteForceTest, BruteForce_CrackHash_Success) {
    // Arrange
    hdef = hsh_get_hash(GetParam());
    auto digest = (apr_byte_t*)apr_pcalloc(pool_, sizeof(apr_byte_t) * hdef->hash_length_);
    auto t = "123";
    uint64_t attempts = 0;

    if (hdef->use_wide_string_) {
        auto s = enc_from_ansi_to_unicode(t, pool_);
        hdef->pfn_digest_(digest, s, wcslen(s) * sizeof(wchar_t));
    }
    else {
        hdef->pfn_digest_(digest, t, strlen(t));
    }

    auto hash_str = out_hash_to_string(digest, FALSE, hdef->hash_length_, pool_);

    // Act
    auto result = bf_brute_force(1, 3, "12345", hash_str, &attempts, bf_create_digest, 1, hdef->use_wide_string_, pool_);

    // Assert
    ASSERT_STREQ("123", result);
}

INSTANTIATE_TEST_CASE_P(All,
    BruteForceTest,
    ::testing::Values("crc32", "edonr256", "edonr512", "gost", "haval-128-3", "haval-128-4", "haval-128-5", "haval-160-3", "haval-160-4", "haval-160-5", "haval-192-3", "haval-192-4", "haval-192-5", "haval-224-3", "haval-224-4", "haval-224-5", "haval-256-3", "haval-256-4", "haval-256-5", "md2", "md4", "md5", "ntlm", "ripemd128", "ripemd160", "ripemd256", "ripemd320", "sha-3-224", "sha-3-256", "sha-3-384", "sha-3-512", "sha-3k-224", "sha-3k-256", "sha-3k-384", "sha-3k-512", "sha1", "sha224", "sha256", "sha384", "sha512", "snefru128", "snefru256", "tiger", "tiger2", "tth", "whirlpool"));