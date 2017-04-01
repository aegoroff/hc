/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains hashes test class implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */


#include "HashTest.h"
#include "output.h"

const char* HashTest::GetHash(const char* algorithm) {
    return (const char*)apr_hash_get(htest_algorithms, algorithm, APR_HASH_KEY_STRING);
}

void HashTest::SetUp()
{
}

void HashTest::TearDown()
{
}

hash_definition_t* ht_hdef;

TEST_P(HashTest, Hash_Str123_HashAsSpecified) {
    // Arrange
    ht_hdef = hsh_get_hash(GetParam());
    auto t = "123";
    auto digest = (apr_byte_t*)apr_pcalloc(pool_, sizeof(apr_byte_t) * ht_hdef->hash_length_);

    // Act
    ht_hdef->pfn_digest_(digest, t, strlen(t));
    auto hash_str = out_hash_to_string(digest, FALSE, ht_hdef->hash_length_, pool_);

    // Assert
    ASSERT_STREQ(hash_str, GetHash(GetParam()));
}

INSTANTIATE_TEST_CASE_P(All,
    HashTest,
    ::testing::Values("crc32", "edonr256", "edonr512", "gost", "haval-128-3", "haval-128-4", "haval-128-5", "haval-160-3", "haval-160-4", "haval-160-5", "haval-192-3", "haval-192-4", "haval-192-5", "haval-224-3", "haval-224-4", "haval-224-5", "haval-256-3", "haval-256-4", "haval-256-5", "md2", "md4", "md5", "ntlm", "ripemd128", "ripemd160", "ripemd256", "ripemd320", "sha-3-224", "sha-3-256", "sha-3-384", "sha-3-512", "sha-3k-224", "sha-3k-256", "sha-3k-384", "sha-3k-512", "sha1", "sha224", "sha256", "sha384", "sha512", "snefru128", "snefru256", "tiger", "tiger2", "tth", "whirlpool", "blake2b", "blake2s"));