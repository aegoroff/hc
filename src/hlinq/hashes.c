/*!
 * \brief   The file contains hashes from libtom lib API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
 */

#include "apr_hash.h"
#include    <tomcrypt.h>
#include "hashes.h"
#include "sph_md2.h"
#include "sph_ripemd.h"
#include "sph_sha2.h"
#include "sph_tiger.h"
#include "sph_sha2.h"
#include "sph_md5.h"
#include "sph_md4.h"
#include "crc32.h"
#include "sph_sha1.h"
#include "sph_whirlpool.h"
#include "gost.h"
#include "snefru.h"
#include "tth.h"

apr_hash_t* htAlgorithms = NULL;
apr_pool_t* pool;

void SetHash(
    const char* alg,
    int weight,
    size_t contextSize,
    apr_size_t  length,
    void (*PfnDigest)(apr_byte_t * digest, const void* input, const apr_size_t inputLen),
    void (*PfnInit)(void* context),
    void (*PfnFinal)(void* context, apr_byte_t * digest),
    void (*PfnUpdate)(void* context, const void* input, const apr_size_t inputLen)
);

HashDefinition* GetHash(const char* str)
{
    return (HashDefinition*)apr_hash_get(htAlgorithms, str, APR_HASH_KEY_STRING);
}


void SetHash(
    const char* alg,
    int weight,
    size_t contextSize,
    apr_size_t  length,
    void (*PfnDigest)(apr_byte_t * digest, const void* input, const apr_size_t inputLen),
    void (*PfnInit)(void* context),
    void (*PfnFinal)(void* context, apr_byte_t * digest),
    void (*PfnUpdate)(void* context, const void* input, const apr_size_t inputLen)
)
{
    HashDefinition* hash = (HashDefinition*)apr_pcalloc(pool, sizeof(HashDefinition));
    hash->ContextSize = contextSize;
    hash->PfnInit = PfnInit;
    hash->PfnUpdate = PfnUpdate;
    hash->PfnFinal = PfnFinal;
    hash->PfnDigest = PfnDigest;
    hash->HashLength = length;
    hash->Weight = weight;

    apr_hash_set(htAlgorithms, alg, APR_HASH_KEY_STRING, hash);
}

void WHIRLPOOLCalculateDigest(apr_byte_t*      digest,
                                      const void*      input,
                                      const apr_size_t inputLen)
{
    sph_whirlpool_context context = { 0 };

    sph_whirlpool_init(&context);
    sph_whirlpool(&context, input, inputLen);
    sph_whirlpool_close(&context, digest);
}

void SHA512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha512_context context = { 0 };

    sph_sha512_init(&context);
    sph_sha512(&context, input, inputLen);
    sph_sha512_close(&context, digest);
}

void SHA384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha384_context context = { 0 };

    sph_sha384_init(&context);
    sph_sha384(&context, input, inputLen);
    sph_sha384_close(&context, digest);
}

void SHA256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha256_context context = { 0 };

    sph_sha256_init(&context);
    sph_sha256(&context, input, inputLen);
    sph_sha256_close(&context, digest);
}

void SHA1CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha1_context context = { 0 };

    sph_sha1_init(&context);
    sph_sha1(&context, input, inputLen);
    sph_sha1_close(&context, digest);
}

void CRC32CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    Crc32Context context = { 0 };
    
    Crc32Init(&context);
    Crc32Update(&context, input, inputLen);
    Crc32Final(&context, digest);
}

void MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md2_context context = { 0 };

    sph_md2_init(&context);
    sph_md2(&context, input, inputLen);
    sph_md2_close(&context, digest);
}

void MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md4_context context = { 0 };

    sph_md4_init(&context);
    sph_md4(&context, input, inputLen);
    sph_md4_close(&context, digest);
}

void MD5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_md5_context context = { 0 };
    sph_md5_init(&context);
    sph_md5(&context, input, inputLen);
    sph_md5_close(&context, digest);
}

void TIGERCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_tiger_context context = { 0 };
    sph_tiger_init(&context);
    sph_tiger(&context, input, inputLen);
    sph_tiger_close(&context, digest);
}

void TIGER2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_tiger2_context context = { 0 };
    sph_tiger2_init(&context);
    sph_tiger2(&context, input, inputLen);
    sph_tiger2_close(&context, digest);
}

void SHA224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_sha224_context context = { 0 };
    sph_sha224_init(&context);
    sph_sha224(&context, input, inputLen);
    sph_sha224_close(&context, digest);
}

void RMD128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_ripemd128_context context = { 0 };

    sph_ripemd128_init(&context);
    sph_ripemd128(&context, input, inputLen);
    sph_ripemd128_close(&context, digest);
}

void RMD160CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    sph_ripemd160_context context = { 0 };

    sph_ripemd160_init(&context);
    sph_ripemd160(&context, input, inputLen);
    sph_ripemd160_close(&context, digest);
}

void LibtomCalculateDigest(
    apr_byte_t* digest,
    const void* input,
    const apr_size_t inputLen,
    int (* PfnInit)(hash_state* md),
    int (* PfnProcess)(hash_state* md, const unsigned char* in, unsigned long inlen),
    int (* PfnDone)(hash_state* md, unsigned char* hash)
                                  )
{
    hash_state context = { 0 };

    PfnInit(&context);

    if (input == NULL) {
        PfnProcess(&context, "", 0);
    } else {
        PfnProcess(&context, input, inputLen);
    }
    PfnDone(&context, digest);
}

void RMD256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    LibtomCalculateDigest(digest, input, inputLen, rmd256_init, rmd256_process, rmd256_done);
}

void RMD320CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    LibtomCalculateDigest(digest, input, inputLen, rmd320_init, rmd320_process, rmd320_done);
}

void GOSTCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    gost_ctx context = { 0 };

    rhash_gost_cryptopro_init(&context);
    rhash_gost_update(&context, input, inputLen);
    rhash_gost_final(&context, digest);
}

void SNEFRU128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    snefru_ctx context = { 0 };

    rhash_snefru128_init(&context);
    rhash_snefru_update(&context, input, inputLen);
    rhash_snefru_final(&context, digest);
}

void SNEFRU256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    snefru_ctx context = { 0 };

    rhash_snefru256_init(&context);
    rhash_snefru_update(&context, input, inputLen);
    rhash_snefru_final(&context, digest);
}

void TTHCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    tth_ctx context = { 0 };

    rhash_tth_init(&context);
    rhash_tth_update(&context, input, inputLen);
    rhash_tth_final(&context, digest);
}

/*
 * It MUST be last in the file so as not to declare internal functions in the header
*/
void InitializeHashes(apr_pool_t* p)
{
    htAlgorithms = apr_hash_make(p);
    pool = p;
    SetHash("crc32", 2, sizeof(Crc32Context), CRC32_HASH_SIZE, CRC32CalculateDigest, Crc32Init, Crc32Final, Crc32Update);
    SetHash("md2", 3, sizeof(sph_md2_context), SZ_MD2, MD2CalculateDigest, sph_md2_init, sph_md2_close, sph_md2);
    SetHash("md4", 3, sizeof(sph_md4_context), SZ_MD4, MD4CalculateDigest, sph_md4_init, sph_md4_close, sph_md4);
    SetHash("md5", 4, sizeof(sph_md5_context), SZ_MD5, MD5CalculateDigest, sph_md5_init, sph_md5_close, sph_md5);
    SetHash("sha1", 4, sizeof(sph_sha1_context), SZ_SHA1, SHA1CalculateDigest, sph_sha1_init, sph_sha1_close, sph_sha1);
    SetHash("sha224", 5, sizeof(sph_sha224_context), SZ_SHA224, SHA224CalculateDigest, sph_sha224_init, sph_sha224_close, sph_sha224);
    SetHash("sha256", 6, sizeof(sph_sha256_context), SZ_SHA256, SHA256CalculateDigest, sph_sha256_init, sph_sha256_close, sph_sha256);
    SetHash("sha384", 7, sizeof(sph_sha384_context), SZ_SHA384, SHA384CalculateDigest, sph_sha384_init, sph_sha384_close, sph_sha384);
    SetHash("sha512", 8, sizeof(sph_sha512_context), SZ_SHA512, SHA512CalculateDigest, sph_sha512_init, sph_sha512_close, sph_sha512);
    SetHash("ripemd128", 5, sizeof(sph_ripemd128_context), SZ_RIPEMD128, RMD128CalculateDigest, sph_ripemd128_init, sph_ripemd128_close, sph_ripemd128);
    SetHash("ripemd160", 5, sizeof(sph_ripemd160_context), SZ_RIPEMD160, RMD160CalculateDigest, sph_ripemd160_init, sph_ripemd160_close, sph_ripemd160);
    SetHash("ripemd256", 6, sizeof(hash_state), SZ_RIPEMD256, RMD256CalculateDigest, rmd256_init, rmd256_done, rmd256_process);
    SetHash("ripemd320", 7, sizeof(hash_state), SZ_RIPEMD320, RMD320CalculateDigest, rmd320_init, rmd320_done, rmd320_process);
    SetHash("tiger", 5, sizeof(sph_tiger_context), SZ_TIGER192, TIGERCalculateDigest, sph_tiger_init, sph_tiger_close, sph_tiger);
    SetHash("tiger2", 5, sizeof(sph_tiger2_context), SZ_TIGER192, TIGER2CalculateDigest, sph_tiger2_init, sph_tiger2_close, sph_tiger2);
    SetHash("whirlpool", 8, sizeof(sph_whirlpool_context), SZ_WHIRLPOOL, WHIRLPOOLCalculateDigest, sph_whirlpool_init, sph_whirlpool_close, sph_whirlpool);
    SetHash("gost", 9, sizeof(gost_ctx), SZ_GOST, GOSTCalculateDigest, rhash_gost_cryptopro_init, rhash_gost_final, rhash_gost_update);
    SetHash("snefru128", 10, sizeof(snefru_ctx), SZ_SNEFRU128, SNEFRU128CalculateDigest, rhash_snefru128_init, rhash_snefru_final, rhash_snefru_update);
    SetHash("snefru256", 10, sizeof(snefru_ctx), SZ_SNEFRU256, SNEFRU256CalculateDigest, rhash_snefru256_init, rhash_snefru_final, rhash_snefru_update);
    SetHash("tth", 5, sizeof(tth_ctx), SZ_TTH, TTHCalculateDigest, rhash_tth_init, rhash_tth_final, rhash_tth_update);
}
