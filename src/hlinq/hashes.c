/*!
 * \brief   The file contains hashes from libtom lib API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include <tomcrypt.h>
#include <apr_tables.h>
#include <apr_hash.h>

#include "hashes.h"
#include "sph_md2.h"
#include "sph_ripemd.h"
#include "sph_sha2.h"
#include "sph_tiger.h"
#include "sph_md5.h"
#include "sph_md4.h"
#include "crc32.h"
#include "sph_sha1.h"
#include "sph_whirlpool.h"
#include "gost.h"
#include "snefru.h"
#include "tth.h"
#include "sph_haval.h"
#include "edonr.h"
#include "sha3.h"
#include "output.h"

apr_hash_t* htAlgorithms = NULL;
apr_pool_t* pool;

#define DIGEST_BODY(ctx, init, update, close) \
    ctx CTX = { 0 }; \
    init(&CTX); \
    update(&CTX, input, inputLen); \
    close(&CTX, digest);

#define ARRAY_INIT_SZ 50

/* compare function for qsort(3) */
static int CmpString(const void *v1, const void *v2)
{
    const char *s1 = *(const char**)v1;
    const char *s2 = *(const char**)v2;
    return strcmp(s1, s2);
}

void PrintHashes(void)
{
    apr_hash_index_t* hi = NULL;
    int i = 0;
    apr_array_header_t* arr = apr_array_make(pool, ARRAY_INIT_SZ, sizeof(const char*));

    CrtPrintf("  Supported hash algorithms:");
    NewLine();
    for (hi = apr_hash_first(NULL, htAlgorithms); hi; hi = apr_hash_next(hi)) {
        const char* k;
        HashDefinition* v;

        apr_hash_this(hi, (const void**)&k, NULL, (void**)&v);
        *(const char**)apr_array_push(arr) = k;
    }
    qsort(arr->elts, arr->nelts, arr->elt_size, CmpString);
    for (i = 0; i < arr->nelts; i++) {
        const char* elem = ((const char**)arr->elts)[i];
        CrtPrintf("    %s", elem);
        NewLine();
    }
}

const char* FromBase64(const char* base64, apr_pool_t* pool) {
    unsigned char* d = apr_palloc(pool, SZ_SHA512);
    unsigned long len;
    base64_decode(base64, strlen(base64), d, &len);
    return HashToString(d, TRUE, len, pool);
}

void SetHash(
    const char* alg,
    int weight,
    size_t contextSize,
    apr_size_t length,
    BOOL useWideString,
    void (* PfnDigest)(apr_byte_t* digest, const void* input, const apr_size_t inputLen),
    void (* PfnInit)(void* context),
    void (* PfnFinal)(void* context, apr_byte_t* digest),
    void (* PfnUpdate)(void* context, const void* input, const apr_size_t inputLen)
            );

HashDefinition* GetHash(const char* str)
{
    return (HashDefinition*)apr_hash_get(htAlgorithms, str, APR_HASH_KEY_STRING);
}


void SetHash(
    const char* alg,
    int weight,
    size_t contextSize,
    apr_size_t length,
    BOOL useWideString,
    void (* PfnDigest)(apr_byte_t* digest, const void* input, const apr_size_t inputLen),
    void (* PfnInit)(void* context),
    void (* PfnFinal)(void* context, apr_byte_t* digest),
    void (* PfnUpdate)(void* context, const void* input, const apr_size_t inputLen)
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
    hash->Name = alg;
    hash->UseWideString = useWideString;

    apr_hash_set(htAlgorithms, alg, APR_HASH_KEY_STRING, hash);
}

void WHIRLPOOLCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_whirlpool_context, sph_whirlpool_init, sph_whirlpool, sph_whirlpool_close)
}

void SHA512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_sha512_context, sph_sha512_init, sph_sha512, sph_sha512_close)
}

void SHA384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_sha384_context, sph_sha384_init, sph_sha384, sph_sha384_close)
}

void SHA256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_sha256_context, sph_sha256_init, sph_sha256, sph_sha256_close)
}

void SHA1CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_sha1_context, sph_sha1_init, sph_sha1, sph_sha1_close)
}

void CRC32CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(Crc32Context, Crc32Init, Crc32Update, Crc32Final)
}

void MD2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_md2_context, sph_md2_init, sph_md2, sph_md2_close)
}

void MD4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_md4_context, sph_md4_init, sph_md4, sph_md4_close)
}

void MD5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_md5_context, sph_md5_init, sph_md5, sph_md5_close)
}

void TIGERCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_tiger_context, sph_tiger_init, sph_tiger, sph_tiger_close)
}

void TIGER2CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_tiger2_context, sph_tiger2_init, sph_tiger2, sph_tiger2_close)
}

void SHA224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_sha224_context, sph_sha224_init, sph_sha224, sph_sha224_close)
}

void RMD128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_ripemd128_context, sph_ripemd128_init, sph_ripemd128, sph_ripemd128_close)
}

void RMD160CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_ripemd160_context, sph_ripemd160_init, sph_ripemd160, sph_ripemd160_close)
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
    DIGEST_BODY(gost_ctx, rhash_gost_cryptopro_init, rhash_gost_update, rhash_gost_final)
}

void SNEFRU128CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(snefru_ctx, rhash_snefru128_init, rhash_snefru_update, rhash_snefru_final)
}

void SNEFRU256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(snefru_ctx, rhash_snefru256_init, rhash_snefru_update, rhash_snefru_final)
}

void TTHCalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(tth_ctx, rhash_tth_init, rhash_tth_update, rhash_tth_final)
}

void HAVAL128_3CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval128_3_init, sph_haval128_3, sph_haval128_3_close)
}

void HAVAL128_4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval128_4_init, sph_haval128_4, sph_haval128_4_close)
}

void HAVAL128_5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval128_5_init, sph_haval128_5, sph_haval128_5_close)
}

void HAVAL160_3CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval160_3_init, sph_haval160_3, sph_haval160_3_close)
}

void HAVAL160_4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval160_4_init, sph_haval160_4, sph_haval160_4_close)
}

void HAVAL160_5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval160_5_init, sph_haval160_5, sph_haval160_5_close)
}

void HAVAL192_3CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval192_3_init, sph_haval192_3, sph_haval192_3_close)
}

void HAVAL192_4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval192_4_init, sph_haval192_4, sph_haval192_4_close)
}

void HAVAL192_5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval192_5_init, sph_haval192_5, sph_haval192_5_close)
}

void HAVAL224_3CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval224_3_init, sph_haval224_3, sph_haval224_3_close)
}

void HAVAL224_4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval224_4_init, sph_haval224_4, sph_haval224_4_close)
}

void HAVAL224_5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval224_5_init, sph_haval224_5, sph_haval224_5_close)
}

void HAVAL256_3CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval256_3_init, sph_haval256_3, sph_haval256_3_close)
}

void HAVAL256_4CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval256_4_init, sph_haval256_4, sph_haval256_4_close)
}

void HAVAL256_5CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sph_haval_context, sph_haval256_5_init, sph_haval256_5, sph_haval256_5_close)
}

void EDONR256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(edonr_ctx, rhash_edonr256_init, rhash_edonr256_update, rhash_edonr256_final)
}

void EDONR512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(edonr_ctx, rhash_edonr512_init, rhash_edonr512_update, rhash_edonr512_final)
}

void SHA3224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_224_init, rhash_sha3_update, rhash_sha3_final)
}

void SHA3256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_256_init, rhash_sha3_update, rhash_sha3_final)
}

void SHA3384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_384_init, rhash_sha3_update, rhash_sha3_final)
}

void SHA3512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_512_init, rhash_sha3_update, rhash_sha3_final)
}

void SHA3K224CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_224_init, rhash_keccak_update, rhash_keccak_final)
}

void SHA3K256CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_256_init, rhash_keccak_update, rhash_keccak_final)
}

void SHA3K384CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_384_init, rhash_keccak_update, rhash_keccak_final)
}

void SHA3K512CalculateDigest(apr_byte_t* digest, const void* input, const apr_size_t inputLen)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_512_init, rhash_keccak_update, rhash_keccak_final)
}

/*
 * It MUST be last in the file so as not to declare internal functions in the header
 */
void InitializeHashes(apr_pool_t* p)
{
    htAlgorithms = apr_hash_make(p);
    pool = p;
    SetHash("crc32", 2, sizeof(Crc32Context), CRC32_HASH_SIZE, FALSE, CRC32CalculateDigest, Crc32Init, Crc32Final, Crc32Update);
    SetHash("md2", 3, sizeof(sph_md2_context), SZ_MD2, FALSE, MD2CalculateDigest, sph_md2_init, sph_md2_close, sph_md2);
    SetHash("md4", 3, sizeof(sph_md4_context), SZ_MD4, FALSE, MD4CalculateDigest, sph_md4_init, sph_md4_close, sph_md4);
    SetHash("ntlm", 3, sizeof(sph_md4_context), SZ_MD4, TRUE, MD4CalculateDigest, sph_md4_init, sph_md4_close, sph_md4);
    SetHash("md5", 4, sizeof(sph_md5_context), SZ_MD5, FALSE, MD5CalculateDigest, sph_md5_init, sph_md5_close, sph_md5);
    SetHash("sha1", 4, sizeof(sph_sha1_context), SZ_SHA1, FALSE, SHA1CalculateDigest, sph_sha1_init, sph_sha1_close, sph_sha1);
    SetHash("sha224", 5, sizeof(sph_sha224_context), SZ_SHA224, FALSE, SHA224CalculateDigest, sph_sha224_init, sph_sha224_close, sph_sha224);
    SetHash("sha256", 6, sizeof(sph_sha256_context), SZ_SHA256, FALSE, SHA256CalculateDigest, sph_sha256_init, sph_sha256_close, sph_sha256);
    SetHash("sha384", 7, sizeof(sph_sha384_context), SZ_SHA384, FALSE, SHA384CalculateDigest, sph_sha384_init, sph_sha384_close, sph_sha384);
    SetHash("sha512", 8, sizeof(sph_sha512_context), SZ_SHA512, FALSE, SHA512CalculateDigest, sph_sha512_init, sph_sha512_close, sph_sha512);
    SetHash("ripemd128", 5, sizeof(sph_ripemd128_context), SZ_RIPEMD128, FALSE, RMD128CalculateDigest, sph_ripemd128_init, sph_ripemd128_close, sph_ripemd128);
    SetHash("ripemd160", 5, sizeof(sph_ripemd160_context), SZ_RIPEMD160, FALSE, RMD160CalculateDigest, sph_ripemd160_init, sph_ripemd160_close, sph_ripemd160);
    SetHash("ripemd256", 6, sizeof(hash_state), SZ_RIPEMD256, FALSE, RMD256CalculateDigest, rmd256_init, rmd256_done, rmd256_process);
    SetHash("ripemd320", 7, sizeof(hash_state), SZ_RIPEMD320, FALSE, RMD320CalculateDigest, rmd320_init, rmd320_done, rmd320_process);
    SetHash("tiger", 5, sizeof(sph_tiger_context), SZ_TIGER192, FALSE, TIGERCalculateDigest, sph_tiger_init, sph_tiger_close, sph_tiger);
    SetHash("tiger2", 5, sizeof(sph_tiger2_context), SZ_TIGER192, FALSE, TIGER2CalculateDigest, sph_tiger2_init, sph_tiger2_close, sph_tiger2);
    SetHash("whirlpool", 8, sizeof(sph_whirlpool_context), SZ_WHIRLPOOL, FALSE, WHIRLPOOLCalculateDigest, sph_whirlpool_init, sph_whirlpool_close, sph_whirlpool);
    SetHash("gost", 9, sizeof(gost_ctx), SZ_GOST, FALSE, GOSTCalculateDigest, rhash_gost_cryptopro_init, rhash_gost_final, rhash_gost_update);
    SetHash("snefru128", 10, sizeof(snefru_ctx), SZ_SNEFRU128, FALSE, SNEFRU128CalculateDigest, rhash_snefru128_init, rhash_snefru_final, rhash_snefru_update);
    SetHash("snefru256", 10, sizeof(snefru_ctx), SZ_SNEFRU256, FALSE, SNEFRU256CalculateDigest, rhash_snefru256_init, rhash_snefru_final, rhash_snefru_update);
    SetHash("tth", 5, sizeof(tth_ctx), SZ_TTH, FALSE, TTHCalculateDigest, rhash_tth_init, rhash_tth_final, rhash_tth_update);
    SetHash("haval-128-3", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, HAVAL128_3CalculateDigest, sph_haval128_3_init, sph_haval128_3_close, sph_haval128_3);
    SetHash("haval-128-4", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, HAVAL128_4CalculateDigest, sph_haval128_4_init, sph_haval128_4_close, sph_haval128_4);
    SetHash("haval-128-5", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, HAVAL128_5CalculateDigest, sph_haval128_5_init, sph_haval128_5_close, sph_haval128_5);
    SetHash("haval-160-3", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, HAVAL160_3CalculateDigest, sph_haval160_3_init, sph_haval160_3_close, sph_haval160_3);
    SetHash("haval-160-4", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, HAVAL160_4CalculateDigest, sph_haval160_4_init, sph_haval160_4_close, sph_haval160_4);
    SetHash("haval-160-5", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, HAVAL160_5CalculateDigest, sph_haval160_5_init, sph_haval160_5_close, sph_haval160_5);
    SetHash("haval-192-3", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, HAVAL192_3CalculateDigest, sph_haval192_3_init, sph_haval192_3_close, sph_haval192_3);
    SetHash("haval-192-4", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, HAVAL192_4CalculateDigest, sph_haval192_4_init, sph_haval192_4_close, sph_haval192_4);
    SetHash("haval-192-5", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, HAVAL192_5CalculateDigest, sph_haval192_5_init, sph_haval192_5_close, sph_haval192_5);
    SetHash("haval-224-3", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, HAVAL224_3CalculateDigest, sph_haval224_3_init, sph_haval224_3_close, sph_haval224_3);
    SetHash("haval-224-4", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, HAVAL224_4CalculateDigest, sph_haval224_4_init, sph_haval224_4_close, sph_haval224_4);
    SetHash("haval-224-5", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, HAVAL224_5CalculateDigest, sph_haval224_5_init, sph_haval224_5_close, sph_haval224_5);
    SetHash("haval-256-3", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, HAVAL256_3CalculateDigest, sph_haval256_3_init, sph_haval256_3_close, sph_haval256_3);
    SetHash("haval-256-4", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, HAVAL256_4CalculateDigest, sph_haval256_4_init, sph_haval256_4_close, sph_haval256_4);
    SetHash("haval-256-5", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, HAVAL256_5CalculateDigest, sph_haval256_5_init, sph_haval256_5_close, sph_haval256_5);
    SetHash("edonr256", 5, sizeof(edonr_ctx), SZ_EDONR256, FALSE, EDONR256CalculateDigest, rhash_edonr256_init, rhash_edonr256_final, rhash_edonr256_update);
    SetHash("edonr512", 5, sizeof(edonr_ctx), SZ_EDONR512, FALSE, EDONR512CalculateDigest, rhash_edonr512_init, rhash_edonr512_final, rhash_edonr512_update);
    SetHash("sha-3-224", 9, sizeof(sha3_ctx), SZ_SHA224, FALSE, SHA3224CalculateDigest, rhash_sha3_224_init, rhash_sha3_final, rhash_sha3_update);
    SetHash("sha-3-256", 9, sizeof(sha3_ctx), SZ_SHA256, FALSE, SHA3256CalculateDigest, rhash_sha3_256_init, rhash_sha3_final, rhash_sha3_update);
    SetHash("sha-3-384", 9, sizeof(sha3_ctx), SZ_SHA384, FALSE, SHA3384CalculateDigest, rhash_sha3_384_init, rhash_sha3_final, rhash_sha3_update);
    SetHash("sha-3-512", 9, sizeof(sha3_ctx), SZ_SHA512, FALSE, SHA3512CalculateDigest, rhash_sha3_512_init, rhash_sha3_final, rhash_sha3_update);
    SetHash("sha-3k-224", 9, sizeof(sha3_ctx), SZ_SHA224, FALSE, SHA3K224CalculateDigest, rhash_keccak_224_init, rhash_keccak_final, rhash_keccak_update);
    SetHash("sha-3k-256", 9, sizeof(sha3_ctx), SZ_SHA256, FALSE, SHA3K256CalculateDigest, rhash_keccak_256_init, rhash_keccak_final, rhash_keccak_update);
    SetHash("sha-3k-384", 9, sizeof(sha3_ctx), SZ_SHA384, FALSE, SHA3K384CalculateDigest, rhash_keccak_384_init, rhash_keccak_final, rhash_keccak_update);
    SetHash("sha-3k-512", 9, sizeof(sha3_ctx), SZ_SHA512, FALSE, SHA3K512CalculateDigest, rhash_keccak_512_init, rhash_keccak_final, rhash_keccak_update);
}
