/*!
 * \brief   The file contains hashes from libtom lib API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2015
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

/*
    hsh_ - public members
    prhsh_ - private members
*/

static apr_hash_t* ht_algorithms = NULL;
static apr_pool_t* pool;

#define DIGEST_BODY(ctx, init, update, close) \
    ctx CTX = { 0 }; \
    init(&CTX); \
    update(&CTX, input, input_len); \
    close(&CTX, digest);

#define ARRAY_INIT_SZ 50

/* compare function for qsort(3) */
static int prhsh_cmp_string(const void *v1, const void *v2)
{
    const char *s1 = *(const char**)v1;
    const char *s2 = *(const char**)v2;
    return strcmp(s1, s2);
}

const char* hsh_from_base64(const char* base64, apr_pool_t* pool) {
    unsigned char* d = apr_palloc(pool, SZ_SHA512);
    unsigned long len;
    base64_decode(base64, strlen(base64), d, &len);
    return out_hash_to_string(d, TRUE, len, pool);
}

void hsh_print_hashes(void)
{
    apr_hash_index_t* hi = NULL;
    int i = 0;
    apr_array_header_t* arr = apr_array_make(pool, ARRAY_INIT_SZ, sizeof(const char*));

    lib_printf("  Supported hash algorithms:");
    lib_new_line();
    for (hi = apr_hash_first(NULL, ht_algorithms); hi; hi = apr_hash_next(hi)) {
        const char* k;
        hash_definition_t* v;

        apr_hash_this(hi, (const void**)&k, NULL, (void**)&v);
        *(const char**)apr_array_push(arr) = k;
    }
    qsort(arr->elts, arr->nelts, arr->elt_size, prhsh_cmp_string);
    for (i = 0; i < arr->nelts; i++) {
        const char* elem = ((const char**)arr->elts)[i];
        lib_printf("    %s", elem);
        lib_new_line();
    }
}

void prhsh_set_hash(
    const char* alg,
    int weight,
    size_t context_size,
    apr_size_t length,
    BOOL use_wide_string,
    void (* pfn_digest)(apr_byte_t* digest, const void* input, const apr_size_t input_len),
    void (* pfn_init)(void* context),
    void (* pfn_final)(void* context, apr_byte_t* digest),
    void (* pfn_update)(void* context, const void* input, const apr_size_t input_len)
            );

hash_definition_t* hsh_get_hash(const char* str)
{
    return (hash_definition_t*)apr_hash_get(ht_algorithms, str, APR_HASH_KEY_STRING);
}


void prhsh_set_hash(
    const char* alg,
    int weight,
    size_t context_size,
    apr_size_t length,
    BOOL use_wide_string,
    void (* pfn_digest)(apr_byte_t* digest, const void* input, const apr_size_t input_len),
    void (* pfn_init)(void* context),
    void (* pfn_final)(void* context, apr_byte_t* digest),
    void (* pfn_update)(void* context, const void* input, const apr_size_t input_len)
            )
{
    hash_definition_t* hash = (hash_definition_t*)apr_pcalloc(pool, sizeof(hash_definition_t));
    hash->context_size_ = context_size;
    hash->pfn_init_ = pfn_init;
    hash->pfn_update_ = pfn_update;
    hash->pfn_final_ = pfn_final;
    hash->pfn_digest_ = pfn_digest;
    hash->hash_length_ = length;
    hash->weight_ = weight;
    hash->name_ = alg;
    hash->use_wide_string_ = use_wide_string;

    apr_hash_set(ht_algorithms, alg, APR_HASH_KEY_STRING, hash);
}

void prhsh_whirlpool_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_whirlpool_context, sph_whirlpool_init, sph_whirlpool, sph_whirlpool_close)
}

void prhsh_sha512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_sha512_context, sph_sha512_init, sph_sha512, sph_sha512_close)
}

void prhsh_sha384_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_sha384_context, sph_sha384_init, sph_sha384, sph_sha384_close)
}

void prhsh_sha256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_sha256_context, sph_sha256_init, sph_sha256, sph_sha256_close)
}

void prhsh_sha1_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_sha1_context, sph_sha1_init, sph_sha1, sph_sha1_close)
}

void prhsh_crc32_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(crc32_context_t, crc32_init, crc32_update, crc32_final)
}

void prhsh_md2_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_md2_context, sph_md2_init, sph_md2, sph_md2_close)
}

void prhsh_md4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_md4_context, sph_md4_init, sph_md4, sph_md4_close)
}

void prhsh_md5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_md5_context, sph_md5_init, sph_md5, sph_md5_close)
}

void prhsh_tiger_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_tiger_context, sph_tiger_init, sph_tiger, sph_tiger_close)
}

void prhsh_tiger2_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_tiger2_context, sph_tiger2_init, sph_tiger2, sph_tiger2_close)
}

void prhsh_sha224_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_sha224_context, sph_sha224_init, sph_sha224, sph_sha224_close)
}

void prhsh_rmd128_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_ripemd128_context, sph_ripemd128_init, sph_ripemd128, sph_ripemd128_close)
}

void prhsh_rmd160_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_ripemd160_context, sph_ripemd160_init, sph_ripemd160, sph_ripemd160_close)
}

void prhsh_libtom_calculate_digest(
    apr_byte_t* digest,
    const void* input,
    const apr_size_t input_len,
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
        PfnProcess(&context, input, input_len);
    }
    PfnDone(&context, digest);
}

void prhsh_rmd256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    prhsh_libtom_calculate_digest(digest, input, input_len, rmd256_init, rmd256_process, rmd256_done);
}

void prhsh_rmd320_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    prhsh_libtom_calculate_digest(digest, input, input_len, rmd320_init, rmd320_process, rmd320_done);
}

void prhsh_gost_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(gost_ctx, rhash_gost_cryptopro_init, rhash_gost_update, rhash_gost_final)
}

void prhsh_snefru128_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(snefru_ctx, rhash_snefru128_init, rhash_snefru_update, rhash_snefru_final)
}

void prhsh_snefru256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(snefru_ctx, rhash_snefru256_init, rhash_snefru_update, rhash_snefru_final)
}

void prhsh_tth_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(tth_ctx, rhash_tth_init, rhash_tth_update, rhash_tth_final)
}

void prhsh_haval128_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval128_3_init, sph_haval128_3, sph_haval128_3_close)
}

void prhsh_haval128_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval128_4_init, sph_haval128_4, sph_haval128_4_close)
}

void prhsh_haval128_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval128_5_init, sph_haval128_5, sph_haval128_5_close)
}

void prhsh_haval160_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval160_3_init, sph_haval160_3, sph_haval160_3_close)
}

void prhsh_haval160_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval160_4_init, sph_haval160_4, sph_haval160_4_close)
}

void prhsh_haval160_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval160_5_init, sph_haval160_5, sph_haval160_5_close)
}

void prhsh_haval192_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval192_3_init, sph_haval192_3, sph_haval192_3_close)
}

void prhsh_haval192_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval192_4_init, sph_haval192_4, sph_haval192_4_close)
}

void prhsh_haval192_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval192_5_init, sph_haval192_5, sph_haval192_5_close)
}

void prhsh_haval224_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval224_3_init, sph_haval224_3, sph_haval224_3_close)
}

void prhsh_haval224_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval224_4_init, sph_haval224_4, sph_haval224_4_close)
}

void prhsh_haval224_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval224_5_init, sph_haval224_5, sph_haval224_5_close)
}

void prhsh_haval256_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval256_3_init, sph_haval256_3, sph_haval256_3_close)
}

void prhsh_haval256_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval256_4_init, sph_haval256_4, sph_haval256_4_close)
}

void prhsh_haval256_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sph_haval_context, sph_haval256_5_init, sph_haval256_5, sph_haval256_5_close)
}

void prhsh_edonr256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(edonr_ctx, rhash_edonr256_init, rhash_edonr256_update, rhash_edonr256_final)
}

void prhsh_edonr512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(edonr_ctx, rhash_edonr512_init, rhash_edonr512_update, rhash_edonr512_final)
}

void prhsh_sha3224_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_224_init, rhash_sha3_update, rhash_sha3_final)
}

void prhsh_sha3256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_256_init, rhash_sha3_update, rhash_sha3_final)
}

void prhsh_sha3384_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_384_init, rhash_sha3_update, rhash_sha3_final)
}

void prhsh_sha3512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_sha3_512_init, rhash_sha3_update, rhash_sha3_final)
}

void prhsh_sha3_k224_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_224_init, rhash_keccak_update, rhash_keccak_final)
}

void prhsh_sha3_k256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_256_init, rhash_keccak_update, rhash_keccak_final)
}

void prhsh_sha3_k384_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_384_init, rhash_keccak_update, rhash_keccak_final)
}

void prhsh_sha3_k512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len)
{
    DIGEST_BODY(sha3_ctx, rhash_keccak_512_init, rhash_keccak_update, rhash_keccak_final)
}

/*
 * It MUST be last in the file so as not to declare internal functions in the header
 */
void hsh_initialize_hashes(apr_pool_t* p)
{
    ht_algorithms = apr_hash_make(p);
    pool = p;
    prhsh_set_hash("crc32", 2, sizeof(crc32_context_t), CRC32_HASH_SIZE, FALSE, prhsh_crc32_calculate_digest, crc32_init, crc32_final, crc32_update);
    prhsh_set_hash("md2", 3, sizeof(sph_md2_context), SZ_MD2, FALSE, prhsh_md2_calculate_digest, sph_md2_init, sph_md2_close, sph_md2);
    prhsh_set_hash("md4", 3, sizeof(sph_md4_context), SZ_MD4, FALSE, prhsh_md4_calculate_digest, sph_md4_init, sph_md4_close, sph_md4);
    prhsh_set_hash("ntlm", 3, sizeof(sph_md4_context), SZ_MD4, TRUE, prhsh_md4_calculate_digest, sph_md4_init, sph_md4_close, sph_md4);
    prhsh_set_hash("md5", 4, sizeof(sph_md5_context), SZ_MD5, FALSE, prhsh_md5_calculate_digest, sph_md5_init, sph_md5_close, sph_md5);
    prhsh_set_hash("sha1", 4, sizeof(sph_sha1_context), SZ_SHA1, FALSE, prhsh_sha1_calculate_digest, sph_sha1_init, sph_sha1_close, sph_sha1);
    prhsh_set_hash("sha224", 5, sizeof(sph_sha224_context), SZ_SHA224, FALSE, prhsh_sha224_calculate_digest, sph_sha224_init, sph_sha224_close, sph_sha224);
    prhsh_set_hash("sha256", 6, sizeof(sph_sha256_context), SZ_SHA256, FALSE, prhsh_sha256_calculate_digest, sph_sha256_init, sph_sha256_close, sph_sha256);
    prhsh_set_hash("sha384", 7, sizeof(sph_sha384_context), SZ_SHA384, FALSE, prhsh_sha384_calculate_digest, sph_sha384_init, sph_sha384_close, sph_sha384);
    prhsh_set_hash("sha512", 8, sizeof(sph_sha512_context), SZ_SHA512, FALSE, prhsh_sha512_calculate_digest, sph_sha512_init, sph_sha512_close, sph_sha512);
    prhsh_set_hash("ripemd128", 5, sizeof(sph_ripemd128_context), SZ_RIPEMD128, FALSE, prhsh_rmd128_calculate_digest, sph_ripemd128_init, sph_ripemd128_close, sph_ripemd128);
    prhsh_set_hash("ripemd160", 5, sizeof(sph_ripemd160_context), SZ_RIPEMD160, FALSE, prhsh_rmd160_calculate_digest, sph_ripemd160_init, sph_ripemd160_close, sph_ripemd160);
    prhsh_set_hash("ripemd256", 6, sizeof(hash_state), SZ_RIPEMD256, FALSE, prhsh_rmd256_calculate_digest, rmd256_init, rmd256_done, rmd256_process);
    prhsh_set_hash("ripemd320", 7, sizeof(hash_state), SZ_RIPEMD320, FALSE, prhsh_rmd320_calculate_digest, rmd320_init, rmd320_done, rmd320_process);
    prhsh_set_hash("tiger", 5, sizeof(sph_tiger_context), SZ_TIGER192, FALSE, prhsh_tiger_calculate_digest, sph_tiger_init, sph_tiger_close, sph_tiger);
    prhsh_set_hash("tiger2", 5, sizeof(sph_tiger2_context), SZ_TIGER192, FALSE, prhsh_tiger2_calculate_digest, sph_tiger2_init, sph_tiger2_close, sph_tiger2);
    prhsh_set_hash("whirlpool", 8, sizeof(sph_whirlpool_context), SZ_WHIRLPOOL, FALSE, prhsh_whirlpool_calculate_digest, sph_whirlpool_init, sph_whirlpool_close, sph_whirlpool);
    prhsh_set_hash("gost", 9, sizeof(gost_ctx), SZ_GOST, FALSE, prhsh_gost_calculate_digest, rhash_gost_cryptopro_init, rhash_gost_final, rhash_gost_update);
    prhsh_set_hash("snefru128", 10, sizeof(snefru_ctx), SZ_SNEFRU128, FALSE, prhsh_snefru128_calculate_digest, rhash_snefru128_init, rhash_snefru_final, rhash_snefru_update);
    prhsh_set_hash("snefru256", 10, sizeof(snefru_ctx), SZ_SNEFRU256, FALSE, prhsh_snefru256_calculate_digest, rhash_snefru256_init, rhash_snefru_final, rhash_snefru_update);
    prhsh_set_hash("tth", 5, sizeof(tth_ctx), SZ_TTH, FALSE, prhsh_tth_calculate_digest, rhash_tth_init, rhash_tth_final, rhash_tth_update);
    prhsh_set_hash("haval-128-3", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, prhsh_haval128_3_calculate_digest, sph_haval128_3_init, sph_haval128_3_close, sph_haval128_3);
    prhsh_set_hash("haval-128-4", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, prhsh_haval128_4_calculate_digest, sph_haval128_4_init, sph_haval128_4_close, sph_haval128_4);
    prhsh_set_hash("haval-128-5", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, prhsh_haval128_5_calculate_digest, sph_haval128_5_init, sph_haval128_5_close, sph_haval128_5);
    prhsh_set_hash("haval-160-3", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, prhsh_haval160_3_calculate_digest, sph_haval160_3_init, sph_haval160_3_close, sph_haval160_3);
    prhsh_set_hash("haval-160-4", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, prhsh_haval160_4_calculate_digest, sph_haval160_4_init, sph_haval160_4_close, sph_haval160_4);
    prhsh_set_hash("haval-160-5", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, prhsh_haval160_5_calculate_digest, sph_haval160_5_init, sph_haval160_5_close, sph_haval160_5);
    prhsh_set_hash("haval-192-3", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, prhsh_haval192_3_calculate_digest, sph_haval192_3_init, sph_haval192_3_close, sph_haval192_3);
    prhsh_set_hash("haval-192-4", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, prhsh_haval192_4_calculate_digest, sph_haval192_4_init, sph_haval192_4_close, sph_haval192_4);
    prhsh_set_hash("haval-192-5", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, prhsh_haval192_5_calculate_digest, sph_haval192_5_init, sph_haval192_5_close, sph_haval192_5);
    prhsh_set_hash("haval-224-3", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, prhsh_haval224_3_calculate_digest, sph_haval224_3_init, sph_haval224_3_close, sph_haval224_3);
    prhsh_set_hash("haval-224-4", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, prhsh_haval224_4_calculate_digest, sph_haval224_4_init, sph_haval224_4_close, sph_haval224_4);
    prhsh_set_hash("haval-224-5", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, prhsh_haval224_5_calculate_digest, sph_haval224_5_init, sph_haval224_5_close, sph_haval224_5);
    prhsh_set_hash("haval-256-3", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, prhsh_haval256_3_calculate_digest, sph_haval256_3_init, sph_haval256_3_close, sph_haval256_3);
    prhsh_set_hash("haval-256-4", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, prhsh_haval256_4_calculate_digest, sph_haval256_4_init, sph_haval256_4_close, sph_haval256_4);
    prhsh_set_hash("haval-256-5", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, prhsh_haval256_5_calculate_digest, sph_haval256_5_init, sph_haval256_5_close, sph_haval256_5);
    prhsh_set_hash("edonr256", 5, sizeof(edonr_ctx), SZ_EDONR256, FALSE, prhsh_edonr256_calculate_digest, rhash_edonr256_init, rhash_edonr256_final, rhash_edonr256_update);
    prhsh_set_hash("edonr512", 5, sizeof(edonr_ctx), SZ_EDONR512, FALSE, prhsh_edonr512_calculate_digest, rhash_edonr512_init, rhash_edonr512_final, rhash_edonr512_update);
    prhsh_set_hash("sha-3-224", 9, sizeof(sha3_ctx), SZ_SHA224, FALSE, prhsh_sha3224_calculate_digest, rhash_sha3_224_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3-256", 9, sizeof(sha3_ctx), SZ_SHA256, FALSE, prhsh_sha3256_calculate_digest, rhash_sha3_256_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3-384", 9, sizeof(sha3_ctx), SZ_SHA384, FALSE, prhsh_sha3384_calculate_digest, rhash_sha3_384_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3-512", 9, sizeof(sha3_ctx), SZ_SHA512, FALSE, prhsh_sha3512_calculate_digest, rhash_sha3_512_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3k-224", 9, sizeof(sha3_ctx), SZ_SHA224, FALSE, prhsh_sha3_k224_calculate_digest, rhash_keccak_224_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("sha-3k-256", 9, sizeof(sha3_ctx), SZ_SHA256, FALSE, prhsh_sha3_k256_calculate_digest, rhash_keccak_256_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("sha-3k-384", 9, sizeof(sha3_ctx), SZ_SHA384, FALSE, prhsh_sha3_k384_calculate_digest, rhash_keccak_384_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("sha-3k-512", 9, sizeof(sha3_ctx), SZ_SHA512, FALSE, prhsh_sha3_k512_calculate_digest, rhash_keccak_512_init, rhash_keccak_final, rhash_keccak_update);
}
