/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains hashes from libtom lib API implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include <tomcrypt.h>
#include <apr_tables.h>
#include <apr_hash.h>

#include "hashes.h"
#include "sph_md2.h"
#include "sph_ripemd.h"
#include "sph_tiger.h"
#include "sph_md4.h"
#include "crc32.h"
#include "gost.h"
#include "snefru.h"
#include "tth.h"
#include "sph_haval.h"
#include "edonr.h"
#include "sha3.h"
#include "output.h"

#include "openssl/sha.h"
#include "openssl/md5.h"
#include "openssl/whrlpool.h"
#include "openssl/ripemd.h"
#include "openssl/blake2_locl.h"
#include "intl.h"
#include "b64.h"
#include "sha1.h"
#include "sha256.h"
#include "md5.h"

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

#define DIGEST_BODY_CLOSE_REVERSE(ctx, init, update, close) \
    ctx CTX = { 0 }; \
    init(&CTX); \
    update(&CTX, input, input_len); \
    close(digest, &CTX);

#define ARRAY_INIT_SZ 50

/* compare function for qsort(3) */
static int prhsh_cmp_string(const void* v1, const void* v2) {
    const char* s1 = *(const char**)v1;
    const char* s2 = *(const char**)v2;
    return strcmp(s1, s2);
}

const char* hsh_from_base64(const char* base64, apr_pool_t* p) {
    size_t len = 0;
    apr_byte_t* result = (apr_byte_t*)b64_decode(base64, strlen(base64), &len, p);
    if(result == NULL) {
        return "";
    }
    return out_hash_to_string(result, TRUE, len, p);
}

void hsh_print_hashes(void) {
    apr_hash_index_t* hi = NULL;
    int i = 0;
    apr_array_header_t* arr = apr_array_make(pool, ARRAY_INIT_SZ, sizeof(const char*));

    lib_printf(_("  Supported hash algorithms:"));
    lib_new_line();
    for(hi = apr_hash_first(NULL, ht_algorithms); hi; hi = apr_hash_next(hi)) {
        const char* k;
        hash_definition_t* v;

        apr_hash_this(hi, (const void**)&k, NULL, (void**)&v);
        *(const char**)apr_array_push(arr) = k;
    }
    qsort(arr->elts, arr->nelts, arr->elt_size, prhsh_cmp_string);
    for(i = 0; i < arr->nelts; i++) {
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
    BOOL has_gpu_implementation,
    void (* pfn_digest)(apr_byte_t* digest, const void* input, const apr_size_t input_len),
    void (* pfn_init)(void* context),
    void (* pfn_final)(void* context, apr_byte_t* digest),
    void (* pfn_update)(void* context, const void* input, const apr_size_t input_len)
);

void prhsh_set_gpu_functions(
    const char* alg,
    void (*pfn_run)(void* context, const size_t dict_len, unsigned char* variants,
                    const size_t variants_size),
    void (*pfn_prepare)(int device_ix, const unsigned char* dict, size_t dict_len,
                         const unsigned char* hash, unsigned char** variants, size_t variants_len),
    void (*pfn_cleanup)(void* context));


hash_definition_t* hsh_get_hash(const char* str) {
    return (hash_definition_t*)apr_hash_get(ht_algorithms, str, APR_HASH_KEY_STRING);
}

static void prhsh_set_hash(
    const char* alg,
    int weight,
    size_t context_size,
    apr_size_t length,
    BOOL use_wide_string,
    BOOL has_gpu_implementation,
    void (* pfn_digest)(apr_byte_t* digest, const void* input, const apr_size_t input_len),
    void (* pfn_init)(void* context),
    void (* pfn_final)(void* context, apr_byte_t* digest),
    void (* pfn_update)(void* context, const void* input, const apr_size_t input_len)
) {
    hash_definition_t* hash = (hash_definition_t*)apr_pcalloc(pool, sizeof(hash_definition_t));
    hash->context_size_ = context_size;
    hash->pfn_init_ = pfn_init;
    hash->pfn_update_ = pfn_update;
    hash->pfn_final_ = pfn_final;
    hash->pfn_digest_ = pfn_digest;
    hash->hash_length_ = length;
    hash->weight_ = weight;
    hash->name_ = alg;
    hash->has_gpu_implementation_ = has_gpu_implementation;
    hash->use_wide_string_ = use_wide_string;
    hash->gpu_context_ = (gpu_context_t*)apr_pcalloc(pool, sizeof(gpu_context_t));

    apr_hash_set(ht_algorithms, alg, APR_HASH_KEY_STRING, hash);
}

static void prhsh_set_gpu_functions(const char* alg,
                                    void (* pfn_run)(void* context, const size_t dict_len, unsigned char* variants,
                                                     const size_t variants_size),
                                    void (* pfn_prepare)(int device_ix, const unsigned char* dict, size_t dict_len,
                                                         const unsigned char* hash, unsigned
                                                         char** variants, size_t variants_len),
                                    void (* pfn_cleanup)(void* context)) {

    hash_definition_t* h = hsh_get_hash(alg);
    if(h == NULL) {
        lib_printf(_("Unknown hash: %s"), alg);
        lib_new_line();
        return;
    }
    h->gpu_context_->pfn_run_ = pfn_run;
    h->gpu_context_->pfn_prepare_ = pfn_prepare;
    h->gpu_context_->pfn_cleanup_ = pfn_cleanup;
}

static void prhsh_whirlpool_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(WHIRLPOOL_CTX, WHIRLPOOL_Init, WHIRLPOOL_Update, WHIRLPOOL_Final)
}

static void prhsh_whirlpool_final(void* context, apr_byte_t* digest) {
    WHIRLPOOL_Final(digest, context);
}

static void prhsh_sha512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(SHA512_CTX, SHA512_Init, SHA512_Update, SHA512_Final)
}

static void prhsh_sha384_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(SHA512_CTX, SHA384_Init, SHA384_Update, SHA384_Final)
}

static void prhsh_sha256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(SHA256_CTX, SHA256_Init, SHA256_Update, SHA256_Final)
}

static void prhsh_sha224_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(SHA256_CTX, SHA224_Init, SHA224_Update, SHA224_Final)
}

static void prhsh_sha1_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(SHA_CTX, SHA1_Init, SHA1_Update, SHA1_Final)
}

static void prhsh_md5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(MD5_CTX, MD5_Init, MD5_Update, MD5_Final)
}

static void prhsh_sha1_final(void* context, apr_byte_t* digest) {
    SHA1_Final(digest, context);
}

static void prhsh_sha224_final(void* context, apr_byte_t* digest) {
    SHA224_Final(digest, context);
}

static void prhsh_sha256_final(void* context, apr_byte_t* digest) {
    SHA256_Final(digest, context);
}

static void prhsh_sha384_final(void* context, apr_byte_t* digest) {
    SHA384_Final(digest, context);
}

static void prhsh_sha512_final(void* context, apr_byte_t* digest) {
    SHA512_Final(digest, context);
}

static void prhsh_md5_final(void* context, apr_byte_t* digest) {
    MD5_Final(digest, context);
}

static void prhsh_crc32_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(crc32_context_t, crc32_init, crc32_update, crc32_final)
}

static void prhsh_md2_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_md2_context, sph_md2_init, sph_md2, sph_md2_close)
}

static void prhsh_md4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_md4_context, sph_md4_init, sph_md4, sph_md4_close)
}

static void prhsh_tiger_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_tiger_context, sph_tiger_init, sph_tiger, sph_tiger_close)
}

static void prhsh_tiger2_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_tiger2_context, sph_tiger2_init, sph_tiger2, sph_tiger2_close)
}

static void prhsh_rmd128_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_ripemd128_context, sph_ripemd128_init, sph_ripemd128, sph_ripemd128_close)
}

static void prhsh_rmd160_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(RIPEMD160_CTX, RIPEMD160_Init, RIPEMD160_Update, RIPEMD160_Final)
}

static void prhsh_rmd160_final(void* context, apr_byte_t* digest) {
    RIPEMD160_Final(digest, context);
}

static void prhsh_blake2b_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(BLAKE2B_CTX, BLAKE2b_Init, BLAKE2b_Update, BLAKE2b_Final)
}

static void prhsh_blake2s_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY_CLOSE_REVERSE(BLAKE2S_CTX, BLAKE2s_Init, BLAKE2s_Update, BLAKE2s_Final)
}

static void prhsh_blake2b_final(void* context, apr_byte_t* digest) {
    BLAKE2b_Final(digest, context);
}

static void prhsh_blake2s_final(void* context, apr_byte_t* digest) {
    BLAKE2s_Final(digest, context);
}

static void prhsh_libtom_calculate_digest(
    apr_byte_t* digest,
    const void* input,
    const apr_size_t input_len,
    int (* pfn_init)(hash_state* md),
    int (* pfn_process)(hash_state* md, const unsigned char* in, unsigned long inlen),
    int (* pfn_done)(hash_state* md, unsigned char* hash)
) {
    hash_state context = { 0 };

    pfn_init(&context);

    if(input == NULL) {
        pfn_process(&context, "", 0);
    } else {
        pfn_process(&context, input, input_len);
    }
    pfn_done(&context, digest);
}

static void prhsh_rmd256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    prhsh_libtom_calculate_digest(digest, input, input_len, rmd256_init, rmd256_process, rmd256_done);
}

static void prhsh_rmd320_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    prhsh_libtom_calculate_digest(digest, input, input_len, rmd320_init, rmd320_process, rmd320_done);
}

static void prhsh_gost_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(gost_ctx, rhash_gost_cryptopro_init, rhash_gost_update, rhash_gost_final)
}

static void prhsh_snefru128_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(snefru_ctx, rhash_snefru128_init, rhash_snefru_update, rhash_snefru_final)
}

static void prhsh_snefru256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(snefru_ctx, rhash_snefru256_init, rhash_snefru_update, rhash_snefru_final)
}

static void prhsh_tth_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(tth_ctx, rhash_tth_init, rhash_tth_update, rhash_tth_final)
}

static void prhsh_haval128_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval128_3_init, sph_haval128_3, sph_haval128_3_close)
}

static void prhsh_haval128_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval128_4_init, sph_haval128_4, sph_haval128_4_close)
}

static void prhsh_haval128_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval128_5_init, sph_haval128_5, sph_haval128_5_close)
}

static void prhsh_haval160_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval160_3_init, sph_haval160_3, sph_haval160_3_close)
}

static void prhsh_haval160_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval160_4_init, sph_haval160_4, sph_haval160_4_close)
}

static void prhsh_haval160_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval160_5_init, sph_haval160_5, sph_haval160_5_close)
}

static void prhsh_haval192_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval192_3_init, sph_haval192_3, sph_haval192_3_close)
}

static void prhsh_haval192_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval192_4_init, sph_haval192_4, sph_haval192_4_close)
}

static void prhsh_haval192_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval192_5_init, sph_haval192_5, sph_haval192_5_close)
}

static void prhsh_haval224_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval224_3_init, sph_haval224_3, sph_haval224_3_close)
}

static void prhsh_haval224_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval224_4_init, sph_haval224_4, sph_haval224_4_close)
}

static void prhsh_haval224_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval224_5_init, sph_haval224_5, sph_haval224_5_close)
}

static void prhsh_haval256_3_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval256_3_init, sph_haval256_3, sph_haval256_3_close)
}

static void prhsh_haval256_4_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval256_4_init, sph_haval256_4, sph_haval256_4_close)
}

static void prhsh_haval256_5_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sph_haval_context, sph_haval256_5_init, sph_haval256_5, sph_haval256_5_close)
}

static void prhsh_edonr256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(edonr_ctx, rhash_edonr256_init, rhash_edonr256_update, rhash_edonr256_final)
}

static void prhsh_edonr512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(edonr_ctx, rhash_edonr512_init, rhash_edonr512_update, rhash_edonr512_final)
}

static void prhsh_sha3224_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_sha3_224_init, rhash_sha3_update, rhash_sha3_final)
}

static void prhsh_sha3256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_sha3_256_init, rhash_sha3_update, rhash_sha3_final)
}

static void prhsh_sha3384_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_sha3_384_init, rhash_sha3_update, rhash_sha3_final)
}

static void prhsh_sha3512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_sha3_512_init, rhash_sha3_update, rhash_sha3_final)
}

static void prhsh_sha3_k224_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_keccak_224_init, rhash_keccak_update, rhash_keccak_final)
}

static void prhsh_sha3_k256_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_keccak_256_init, rhash_keccak_update, rhash_keccak_final)
}

static void prhsh_sha3_k384_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_keccak_384_init, rhash_keccak_update, rhash_keccak_final)
}

static void prhsh_sha3_k512_calculate_digest(apr_byte_t* digest, const void* input, const apr_size_t input_len) {
    DIGEST_BODY(sha3_ctx, rhash_keccak_512_init, rhash_keccak_update, rhash_keccak_final)
}

/*
 * It MUST be last in the file so as not to declare internal functions in the header
 */
void hsh_initialize_hashes(apr_pool_t* p) {
    ht_algorithms = apr_hash_make(p);
    pool = p;
    prhsh_set_hash("crc32", 2, sizeof(crc32_context_t), CRC32_HASH_SIZE, FALSE, FALSE, prhsh_crc32_calculate_digest, crc32_init, crc32_final, crc32_update);
    prhsh_set_hash("md2", 3, sizeof(sph_md2_context), SZ_MD2, FALSE, FALSE, prhsh_md2_calculate_digest, sph_md2_init, sph_md2_close, sph_md2);
    prhsh_set_hash("md4", 3, sizeof(sph_md4_context), SZ_MD4, FALSE, FALSE, prhsh_md4_calculate_digest, sph_md4_init, sph_md4_close, sph_md4);
    prhsh_set_hash("ntlm", 3, sizeof(sph_md4_context), SZ_MD4, TRUE, FALSE, prhsh_md4_calculate_digest, sph_md4_init, sph_md4_close, sph_md4);
    prhsh_set_hash("md5", 4, sizeof(MD5_CTX), SZ_MD5, FALSE, TRUE, prhsh_md5_calculate_digest, MD5_Init, prhsh_md5_final, MD5_Update);
    prhsh_set_hash("sha1", 4, sizeof(SHA_CTX), SZ_SHA1, FALSE, TRUE, prhsh_sha1_calculate_digest, SHA1_Init, prhsh_sha1_final, SHA1_Update);
    prhsh_set_hash("sha224", 5, sizeof(SHA256_CTX), SZ_SHA224, FALSE, FALSE, prhsh_sha224_calculate_digest, SHA224_Init, prhsh_sha224_final, SHA224_Update);
    prhsh_set_hash("sha256", 6, sizeof(SHA256_CTX), SZ_SHA256, FALSE, TRUE, prhsh_sha256_calculate_digest, SHA256_Init, prhsh_sha256_final, SHA256_Update);
    prhsh_set_hash("sha384", 7, sizeof(SHA512_CTX), SZ_SHA384, FALSE, FALSE, prhsh_sha384_calculate_digest, SHA384_Init, prhsh_sha384_final, SHA384_Update);
    prhsh_set_hash("sha512", 8, sizeof(SHA512_CTX), SZ_SHA512, FALSE, FALSE, prhsh_sha512_calculate_digest, SHA512_Init, prhsh_sha512_final, SHA512_Update);
    prhsh_set_hash("ripemd128", 5, sizeof(sph_ripemd128_context), SZ_RIPEMD128, FALSE, FALSE, prhsh_rmd128_calculate_digest, sph_ripemd128_init, sph_ripemd128_close, sph_ripemd128);
    prhsh_set_hash("ripemd160", 5, sizeof(RIPEMD160_CTX), SZ_RIPEMD160, FALSE, FALSE, prhsh_rmd160_calculate_digest, RIPEMD160_Init, prhsh_rmd160_final, RIPEMD160_Update);
    prhsh_set_hash("ripemd256", 6, sizeof(hash_state), SZ_RIPEMD256, FALSE, FALSE, prhsh_rmd256_calculate_digest, rmd256_init, rmd256_done, rmd256_process);
    prhsh_set_hash("ripemd320", 7, sizeof(hash_state), SZ_RIPEMD320, FALSE, FALSE, prhsh_rmd320_calculate_digest, rmd320_init, rmd320_done, rmd320_process);
    prhsh_set_hash("tiger", 5, sizeof(sph_tiger_context), SZ_TIGER192, FALSE, FALSE, prhsh_tiger_calculate_digest, sph_tiger_init, sph_tiger_close, sph_tiger);
    prhsh_set_hash("tiger2", 5, sizeof(sph_tiger2_context), SZ_TIGER192, FALSE, FALSE, prhsh_tiger2_calculate_digest, sph_tiger2_init, sph_tiger2_close, sph_tiger2);
    prhsh_set_hash("whirlpool", 8, sizeof(WHIRLPOOL_CTX), SZ_WHIRLPOOL, FALSE, FALSE, prhsh_whirlpool_calculate_digest, WHIRLPOOL_Init, prhsh_whirlpool_final, WHIRLPOOL_Update);
    prhsh_set_hash("gost", 9, sizeof(gost_ctx), SZ_GOST, FALSE, FALSE, prhsh_gost_calculate_digest, rhash_gost_cryptopro_init, rhash_gost_final, rhash_gost_update);
    prhsh_set_hash("snefru128", 10, sizeof(snefru_ctx), SZ_SNEFRU128, FALSE, FALSE, prhsh_snefru128_calculate_digest, rhash_snefru128_init, rhash_snefru_final, rhash_snefru_update);
    prhsh_set_hash("snefru256", 10, sizeof(snefru_ctx), SZ_SNEFRU256, FALSE, FALSE, prhsh_snefru256_calculate_digest, rhash_snefru256_init, rhash_snefru_final, rhash_snefru_update);
    prhsh_set_hash("tth", 5, sizeof(tth_ctx), SZ_TTH, FALSE, FALSE, prhsh_tth_calculate_digest, rhash_tth_init, rhash_tth_final, rhash_tth_update);
    prhsh_set_hash("haval-128-3", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, FALSE, prhsh_haval128_3_calculate_digest, sph_haval128_3_init, sph_haval128_3_close, sph_haval128_3);
    prhsh_set_hash("haval-128-4", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, FALSE, prhsh_haval128_4_calculate_digest, sph_haval128_4_init, sph_haval128_4_close, sph_haval128_4);
    prhsh_set_hash("haval-128-5", 5, sizeof(sph_haval_context), SZ_HAVAL128, FALSE, FALSE, prhsh_haval128_5_calculate_digest, sph_haval128_5_init, sph_haval128_5_close, sph_haval128_5);
    prhsh_set_hash("haval-160-3", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, FALSE, prhsh_haval160_3_calculate_digest, sph_haval160_3_init, sph_haval160_3_close, sph_haval160_3);
    prhsh_set_hash("haval-160-4", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, FALSE, prhsh_haval160_4_calculate_digest, sph_haval160_4_init, sph_haval160_4_close, sph_haval160_4);
    prhsh_set_hash("haval-160-5", 5, sizeof(sph_haval_context), SZ_HAVAL160, FALSE, FALSE, prhsh_haval160_5_calculate_digest, sph_haval160_5_init, sph_haval160_5_close, sph_haval160_5);
    prhsh_set_hash("haval-192-3", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, FALSE, prhsh_haval192_3_calculate_digest, sph_haval192_3_init, sph_haval192_3_close, sph_haval192_3);
    prhsh_set_hash("haval-192-4", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, FALSE, prhsh_haval192_4_calculate_digest, sph_haval192_4_init, sph_haval192_4_close, sph_haval192_4);
    prhsh_set_hash("haval-192-5", 5, sizeof(sph_haval_context), SZ_HAVAL192, FALSE, FALSE, prhsh_haval192_5_calculate_digest, sph_haval192_5_init, sph_haval192_5_close, sph_haval192_5);
    prhsh_set_hash("haval-224-3", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, FALSE, prhsh_haval224_3_calculate_digest, sph_haval224_3_init, sph_haval224_3_close, sph_haval224_3);
    prhsh_set_hash("haval-224-4", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, FALSE, prhsh_haval224_4_calculate_digest, sph_haval224_4_init, sph_haval224_4_close, sph_haval224_4);
    prhsh_set_hash("haval-224-5", 5, sizeof(sph_haval_context), SZ_HAVAL224, FALSE, FALSE, prhsh_haval224_5_calculate_digest, sph_haval224_5_init, sph_haval224_5_close, sph_haval224_5);
    prhsh_set_hash("haval-256-3", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, FALSE, prhsh_haval256_3_calculate_digest, sph_haval256_3_init, sph_haval256_3_close, sph_haval256_3);
    prhsh_set_hash("haval-256-4", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, FALSE, prhsh_haval256_4_calculate_digest, sph_haval256_4_init, sph_haval256_4_close, sph_haval256_4);
    prhsh_set_hash("haval-256-5", 5, sizeof(sph_haval_context), SZ_HAVAL256, FALSE, FALSE, prhsh_haval256_5_calculate_digest, sph_haval256_5_init, sph_haval256_5_close, sph_haval256_5);
    prhsh_set_hash("edonr256", 5, sizeof(edonr_ctx), SZ_EDONR256, FALSE, FALSE, prhsh_edonr256_calculate_digest, rhash_edonr256_init, rhash_edonr256_final, rhash_edonr256_update);
    prhsh_set_hash("edonr512", 5, sizeof(edonr_ctx), SZ_EDONR512, FALSE, FALSE, prhsh_edonr512_calculate_digest, rhash_edonr512_init, rhash_edonr512_final, rhash_edonr512_update);
    prhsh_set_hash("sha-3-224", 9, sizeof(sha3_ctx), SZ_SHA224, FALSE, FALSE, prhsh_sha3224_calculate_digest, rhash_sha3_224_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3-256", 9, sizeof(sha3_ctx), SZ_SHA256, FALSE, FALSE, prhsh_sha3256_calculate_digest, rhash_sha3_256_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3-384", 9, sizeof(sha3_ctx), SZ_SHA384, FALSE, FALSE, prhsh_sha3384_calculate_digest, rhash_sha3_384_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3-512", 9, sizeof(sha3_ctx), SZ_SHA512, FALSE, FALSE, prhsh_sha3512_calculate_digest, rhash_sha3_512_init, rhash_sha3_final, rhash_sha3_update);
    prhsh_set_hash("sha-3k-224", 9, sizeof(sha3_ctx), SZ_SHA224, FALSE, FALSE, prhsh_sha3_k224_calculate_digest, rhash_keccak_224_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("sha-3k-256", 9, sizeof(sha3_ctx), SZ_SHA256, FALSE, FALSE, prhsh_sha3_k256_calculate_digest, rhash_keccak_256_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("sha-3k-384", 9, sizeof(sha3_ctx), SZ_SHA384, FALSE, FALSE, prhsh_sha3_k384_calculate_digest, rhash_keccak_384_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("sha-3k-512", 9, sizeof(sha3_ctx), SZ_SHA512, FALSE, FALSE, prhsh_sha3_k512_calculate_digest, rhash_keccak_512_init, rhash_keccak_final, rhash_keccak_update);
    prhsh_set_hash("blake2b", 8, sizeof(BLAKE2B_CTX), SZ_BLAKE2B, FALSE, FALSE, prhsh_blake2b_calculate_digest, BLAKE2b_Init, prhsh_blake2b_final, BLAKE2b_Update);
    prhsh_set_hash("blake2s", 6, sizeof(BLAKE2S_CTX), SZ_BLAKE2S, FALSE, FALSE, prhsh_blake2s_calculate_digest, BLAKE2s_Init, prhsh_blake2s_final, BLAKE2s_Update);

    // Init GPU functions
    prhsh_set_gpu_functions("sha1", sha1_run_on_gpu, sha1_on_gpu_prepare, sha1_on_gpu_cleanup);
    prhsh_set_gpu_functions("md5", md5_run_on_gpu, md5_on_gpu_prepare, md5_on_gpu_cleanup);
    prhsh_set_gpu_functions("sha256", sha256_run_on_gpu, sha256_on_gpu_prepare, sha256_on_gpu_cleanup);
}
