/*!
 * \brief   The file contains hash builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include "hash.h"
#include "bf.h"
#include "traverse.h"

#define MIN_DEFAULT 1
#define MAX_DEFAULT 10

static char* hash_alphabet = DIGITS LOW_CASE UPPER_CASE;

hash_definition_t* prhash_hash;
apr_size_t prhash_length;
apr_pool_t* hash_pool;

void hash_run(hash_builtin_ctx_t* ctx) {
    const char* hash_string;
    int passmin = ctx->min_ > 0 ? ctx->min_ : MIN_DEFAULT;
    int passmax = ctx->max_ > 0 ? ctx->max_ : MAX_DEFAULT;
    const char* dictionary = ctx->dictionary_ != NULL ? ctx->dictionary_ : hash_alphabet;
    hash_pool = builtin_get_pool();

    prhash_hash = builtin_get_hash_definition();
    prhash_length = prhash_hash->hash_length_;

    if(ctx->performance_) {
        apr_byte_t* digest = builtin_hash_from_string("12345");
        hash_string = out_hash_to_string(digest, FALSE, prhash_length, hash_pool);
    }
    else if(ctx->is_base64_) {
        hash_string = hsh_from_base64(ctx->hash_, hash_pool);
    }
    else {
        hash_string = ctx->hash_;
    }

    bf_crack_hash(dictionary,
                  hash_string,
                  passmin,
                  passmax,
                  prhash_length,
                  prhash_hash->pfn_digest_,
                  ctx->no_probe_,
                  ctx->threads_,
                  prhash_hash->use_wide_string_,
                  hash_pool);
}

int bf_compare_hash_attempt(void* hash, const void* pass, const uint32_t length) {
    apr_byte_t attempt[SZ_SHA512]; // hack to improve performance
    prhash_hash->pfn_digest_(attempt, pass, (apr_size_t)length);
    return fhash_compare_digests(attempt, hash);
}


void* bf_create_digest(const char* hash, apr_pool_t* p) {
    apr_byte_t* result = (apr_byte_t*)apr_pcalloc(p, prhash_length);
    fhash_to_digest(hash, result);
    return result;
}

int bf_compare_hash(apr_byte_t* digest, const char* checkSum) {
    apr_byte_t* bytes = (apr_byte_t*)apr_pcalloc(hash_pool, sizeof(apr_byte_t) * fhash_get_digest_size());

    fhash_to_digest(checkSum, bytes);
    return fhash_compare_digests(bytes, digest);
}