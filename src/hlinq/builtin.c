/*!
 * \brief   The file contains common builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include "builtin.h"
#include "../linq2hash/hashes.h"
#include "encoding.h"

apr_pool_t* builtin_pool = NULL;
hash_definition_t* builtin_hash = NULL;

BOOL builtin_init(builtin_ctx_t* ctx, apr_pool_t* root) {
    apr_pool_create(&builtin_pool, root);
    hsh_initialize_hashes(builtin_pool);
    builtin_hash = hsh_get_hash(ctx->hash_algorithm_);
    if(builtin_hash == NULL) {
        lib_printf("Unknown hash: %s" NEW_LINE, ctx->hash_algorithm_);
        builtin_close();
        return FALSE;
    }
    return TRUE;
}

void builtin_close() {
    apr_pool_destroy(builtin_pool);
    builtin_hash = NULL;
}

apr_pool_t* builtin_get_pool() {
    return builtin_pool;
}

hash_definition_t* builtin_get_hash_definition() {
    return builtin_hash;
}

void builtin_run(builtin_ctx_t* ctx, void* concrete_ctx, void (*pfn_action)(void* concrete_builtin_ctx), apr_pool_t* root) {
    if(!builtin_init(ctx, root)) {
        return;
    }

    pfn_action(concrete_ctx);

    builtin_close();
}

apr_byte_t* builtin_hash_from_string(const char* string) {
    apr_byte_t* digest;

    digest = apr_pcalloc(builtin_pool, sizeof(apr_byte_t) * builtin_hash->hash_length_);

    // some hashes like NTLM required unicode string so convert multi byte string to unicode one
    if(builtin_hash->use_wide_string_) {
        wchar_t* str = enc_from_ansi_to_unicode(string, builtin_pool);
        builtin_hash->pfn_digest_(digest, str, wcslen(str) * sizeof(wchar_t));
    }
    else {
        builtin_hash->pfn_digest_(digest, string, strlen(string));
    }
    return digest;
}

apr_size_t fhash_get_digest_size() {
    return builtin_hash->hash_length_;
}