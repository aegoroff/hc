/*!
 * \brief   The file contains string builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#include "str.h"
#include "encoding.h"

void str_run(string_builtin_ctx_t* ctx) {
    apr_byte_t* digest;
    apr_size_t sz;
    out_context_t o = { 0 };
    apr_pool_t* pool;
    hash_definition_t* hash;
    builtin_ctx_t* builtin_ctx;

    builtin_ctx = ctx->builtin_ctx_;

    pool = builtin_get_pool();
    hash = builtin_get_hash_definition();
    sz = hash->hash_length_;
    
    digest = apr_pcalloc(pool, sizeof(apr_byte_t) * sz);

    // some hashes like NTLM required unicode string so convert multi byte string to unicode one
    if (hash->use_wide_string_) {
        wchar_t* str = enc_from_ansi_to_unicode(ctx->string_, pool);
        hash->pfn_digest_(digest, str, wcslen(str) * sizeof(wchar_t));
    }
    else {
        hash->pfn_digest_(digest, ctx->string_, strlen(ctx->string_));
    }

    o.is_finish_line_ = TRUE;
    o.is_print_separator_ = FALSE;
    o.string_to_print_ = out_hash_to_string(digest, builtin_ctx->is_print_low_case_, sz, pool);
    builtin_ctx->pfn_output_(&o);
}
