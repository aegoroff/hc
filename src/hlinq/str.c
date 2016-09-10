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

void str_run(string_builtin_ctx_t* ctx, apr_pool_t* pool) {
    apr_byte_t* digest;
    apr_size_t sz;
    out_context_t o = { 0 };
    apr_pool_t* p;
    hash_definition_t* hash;

    if(!builtin_init(ctx->builtin_ctx_, pool)) {
        return;
    }

    p = builtin_get_pool();
    hash = builtin_get_hash_definition();
    sz = hash->hash_length_;
    
    digest = (apr_byte_t*)apr_pcalloc(p, sizeof(apr_byte_t) * sz);

    // some hashes like NTLM required unicode string so convert multi byte string to unicode one
    if (hash->use_wide_string_) {
        wchar_t* str = enc_from_ansi_to_unicode(ctx->string_, p);
        hash->pfn_digest_(digest, str, wcslen(str) * sizeof(wchar_t));
    }
    else {
        hash->pfn_digest_(digest, ctx->string_, strlen(ctx->string_));
    }

    o.is_finish_line_ = TRUE;
    o.is_print_separator_ = FALSE;
    o.string_to_print_ = out_hash_to_string(digest, ctx->builtin_ctx_->is_print_low_case_, sz, p);
    ctx->builtin_ctx_->pfn_output_(&o);
    builtin_close();
}
