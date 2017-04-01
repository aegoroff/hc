/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains string builtin implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-10
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2017
 */

#include "str.h"

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
    
    digest = builtin_hash_from_string(ctx->string_);

    o.is_finish_line_ = TRUE;
    o.is_print_separator_ = FALSE;
    o.string_to_print_ = out_hash_to_string(digest, builtin_ctx->is_print_low_case_, sz, pool);
    builtin_ctx->pfn_output_(&o);
}
