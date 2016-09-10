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
