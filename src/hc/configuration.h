/*!
 * \brief   The file contains configuration module interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2016-09-13
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2024
 */

#ifndef HLINQ_CONFIGURATION_H_
#define HLINQ_CONFIGURATION_H_

#include "apr_pools.h"
#include "builtin.h"
#include "str.h"
#include "hash.h"
#include "file.h"
#include "dir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct configuration_ctx_t {
    int argc;
    const char* const* argv;
    apr_pool_t* pool;
    void (*pfn_on_string)(builtin_ctx_t* bctx, string_builtin_ctx_t* sctx, apr_pool_t* pool);
    void (*pfn_on_hash)(builtin_ctx_t* bctx, hash_builtin_ctx_t* hctx, apr_pool_t* pool);
    void (*pfn_on_file)(builtin_ctx_t* bctx, file_builtin_ctx_t* fctx, apr_pool_t* pool);
    void (*pfn_on_dir)(builtin_ctx_t* bctx, dir_builtin_ctx_t* dctx, apr_pool_t* pool);
} configuration_ctx_t;

void conf_run_app(configuration_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // HLINQ_CONFIGURATION_H_
