#ifndef HC_APR_SHIM_APR_POOLS_H
#define HC_APR_SHIM_APR_POOLS_H

/* See apr.h: translate-c-only shim. Mirrors the pool API bf.zig uses. */

#include "apr.h"

typedef struct apr_pool_t apr_pool_t;
typedef apr_status_t (*apr_abortfunc_t)(apr_int32_t);

/* 4th param is apr_allocator_t * in real APR; a void pointer is ABI-identical
 * and bf.zig only ever passes null. */
apr_status_t apr_pool_create_ex(apr_pool_t **newpool,
                                apr_pool_t *parent,
                                apr_abortfunc_t abort_fn,
                                void *allocator);

void apr_pool_destroy(apr_pool_t *pool);

#endif /* HC_APR_SHIM_APR_POOLS_H */
