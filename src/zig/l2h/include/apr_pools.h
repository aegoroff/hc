#ifndef L2H_ZIG_APR_POOLS_H
#define L2H_ZIG_APR_POOLS_H

/*
 * Minimal APR shim for the Zig l2h skeleton.
 *
 * The CMake-based build links the real Apache APR; the Zig port only needs the
 * type names referenced by src/l2h/frontend.h (currently just apr_pool_t in
 * fend_init). Task 8 will replace this once pool ownership is ported to Zig.
 */

typedef struct apr_pool_t apr_pool_t;

#endif /* L2H_ZIG_APR_POOLS_H */
