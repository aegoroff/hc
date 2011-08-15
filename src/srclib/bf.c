/*!
 * \brief   The file contains brute force algorithm implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#include "targetver.h"
#include "bf.h"

char* BruteForce(uint32_t    passmin,
                 uint32_t    passmax,
                 const char* dict,
                 const char* hash,
                 uint64_t*   attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 apr_pool_t* pool)
{
    BruteForceContext ctx = { 0 };

    if (passmax > INT_MAX / sizeof(int)) {
        CrtPrintf("Max password length is too big: %lu", passmax);
        return NULL;
    }
    ctx.Pass = (char*)apr_pcalloc(pool, passmax + 1);
    if (ctx.Pass == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax + 1, __FILE__, __LINE__);
        return NULL;
    }
    ctx.Indexes = (int*)apr_pcalloc(pool, passmax * sizeof(int));
    if (ctx.Indexes == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax * sizeof(int), __FILE__, __LINE__);
        return NULL;
    }

    ctx.Attempts = attempts;
    ctx.Desired = PfnHashPrepare(hash, pool);
    ctx.MaxIndex = strlen(dict) - 1;
    ctx.Length = passmin;
    ctx.PfnHashCompare = CompareHashAttempt;
    ctx.Dict = dict;
    for (; ctx.Length <= passmax; ++(ctx.Length)) {
        if (MakeAttempt(0, &ctx)) {
            return ctx.Pass;
        }
    }
    return NULL;
}

int MakeAttempt(uint32_t pos, BruteForceContext* ctx)
{
    int i = 0;
    uint32_t j = 0;

    for (; i <= ctx->MaxIndex; ++i) {
        ctx->Indexes[pos] = i;

        if (pos == ctx->Length - 1) {
            for (j = 0; j < ctx->Length; ++j) {
                ctx->Pass[j] = ctx->Dict[ctx->Indexes[j]];
            }
            ++*(ctx->Attempts);

            if (ctx->PfnHashCompare(ctx->Desired, ctx->Pass, ctx->Length)) {
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1, ctx)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}
