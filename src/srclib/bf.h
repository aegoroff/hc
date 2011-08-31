/*!
 * \brief   The file contains brute force algorithm interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2011
 */

#ifndef BF_HCALC_H_
#define BF_HCALC_H_

#include <stdio.h>
#include "apr_pools.h"
#include "lib.h"

#define DIGITS "0123456789"
#define DIGITS_TPL "0-9"
#define LOW_CASE "abcdefghijklmnopqrstuvwxyz"
#define LOW_CASE_TPL "a-z"
#define UPPER_CASE "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define UPPER_CASE_TPL "A-Z"

typedef struct BruteForceContext {
    uint32_t    Length;
    const char* Dict;
    int*        Indexes;
    char*       Pass;
    void*       Desired;
    uint64_t*   Attempts;
    int         MaxIndex;
    int (* PfnHashCompare)(void* hash, const char* pass, uint32_t length);
} BruteForceContext;

int CompareHashAttempt(void* hash, const char* pass, uint32_t length);

char* BruteForce(uint32_t    passmin,
                 uint32_t    passmax,
                 const char* dict,
                 const char* hash,
                 uint64_t*   attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 apr_pool_t* pool);

int MakeAttempt(uint32_t pos, BruteForceContext* ctx);

const char* PrepareDictionary(const char* dict);

#endif // BF_HCALC_H_
