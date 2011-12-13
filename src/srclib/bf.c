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
#include "apr_strings.h"
#include "bf.h"

int maxIndex;
uint32_t length;
uint64_t noOfAttempts;
BruteForceContext* ctx;

char* BruteForce(const uint32_t    passmin,
                 const uint32_t    passmax,
                 const char*       dict,
                 const char*       hash,
                 uint64_t*         attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 apr_pool_t*       pool)
{
    BruteForceContext local = { 0 };
    noOfAttempts = 0;

    if (passmax > INT_MAX / sizeof(int)) {
        CrtPrintf("Max string length is too big: %lu", passmax);
        return NULL;
    }

    local.Pass = (char*)apr_pcalloc(pool, passmax + 1);
    if (local.Pass == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax + 1, __FILE__, __LINE__);
        return NULL;
    }
    local.Indexes = (int*)apr_pcalloc(pool, passmax * sizeof(int));
    if (local.Indexes == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax * sizeof(int), __FILE__, __LINE__);
        return NULL;
    }

    
    local.Desired = PfnHashPrepare(hash, pool);
    local.PfnHashCompare = CompareHashAttempt;
    local.Dict = PrepareDictionary(dict);
    maxIndex = strlen(local.Dict) - 1;
    length = passmin;
    ctx = &local;
    for (; length <= passmax; ++length) {
        if (MakeAttempt(0)) {
            goto result;
        }
    }
    local.Pass = NULL;
result:
    *attempts = noOfAttempts;
    return local.Pass;
}

int MakeAttempt(const uint32_t pos)
{
    int i = 0;

    for (; i <= maxIndex; ++i) {
        ctx->Indexes[pos] = i;

        if (pos == length - 1) {
            uint32_t j = 0;
            for (; j < length; ++j) {
                ctx->Pass[j] = ctx->Dict[ctx->Indexes[j]];
            }
            ++noOfAttempts;

            if (ctx->PfnHashCompare(ctx->Desired, ctx->Pass, length)) {
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}

const char* PrepareDictionary(const char* dict)
{
    const char* digitsClass = NULL;
    const char* lowCaseClass = NULL;
    const char* upperCaseClass = NULL;
    
    digitsClass = strstr(dict, DIGITS_TPL);
    lowCaseClass = strstr(dict, LOW_CASE_TPL);
    upperCaseClass = strstr(dict, UPPER_CASE_TPL);

    if (!digitsClass && !lowCaseClass && !upperCaseClass) {
        return dict;
    }
    if (digitsClass && lowCaseClass && upperCaseClass) {
        return DIGITS LOW_CASE UPPER_CASE;
    }
    if (!digitsClass && lowCaseClass && upperCaseClass) {
        return LOW_CASE UPPER_CASE;
    }
    if (digitsClass && !lowCaseClass && upperCaseClass) {
        return DIGITS UPPER_CASE;
    }
    if (digitsClass && lowCaseClass && !upperCaseClass) {
        return DIGITS LOW_CASE;
    }
    if (digitsClass && !lowCaseClass && !upperCaseClass) {
        return DIGITS;
    }
    if (!digitsClass && !lowCaseClass && upperCaseClass) {
        return UPPER_CASE;
    }
    if (!digitsClass && lowCaseClass && !upperCaseClass) {
        return LOW_CASE;
    }

    return dict;
}