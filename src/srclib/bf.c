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
    char* pass = NULL;
    int* indexes = NULL;
    uint32_t passLength = passmin;
    void* desired = NULL;
    int maxIndex = strlen(dict) - 1;

    if (passmax > INT_MAX / sizeof(int)) {
        CrtPrintf("Max password length is too big: %lu", passmax);
        return NULL;
    }
    pass = (char*)apr_pcalloc(pool, passmax + 1);
    if (pass == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax + 1, __FILE__, __LINE__);
        return NULL;
    }
    indexes = (int*)apr_pcalloc(pool, passmax * sizeof(int));
    if (indexes == NULL) {
        CrtPrintf(ALLOCATION_FAILURE_MESSAGE, passmax * sizeof(int), __FILE__, __LINE__);
        return NULL;
    }

    desired = PfnHashPrepare(hash, pool);
    for (; passLength <= passmax; ++passLength) {
        if (MakeAttempt(0, passLength, dict, indexes, pass, desired, attempts, maxIndex, CompareHashAttempt)) {
            return pass;
        }
    }
    return NULL;
}

int MakeAttempt(uint32_t pos, uint32_t length, const char* dict, int* indexes, char* pass,
                void* desired, uint64_t* attempts, int maxIndex, int (* PfnHashCompare)(void* hash, const char* pass, uint32_t length))
{
    int i = 0;
    uint32_t j = 0;

    for (; i <= maxIndex; ++i) {
        indexes[pos] = i;

        if (pos == length - 1) {
            for (j = 0; j < length; ++j) {
                pass[j] = dict[indexes[j]];
            }
            ++*attempts;

            if (PfnHashCompare(desired, pass, length)) {
                return TRUE;
            }
        } else {
            if (MakeAttempt(pos + 1, length, dict, indexes, pass, desired, attempts, maxIndex, PfnHashCompare)) {
                return TRUE;
            }
        }
    }
    return FALSE;
}
