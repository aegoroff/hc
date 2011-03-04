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

int CompareHashAttempt(void* hash, const char* pass, uint32_t length);

char* BruteForce(uint32_t    passmin,
                 uint32_t    passmax,
                 const char* dict,
                 const char* hash,
                 uint64_t*   attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 apr_pool_t* pool);
int MakeAttempt(uint32_t pos,
                uint32_t length,
                const char* dict,
                int* indexes,
                char* pass,
                void* desired,
                uint64_t* attempts,
                int maxIndex,
                int (* PfnHashCompare)(void* hash, const char* pass, uint32_t length)
                );

#endif // BF_HCALC_H_
