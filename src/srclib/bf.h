/*!
 * \brief   The file contains brute force algorithm interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2013
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
#define MAX_DEFAULT 10

typedef struct BruteForceContext {
    const char* Dict;
    void*       Desired;
    int (* PfnHashCompare)(void* hash, const char* pass, const uint32_t length);
} BruteForceContext;

int CompareHashAttempt(void* hash, const char* pass, const uint32_t length);

void CrackHash(const char* dict,
               const char* hash,
               uint32_t    passmin,
               uint32_t    passmax,
               apr_size_t  hashLength,
               void (*digestFunction)(apr_byte_t* digest, const char* string, const apr_size_t inputLen),
               BOOL noProbe,
               apr_pool_t* pool);

void* CreateDigest(const char* hash, apr_pool_t* pool);
int CompareHash(apr_byte_t* digest, const char* checkSum);

char* BruteForce(const uint32_t    passmin,
                 const uint32_t    passmax,
                 const char* dict,
                 const char* hash,
                 uint64_t*   attempts,
                 void* (* PfnHashPrepare)(const char* hash, apr_pool_t* pool),
                 apr_pool_t* pool);
#endif // BF_HCALC_H_
