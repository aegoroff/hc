/*!
 * \brief   The file contains brute force algorithm interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2011-03-04
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2016
 */

#ifndef LINQ2HASH_BF_H_
#define LINQ2HASH_BF_H_

#include <stdio.h>
#include "apr_pools.h"
#include "lib.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DIGITS "0123456789"
#define DIGITS_TPL "0-9"
#define LOW_CASE "abcdefghijklmnopqrstuvwxyz"
#define LOW_CASE_TPL "a-z"
#define UPPER_CASE "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define UPPER_CASE_TPL "A-Z"
#define MAX_DEFAULT 10

    typedef struct tread_ctx_t {
        uint32_t passmin_;
        uint32_t passmax_;
        uint32_t pass_length_;
        uint32_t thread_num_;
        char* pass_;
        wchar_t* wide_pass_;
        size_t* chars_indexes_;
        uint64_t num_of_attempts_;
        uint32_t num_of_threads;
        BOOL use_wide_pass_;
        BOOL found_in_the_thread_;
    } tread_ctx_t;


int bf_compare_hash_attempt(void* hash, const void* pass, const uint32_t length);

void bf_crack_hash(const char* dict,
                   const char* hash,
                   uint32_t passmin,
                   uint32_t passmax,
                   apr_size_t hash_length,
                   void (*digest_function)(apr_byte_t* digest, const void* string, const apr_size_t input_len),
                   BOOL no_probe,
                   uint32_t num_of_threads,
                   BOOL use_wide_pass,
                   BOOL has_gpu_implementation,
                   apr_pool_t* pool);

void* bf_create_digest(const char* hash, apr_pool_t* pool);
int bf_compare_hash(apr_byte_t* digest, const char* checkSum);

char* bf_brute_force(const uint32_t passmin,
                     const uint32_t passmax,
                     const char* dict,
                     const char* hash,
                     uint64_t* attempts,
                     void* (* pfn_hash_prepare)(const char* h, apr_pool_t* pool),
                     uint32_t num_of_threads,
                     BOOL use_wide_pass,
                     BOOL has_gpu_implementation,
                     apr_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_BF_H_
