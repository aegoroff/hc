/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
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

#include "apr_pools.h"
#include "lib.h"
#include "../l2h/hashes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DIGITS "0123456789"
#define ASCII_TPL "ASCII"
#define DIGITS_TPL "0-9"
#define LOW_CASE "abcdefghijklmnopqrstuvwxyz"
#define LOW_CASE_TPL "a-z"
#define UPPER_CASE "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define UPPER_CASE_TPL "A-Z"
#define MAX_DEFAULT 10
#define GPU_ATTEMPT_SIZE 16

typedef struct gpu_tread_ctx_t {
    unsigned char* variants_;
    unsigned char* attempt_;
    wchar_t* wide_attempt_;
    unsigned char* result_;
    gpu_context_t* gpu_context_;
    uint64_t num_of_attempts_;
    size_t variants_size_;
    size_t variants_count_;
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t pass_length_;
    BOOL found_in_the_thread_;
    int max_gpu_blocks_number_;
    int max_threads_per_block_;
    int device_ix_;
    BOOL use_wide_attempt_;
} gpu_tread_ctx_t;

int bf_compare_hash_attempt(void* hash, const void* pass, const uint32_t length);

void bf_crack_hash(const char* dict,
                   const char* hash,
                   const uint32_t passmin,
                   uint32_t passmax,
                   apr_size_t hash_length,
                   void (*pfn_digest_function)(apr_byte_t* digest, const void* string, const apr_size_t input_len),
                   const BOOL no_probe,
                   const uint32_t num_of_threads,
                   const BOOL use_wide_pass,
                   const BOOL has_gpu_implementation,
                   gpu_context_t* gpu_context,
                   apr_pool_t* pool);

void* bf_create_digest(const char* hash, apr_pool_t* pool);
int bf_compare_hash(apr_byte_t* digest, const char* check_sum);

char* bf_brute_force(const uint32_t passmin,
                     const uint32_t passmax,
                     const char* dict,
                     const char* hash,
                     uint64_t* attempts,
                     void* (*pfn_hash_prepare)(const char* h, apr_pool_t* pool),
                     uint32_t num_of_threads,
                     const BOOL use_wide_pass,
                     BOOL has_gpu_implementation,
                     gpu_context_t* gpu_context,
                     apr_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_BF_H_
