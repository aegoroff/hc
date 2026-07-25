#ifndef LINQ2HASH_HASHES_H_
#define LINQ2HASH_HASHES_H_

#include <stddef.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include "types.h"
#include "apr_pools.h"

#ifdef __cplusplus
extern "C" {
#endif

struct gpu_context_t;

typedef struct gpu_tread_ctx_t {
    unsigned char* variants_;
    unsigned char* dev_variants_;
    unsigned char* attempt_;
    unsigned char* result_;
    unsigned char* dev_result_;
    struct gpu_context_t* gpu_context_;
    size_t variants_size_;
    size_t variants_count_;
    uint32_t passmin_;
    uint32_t passmax_;
    uint32_t pass_length_;
    BOOL found_in_the_thread_;
    int max_gpu_blocks_number_;
    int max_threads_per_block_;
    int multiprocessor_count_;
    int device_ix_;
    BOOL use_wide_pass_;
    int max_threads_decrease_factor_;
    int comparisons_per_iteration_;
    apr_pool_t* pool_;
} gpu_tread_ctx_t;

typedef struct gpu_context_t {
    void (*pfn_run_)(void* context, const size_t dict_len, unsigned char* variants,
                     const size_t variants_size);
    void (*pfn_prepare_)(int device_ix, const unsigned char* dict, size_t dict_len,
                         const unsigned char* hash, gpu_tread_ctx_t* ctx);
    int max_threads_decrease_factor_;
    int comparisons_per_iteration_;
} gpu_context_t;

#ifdef __cplusplus
}
#endif

#endif /* LINQ2HASH_HASHES_H_ */
