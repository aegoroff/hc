
/*!
 * \brief   The file contains hashes from libtom lib API interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2013-06-18
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2025
 */

#ifndef LINQ2HASH_HASHES_H_
#define LINQ2HASH_HASHES_H_

#include <apr.h>
#include <apr_pools.h>
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
   Hash sizes:
 */

#define SZ_GOST         32
#define SZ_WHIRLPOOL    64
#define SZ_SHA512       64
#define SZ_SHA384       48
#define SZ_RIPEMD320    40
#define SZ_SHA256       32
#define SZ_RIPEMD256    32
#define SZ_SHA224       28
#define SZ_TIGER192     24
#define SZ_SHA1         20
#define SZ_RIPEMD160    20
#define SZ_RIPEMD128    16
#define SZ_MD5          16
#define SZ_MD4          16
#define SZ_MD2          16
#define SZ_SNEFRU128    16
#define SZ_SNEFRU256    32
#define SZ_TTH          24
#define SZ_HAVAL128     16
#define SZ_HAVAL160     20
#define SZ_HAVAL192     24
#define SZ_HAVAL224     28
#define SZ_HAVAL256     32
#define SZ_EDONR256     32
#define SZ_EDONR512     64
#define SZ_BLAKE2B      64
#define SZ_BLAKE2S      32
#define SZ_BLAKE3       32

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

typedef struct hash_definition_t {
    size_t context_size_;
    apr_size_t hash_length_;
    const char* name_;
    void (* pfn_digest_)(apr_byte_t* digest, const void* input,
                         const apr_size_t input_len);
    void (* pfn_init_)(void* context);
    void (* pfn_final_)(void* context, apr_byte_t* digest);
    void (* pfn_update_)(void* context, const void* input,
                         const apr_size_t input_len);
    gpu_context_t* gpu_context_;
    int weight_;
    BOOL use_wide_string_;
    BOOL has_gpu_implementation_;
} hash_definition_t;

hash_definition_t* hsh_get_hash(const char* str);
void hsh_initialize_hashes(apr_pool_t* p);
void hsh_print_hashes(void);
const char* hsh_from_base64(const char* base64, apr_pool_t* p);

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_HASHES_H_
