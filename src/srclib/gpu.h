/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains GPU related code interface
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2019
 */

#ifndef LINQ2HASH_GPU_H_
#define LINQ2HASH_GPU_H_

#define CUDA_SAFE_CALL(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) { \
        fprintf(stderr, "Error:%s \"%s\" at %s:%d\n", cudaGetErrorName(err), cudaGetErrorString(err), \
        __FILE__, __LINE__); return; \
    }} while (0);

#include "bf.h"

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct device_props_t {
        int device_count;
        int max_blocks_number;
        int max_threads_per_block;
    } device_props_t;

    void gpu_get_props(device_props_t* prop);

    BOOL gpu_can_use_gpu();

    void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size,
                 void (*pfn_kernel)(gpu_tread_ctx_t* c, unsigned char* r, unsigned char* v, const size_t dl));

    void gpu_cleanup(gpu_tread_ctx_t* ctx);

/* a simple macro for kernel functions without hash allocations */
#define KERNEL_WITHOUT_ALLOCATION(func_name, compare_name)                       \
__global__ void func_name(unsigned char* result, unsigned char* variants, const uint32_t dict_length) { \
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;                                               \
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;                                          \
    size_t len = 0;                                                                                     \
    while (attempt[len]) {                                                                              \
        ++len;                                                                                          \
    }                                                                                                   \
    if (compare_name(attempt, len)) {                                                                   \
        memcpy(result, attempt, len);                                                                   \
        return;                                                                                         \
    }                                                                                                   \
    const size_t attempt_len = len + 1;                                                                 \
    for (int i = 0; i < dict_length; ++i)                                                               \
    {                                                                                                   \
        attempt[len] = k_dict[i];                                                                       \
        if (compare_name(attempt, attempt_len)) {                                                       \
            memcpy(result, attempt, attempt_len);                                                       \
            return;                                                                                     \
        }                                                                                               \
    }                                                                                                   \
}

/* a simple macro for kernel functions with hash allocations inside function */
#define KERNEL_WITH_ALLOCATION(func_name, compare_name, T, HL)                       \
__global__ void func_name(unsigned char* result, unsigned char* variants, const uint32_t dict_length) { \
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;                                               \
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;                                          \
    T* hash = (T*)malloc(HL * sizeof(T));                                                               \
    size_t len = 0;                                                                                     \
    while (attempt[len]) {                                                                              \
        ++len;                                                                                          \
    }                                                                                                   \
    if (compare_name(attempt, len, hash)) {                                                             \
        memcpy(result, attempt, len);                                                                   \
        free(hash);                                                                                     \
        return;                                                                                         \
    }                                                                                                   \
    const size_t attempt_len = len + 1;                                                                 \
    for (int i = 0; i < dict_length; ++i)                                                               \
    {                                                                                                   \
        attempt[len] = k_dict[i];                                                                       \
        if (compare_name(attempt, attempt_len, hash)) {                                                 \
            memcpy(result, attempt, attempt_len);                                                       \
            free(hash);                                                                                 \
            return;                                                                                     \
        }                                                                                               \
    }                                                                                                   \
    free(hash);                                                                                         \
}

#ifdef __cplusplus
}
#endif

#endif // LINQ2HASH_GPU_H_
