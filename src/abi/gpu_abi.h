/*!
 * Clean C ABI for GPU brute-force (no APR types).
 * Layout mirrors gpu_tread_ctx_t / gpu_context_t historically defined in hashes.h.
 * This is now the single canonical source for those types (bf, CUDA, stub, Zig).
 */
#ifndef HC_GPU_ABI_H_
#define HC_GPU_ABI_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BOOL
#define BOOL bool
#endif
#ifndef TRUE
#define TRUE true
#endif
#ifndef FALSE
#define FALSE false
#endif

#ifndef GPU_ATTEMPT_SIZE
#define GPU_ATTEMPT_SIZE 16
#endif

typedef struct hc_device_props {
    int device_count;
    int max_blocks_number;
    int max_threads_per_block;
    int multiprocessor_count;
} hc_device_props_t;

typedef struct hc_gpu_versions {
    int major;
    int minor;
} hc_gpu_versions_t;

struct hc_gpu_context;

typedef struct hc_gpu_thread_ctx {
    unsigned char* variants_;
    unsigned char* dev_variants_;
    unsigned char* attempt_;
    unsigned char* result_;
    unsigned char* dev_result_;
    struct hc_gpu_context* gpu_context_;
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
    void* pool_; /* opaque; unused by CUDA kernels */
} hc_gpu_thread_ctx_t;

typedef struct hc_gpu_context {
    void (*pfn_run_)(void* context, const size_t dict_len, unsigned char* variants,
                     const size_t variants_size);
    void (*pfn_prepare_)(int device_ix, const unsigned char* dict, size_t dict_len,
                         const unsigned char* hash, hc_gpu_thread_ctx_t* ctx);
    int max_threads_decrease_factor_;
    int comparisons_per_iteration_;
} hc_gpu_context_t;

/* Compatibility aliases matching the historical C names used by .cu sources. */
typedef hc_device_props_t device_props_t;
typedef hc_gpu_versions_t gpu_versions_t;
typedef hc_gpu_thread_ctx_t gpu_tread_ctx_t;
typedef hc_gpu_context_t gpu_context_t;

void gpu_get_props(device_props_t* prop);
BOOL gpu_can_use_gpu(void);
int gpu_driver_version(void);
int gpu_runtime_version(void);
gpu_versions_t gpu_number_to_version(int version_number);
void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
             const size_t variants_size,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, unsigned char* r, unsigned char* v, const size_t dl));
void gpu_cleanup(gpu_tread_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

/*
 * CUDA helper macros — only meaningful under nvcc (__CUDACC__). Pulled in here
 * so the .cu sources need a single header (#include "gpu_abi.h") for both the
 * host/device ABI above and these kernel-building blocks. Never seen by the
 * CPU stub or by translate-c (compiled as plain C, no __CUDACC__).
 */
#if defined(__CUDACC__)

#include <stdio.h>

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) {                                               \
        fprintf(stderr, "Error:%s \"%s\" at %s:%d\n", cudaGetErrorName(err), cudaGetErrorString(err), \
        __FILE__, __LINE__); exit(1);                                                                 \
    }} while (0);
#endif

 /* a simple macro for kernel functions without hash allocations */
#ifndef KERNEL_WITHOUT_ALLOCATION
#define KERNEL_WITHOUT_ALLOCATION(func_name, compare_func)                       \
__global__ void func_name(unsigned char* result, unsigned char* variants, const uint32_t dict_length) { \
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;                                               \
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;                                          \
    size_t len = 0;                                                                                     \
    while (attempt[len]) {                                                                              \
        ++len;                                                                                          \
    }                                                                                                   \
    if (compare_func(attempt, len)) {                                                                   \
        memcpy(result, attempt, len);                                                                   \
        return;                                                                                         \
    }                                                                                                   \
    const size_t attempt_len = len + 1;                                                                 \
    for (int i = 0; i < dict_length; ++i)                                                               \
    {                                                                                                   \
        attempt[len] = k_dict[i];                                                                       \
        if (compare_func(attempt, attempt_len)) {                                                       \
            memcpy(result, attempt, attempt_len);                                                       \
            return;                                                                                     \
        }                                                                                               \
    }                                                                                                   \
}
#endif

/* a simple macro for kernel functions with hash allocations inside function */
#ifndef KERNEL_WITH_ALLOCATION
#define KERNEL_WITH_ALLOCATION(func_name, compare_func, T, HL)                       \
__global__ void func_name(unsigned char* result, unsigned char* variants, const uint32_t dict_length) { \
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;                                               \
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;                                          \
    T* hash = (T*)malloc(HL * sizeof(T));                                                               \
    size_t len = 0;                                                                                     \
    while (attempt[len]) {                                                                              \
        ++len;                                                                                          \
    }                                                                                                   \
    if (compare_func(attempt, len, hash)) {                                                             \
        memcpy(result, attempt, len);                                                                   \
        free(hash);                                                                                     \
        return;                                                                                         \
    }                                                                                                   \
    const size_t attempt_len = len + 1;                                                                 \
    for (int i = 0; i < dict_length; ++i)                                                               \
    {                                                                                                   \
        attempt[len] = k_dict[i];                                                                       \
        if (compare_func(attempt, attempt_len, hash)) {                                                 \
            memcpy(result, attempt, attempt_len);                                                       \
            free(hash);                                                                                 \
            return;                                                                                     \
        }                                                                                               \
    }                                                                                                   \
    free(hash);                                                                                         \
}
#endif

#endif /* __CUDACC__ */

#endif /* HC_GPU_ABI_H_ */
