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
    unsigned char* attempt_;
    unsigned char* result_;
    unsigned char* dev_result_;
    struct hc_gpu_context* gpu_context_;
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
    /* Per-context fill index so multi-GPU workers do not share a process-global. */
    uint32_t variant_ix_;
    void* stream_; /* cudaStream_t when CUDA; NULL in stub */
    BOOL launch_in_flight_;
    /* GPU-side prefix index: thread ix → prefix at index_start_+ix of
     * length pass_length_; kernel expands comparisons_per_iteration_ chars. */
    uint64_t index_start_;
    uint32_t batch_count_;
} hc_gpu_thread_ctx_t;

typedef struct hc_gpu_context {
    void (*pfn_run_)(void* context, const size_t dict_len);
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
/** Launch geometry for a single device (not summed across GPUs). */
BOOL gpu_get_device_props(int device_ix, device_props_t* prop);
BOOL gpu_can_use_gpu(void);
int gpu_driver_version(void);
int gpu_runtime_version(void);
gpu_versions_t gpu_number_to_version(int version_number);
void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl));
/** Create CUDA stream (index-gen path). */
BOOL gpu_init_pipeline(gpu_tread_ctx_t* ctx);
/** Stream sync + publish found_in_the_thread_ from result_. */
void gpu_synchronize(gpu_tread_ctx_t* ctx);
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

#ifndef GPU_STREAM
#define GPU_STREAM(ctx) ((cudaStream_t)((ctx)->stream_))
#endif

 /* Prefix index + 1-char expand (cpi=1). min_len is passmin — skip shorter hits. */
#ifndef KERNEL_WITHOUT_ALLOCATION
#define KERNEL_WITHOUT_ALLOCATION(func_name, compare_func)                                              \
__global__ void func_name(unsigned char* result, const uint64_t start, const uint32_t count,            \
                          const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len) { \
    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;                                          \
    if (ix >= count) return;                                                                            \
    uint64_t idx = start + ix;                                                                          \
    unsigned char attempt[GPU_ATTEMPT_SIZE];                                                             \
    for (int pos = (int)pass_len - 1; pos >= 0; --pos) {                                                \
        attempt[pos] = k_dict[idx % dict_length];                                                       \
        idx /= dict_length;                                                                             \
    }                                                                                                   \
    if (pass_len >= min_len && compare_func(attempt, (int)pass_len)) {                                  \
        memcpy(result, attempt, pass_len);                                                              \
        return;                                                                                         \
    }                                                                                                   \
    const uint32_t attempt_len = pass_len + 1u;                                                         \
    if (attempt_len < min_len) return;                                                                  \
    for (uint32_t i = 0; i < dict_length; ++i) {                                                        \
        attempt[pass_len] = k_dict[i];                                                                  \
        if (compare_func(attempt, (int)attempt_len)) {                                                  \
            memcpy(result, attempt, attempt_len);                                                       \
            return;                                                                                     \
        }                                                                                               \
    }                                                                                                   \
}
#endif

#ifndef KERNEL_WITH_ALLOCATION
#define KERNEL_WITH_ALLOCATION(func_name, compare_func, T, HL)                                          \
__global__ void func_name(unsigned char* result, const uint64_t start, const uint32_t count,            \
                          const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len) { \
    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;                                          \
    if (ix >= count) return;                                                                            \
    uint64_t idx = start + ix;                                                                          \
    unsigned char attempt[GPU_ATTEMPT_SIZE];                                                             \
    T hash[HL];                                                                                         \
    for (int pos = (int)pass_len - 1; pos >= 0; --pos) {                                                \
        attempt[pos] = k_dict[idx % dict_length];                                                       \
        idx /= dict_length;                                                                             \
    }                                                                                                   \
    if (pass_len >= min_len && compare_func(attempt, (int)pass_len, hash)) {                            \
        memcpy(result, attempt, pass_len);                                                              \
        return;                                                                                         \
    }                                                                                                   \
    const uint32_t attempt_len = pass_len + 1u;                                                         \
    if (attempt_len < min_len) return;                                                                  \
    for (uint32_t i = 0; i < dict_length; ++i) {                                                        \
        attempt[pass_len] = k_dict[i];                                                                  \
        if (compare_func(attempt, (int)attempt_len, hash)) {                                            \
            memcpy(result, attempt, attempt_len);                                                       \
            return;                                                                                     \
        }                                                                                               \
    }                                                                                                   \
}
#endif

#ifndef GPU_LAUNCH_INDEX_KERNEL
#define GPU_LAUNCH_INDEX_KERNEL(kernel_fn, ctx, dict_len)                                               \
    do {                                                                                                \
        const uint32_t threads = (uint32_t)(ctx)->max_threads_per_block_;                               \
        const uint32_t count = (ctx)->batch_count_;                                                     \
        const uint32_t blocks = (count + threads - 1u) / threads;                                        \
        const uint32_t min_len = (ctx)->passmin_ ? (ctx)->passmin_ : 1u;                                \
        kernel_fn<<<blocks, threads, 0, GPU_STREAM(ctx)>>>((ctx)->dev_result_, (ctx)->index_start_,     \
            count, (ctx)->pass_length_, static_cast<uint32_t>(dict_len), min_len);                      \
    } while (0)
#endif

#endif /* __CUDACC__ */

#endif /* HC_GPU_ABI_H_ */
