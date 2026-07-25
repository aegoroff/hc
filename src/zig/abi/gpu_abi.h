/*!
 * Clean C ABI for GPU brute-force (no APR types).
 * Layout mirrors gpu_tread_ctx_t / gpu_context_t from hashes.h.
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

#endif /* HC_GPU_ABI_H_ */
