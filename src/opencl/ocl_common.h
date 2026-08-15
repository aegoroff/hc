#ifndef HC_OCL_COMMON_H_
#define HC_OCL_COMMON_H_

#include "gpu_abi.h"
#include "ocl_api.h"
#include "ocl_runtime.h"

#include <stddef.h>

#define OCL_DICT_MAX GPU_DICT_MAX

typedef struct hc_ocl_algo {
    cl_program program;
    cl_kernel kernel;
    cl_mem dict_buf;
    cl_mem hash_buf;
    cl_mem result_buf;
    cl_mem found_buf;
    size_t hash_len;
    int ready;
    /** If non-zero, kernel arg 9 is ctx->use_wide_pass_ (NTLM / md4). */
    int pass_wide_arg;
} hc_ocl_algo_t;

/** Build/cache program+kernel; upload dict/hash; alloc result/found. */
int hc_ocl_algo_prepare(hc_ocl_algo_t* algo, const char* src, const char* kernel_name,
                        const unsigned char* dict, size_t dict_len, const unsigned char* hash,
                        size_t hash_len, gpu_tread_ctx_t* ctx);

/** Launch kernel for current batch; async read into ctx->result_. */
void hc_ocl_algo_run(hc_ocl_algo_t* algo, gpu_tread_ctx_t* ctx, size_t dict_len);

void hc_ocl_algo_release_bufs(hc_ocl_algo_t* algo);

/** Register algo for gpu_cleanup / entry_run (last prepare wins). */
void hc_ocl_set_active_algo(hc_ocl_algo_t* algo);
void hc_ocl_run_active_cleanup(void);

/** Shared ABI prepare/run used by ocl_algos.c entries. */
void hc_ocl_algo_entry_prepare(hc_ocl_algo_t* algo, const char* src, const char* kernel_name,
                               size_t hash_len, int pass_wide, int device_ix,
                               const unsigned char* dict, size_t dict_len, const unsigned char* hash,
                               gpu_tread_ctx_t* ctx);
void hc_ocl_algo_entry_run(gpu_tread_ctx_t* ctx, size_t dict_len);

#endif /* HC_OCL_COMMON_H_ */
