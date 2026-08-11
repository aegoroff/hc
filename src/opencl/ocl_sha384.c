/*!
 * sha384 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "sha384.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_sha384;

static const char k_sha384_src[] =
#include "kernels/sha384.cl.h"
    ;

static void sha384_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_sha384);
}

static void prsha384_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_sha384, ctx, dict_len);
}

void sha384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&sha384_cleanup);
    (void)hc_ocl_algo_prepare(&g_sha384, k_sha384_src, "prsha384_kernel", dict, dict_len, hash, 48, ctx);
}

void sha384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha384_run);
}
