/*!
 * sha256 brute-force on OpenCL — same index/prefix model as src/cuda/sha256.cu.
 */
#include "gpu_abi.h"
#include "sha256.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_sha256;

static const char k_sha256_src[] = {
#embed "kernels/sha256.cl"
, 0
};

static void sha256_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_sha256);
}

static void prsha256_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_sha256, ctx, dict_len);
}

void sha256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&sha256_cleanup);
    (void)hc_ocl_algo_prepare(&g_sha256, k_sha256_src, "prsha256_kernel", dict, dict_len, hash, 32, ctx);
}

void sha256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha256_run);
}
