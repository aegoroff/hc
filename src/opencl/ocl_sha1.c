/*!
 * sha1 brute-force on OpenCL — same index/prefix model as src/cuda/sha1.cu.
 */
#include "gpu_abi.h"
#include "sha1.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_sha1;

static const char k_sha1_src[] = {
#embed "kernels/sha1.cl"
, 0
};

static void sha1_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_sha1);
}

static void prsha1_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_sha1, ctx, dict_len);
}

void sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&sha1_cleanup);
    (void)hc_ocl_algo_prepare(&g_sha1, k_sha1_src, "prsha1_kernel", dict, dict_len, hash, 20, ctx);
}

void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha1_run);
}
