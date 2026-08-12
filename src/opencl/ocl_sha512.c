/*!
 * sha512 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "sha512.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_sha512;

static const char k_sha512_src[] = {
#embed "kernels/sha512.cl"
, 0
};

static void sha512_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_sha512);
}

static void prsha512_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_sha512, ctx, dict_len);
}

void sha512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&sha512_cleanup);
    (void)hc_ocl_algo_prepare(&g_sha512, k_sha512_src, "prsha512_kernel", dict, dict_len, hash, 64, ctx);
}

void sha512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha512_run);
}
