/*!
 * keccak_256 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "sha3.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_keccak_256;

static const char k_keccak_256_src[] = {
#embed "kernels/keccak_256.cl"
, 0
};

static void keccak_256_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_keccak_256);
}

static void prkeccak_256_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_keccak_256, ctx, dict_len);
}

void keccak_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&keccak_256_cleanup);
    (void)hc_ocl_algo_prepare(&g_keccak_256, k_keccak_256_src, "prkeccak_256_kernel", dict, dict_len, hash, 32, ctx);
}

void keccak_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prkeccak_256_run);
}
