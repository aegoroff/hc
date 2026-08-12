/*!
 * keccak_384 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "sha3.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_keccak_384;

static const char k_keccak_384_src[] = {
#embed "kernels/keccak_384.cl"
, 0
};

static void keccak_384_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_keccak_384);
}

static void prkeccak_384_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_keccak_384, ctx, dict_len);
}

void keccak_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&keccak_384_cleanup);
    (void)hc_ocl_algo_prepare(&g_keccak_384, k_keccak_384_src, "prkeccak_384_kernel", dict, dict_len, hash, 48, ctx);
}

void keccak_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prkeccak_384_run);
}
