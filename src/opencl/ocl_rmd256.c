/*!
 * rmd256 OpenCL brute-force backend.
 */
#include "gpu_abi.h"
#include "rmd256.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_rmd256;

static const char k_rmd256_src[] = {
#embed "kernels/rmd256.cl"
, 0
};

static void rmd256_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_rmd256);
}

static void prrmd256_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_rmd256, ctx, dict_len);
}

void rmd256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&rmd256_cleanup);
    (void)hc_ocl_algo_prepare(&g_rmd256, k_rmd256_src, "prrmd256_kernel", dict, dict_len, hash, 32, ctx);
}

void rmd256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prrmd256_run);
}
