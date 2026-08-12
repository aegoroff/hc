/*!
 * rmd128 OpenCL brute-force backend.
 */
#include "gpu_abi.h"
#include "rmd128.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_rmd128;

static const char k_rmd128_src[] = {
#embed "kernels/rmd128.cl"
, 0
};

static void rmd128_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_rmd128);
}

static void prrmd128_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_rmd128, ctx, dict_len);
}

void rmd128_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&rmd128_cleanup);
    (void)hc_ocl_algo_prepare(&g_rmd128, k_rmd128_src, "prrmd128_kernel", dict, dict_len, hash, 16, ctx);
}

void rmd128_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prrmd128_run);
}
