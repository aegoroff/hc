/*!
 * rmd320 OpenCL brute-force backend.
 */
#include "gpu_abi.h"
#include "rmd320.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_rmd320;

static const char k_rmd320_src[] =
#include "kernels/rmd320.cl.inc"
    ;

static void rmd320_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_rmd320);
}

static void prrmd320_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_rmd320, ctx, dict_len);
}

void rmd320_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&rmd320_cleanup);
    (void)hc_ocl_algo_prepare(&g_rmd320, k_rmd320_src, "prrmd320_kernel", dict, dict_len, hash, 40, ctx);
}

void rmd320_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prrmd320_run);
}
