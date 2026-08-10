/*!
 * rmd160 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "rmd160.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_rmd160;

static const char k_rmd160_src[] =
#include "kernels/rmd160.cl.inc"
    ;

static void rmd160_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_rmd160);
}

static void prrmd160_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_rmd160, ctx, dict_len);
}

void rmd160_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&rmd160_cleanup);
    (void)hc_ocl_algo_prepare(&g_rmd160, k_rmd160_src, "prrmd160_kernel", dict, dict_len, hash, 20, ctx);
}

void rmd160_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prrmd160_run);
}
