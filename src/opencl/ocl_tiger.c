/*!
 * tiger brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "tiger.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_tiger;

static const char k_tiger_src[] =
#include "kernels/tiger.cl.inc"
    ;

static void tiger_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_tiger);
}

static void prtiger_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_tiger, ctx, dict_len);
}

void tiger_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&tiger_cleanup);
    (void)hc_ocl_algo_prepare(&g_tiger, k_tiger_src, "prtiger_kernel", dict, dict_len, hash, 24, ctx);
}

void tiger_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prtiger_run);
}
