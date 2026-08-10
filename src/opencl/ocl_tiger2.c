/*!
 * tiger2 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "tiger.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_tiger2;

static const char k_tiger2_src[] =
#include "kernels/tiger2.cl.inc"
    ;

static void tiger2_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_tiger2);
}

static void prtiger2_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_tiger2, ctx, dict_len);
}

void tiger2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&tiger2_cleanup);
    (void)hc_ocl_algo_prepare(&g_tiger2, k_tiger2_src, "prtiger2_kernel", dict, dict_len, hash, 24, ctx);
}

void tiger2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prtiger2_run);
}
