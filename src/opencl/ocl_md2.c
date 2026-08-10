/*!
 * md2 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "md2.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_md2;

static const char k_md2_src[] =
#include "kernels/md2.cl.inc"
    ;

static void md2_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_md2);
}

static void prmd2_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_md2, ctx, dict_len);
}

void md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&md2_cleanup);
    (void)hc_ocl_algo_prepare(&g_md2, k_md2_src, "prmd2_kernel", dict, dict_len, hash, 16, ctx);
}

void md2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prmd2_run);
}
