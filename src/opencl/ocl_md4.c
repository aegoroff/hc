/*!
 * md4 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "md4.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_md4;

static const char k_md4_src[] =
#include "kernels/md4.cl.inc"
    ;

static void md4_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_md4);
}

static void prmd4_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_md4, ctx, dict_len);
}

void md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&md4_cleanup);
    (void)hc_ocl_algo_prepare(&g_md4, k_md4_src, "prmd4_kernel", dict, dict_len, hash, 16, ctx);
}

void md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prmd4_run);
}
