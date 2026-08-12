/*!
 * whirl brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "whirlpool.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_whirl;

static const char k_whirl_src[] = {
#embed "kernels/whirl.cl"
, 0
};

static void whirl_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_whirl);
}

static void prwhirl_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_whirl, ctx, dict_len);
}

void whirl_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&whirl_cleanup);
    (void)hc_ocl_algo_prepare(&g_whirl, k_whirl_src, "prwhirl_kernel", dict, dict_len, hash, 64, ctx);
}

void whirl_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prwhirl_run);
}
