/*!
 * blake2s brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "blake2s.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_blake2s;

static const char k_blake2s_src[] =
#include "kernels/blake2s.cl.inc"
    ;

static void blake2s_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_blake2s);
}

static void prblake2s_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_blake2s, ctx, dict_len);
}

void blake2s_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&blake2s_cleanup);
    (void)hc_ocl_algo_prepare(&g_blake2s, k_blake2s_src, "prblake2s_kernel", dict, dict_len, hash, 32, ctx);
}

void blake2s_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prblake2s_run);
}
