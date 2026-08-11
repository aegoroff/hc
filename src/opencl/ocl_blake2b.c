/*!
 * blake2b brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "blake2b.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_blake2b;

static const char k_blake2b_src[] =
#include "kernels/blake2b.cl.h"
    ;

static void blake2b_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_blake2b);
}

static void prblake2b_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_blake2b, ctx, dict_len);
}

void blake2b_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&blake2b_cleanup);
    (void)hc_ocl_algo_prepare(&g_blake2b, k_blake2b_src, "prblake2b_kernel", dict, dict_len, hash, 64, ctx);
}

void blake2b_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prblake2b_run);
}
