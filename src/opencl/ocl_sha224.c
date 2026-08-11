/*!
 * sha224 brute-force on OpenCL — same index/prefix model as src/cuda/sha224.cu.
 */
#include "gpu_abi.h"
#include "sha224.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_sha224;

static const char k_sha224_src[] =
#include "kernels/sha224.cl.h"
    ;

static void sha224_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_sha224);
}

static void prsha224_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_sha224, ctx, dict_len);
}

void sha224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&sha224_cleanup);
    (void)hc_ocl_algo_prepare(&g_sha224, k_sha224_src, "prsha224_kernel", dict, dict_len, hash, 28, ctx);
}

void sha224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha224_run);
}
