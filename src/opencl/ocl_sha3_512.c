/*!
 * sha3_512 brute-force on OpenCL — same index/prefix model as CUDA.
 */
#include "gpu_abi.h"
#include "sha3.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_sha3_512;

static const char k_sha3_512_src[] =
#include "kernels/sha3_512.cl.h"
    ;

static void sha3_512_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_sha3_512);
}

static void prsha3_512_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_sha3_512, ctx, dict_len);
}

void sha3_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&sha3_512_cleanup);
    (void)hc_ocl_algo_prepare(&g_sha3_512, k_sha3_512_src, "prsha3_512_kernel", dict, dict_len, hash, 64, ctx);
}

void sha3_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha3_512_run);
}
