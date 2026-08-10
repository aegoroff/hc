/*!
 * crc32 brute-force on OpenCL — same index/prefix model as src/cuda/crc32.cu.
 */
#include "gpu_abi.h"
#include "crc32cu.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_crc32;

static const char k_crc32_src[] =
#include "kernels/crc32.cl.inc"
    ;

static void crc32_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_crc32);
}

static void prcrc32_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_crc32, ctx, dict_len);
}

void crc32_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&crc32_cleanup);
    (void)hc_ocl_algo_prepare(&g_crc32, k_crc32_src, "prcrc32_kernel", dict, dict_len, hash, 4, ctx);
}

void crc32_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prcrc32_run);
}
