/*!
 * MD5 brute-force on OpenCL — same index/prefix model as src/cuda/md5.cu.
 */
#include "gpu_abi.h"
#include "md5.h"
#include "ocl_common.h"

#include <stddef.h>

static hc_ocl_algo_t g_md5;

static const char k_md5_src[] =
#include "kernels/md5.cl.h"
    ;

static void md5_cleanup(void) {
    hc_ocl_algo_release_bufs(&g_md5);
}

static void prmd5_run(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    hc_ocl_algo_run(&g_md5, ctx, dict_len);
}

void md5_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                        const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    (void)device_ix;
    hc_ocl_set_active_cleanup(&md5_cleanup);
    (void)hc_ocl_algo_prepare(&g_md5, k_md5_src, "prmd5_kernel", dict, dict_len, hash, 16, ctx);
}

void md5_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prmd5_run);
}
