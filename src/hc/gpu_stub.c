/*
 * CPU-only stubs for the GPU C ABI. Linked when -Dcuda=false (default) so
 * hc/bf can always call gpu_can_use_gpu() and hash GPU entry points without
 * pulling in libcudart.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "gpu_abi.h"

void gpu_get_props(device_props_t* prop) {
    if (!prop) return;
    memset(prop, 0, sizeof(*prop));
}

BOOL gpu_get_device_props(int device_ix, device_props_t* prop) {
    (void)device_ix;
    if (!prop) return FALSE;
    memset(prop, 0, sizeof(*prop));
    return FALSE;
}

BOOL gpu_can_use_gpu(void) {
    return FALSE;
}

int gpu_driver_version(void) {
    return 0;
}

int gpu_runtime_version(void) {
    return 0;
}

gpu_versions_t gpu_number_to_version(int version_number) {
    gpu_versions_t version = { 0, 0 };
    version.major = version_number / 1000;
    version.minor = (version_number - version.major * 1000) / 10;
    return version;
}

BOOL gpu_init_pipeline(gpu_tread_ctx_t* ctx) {
    if (!ctx) return FALSE;
    ctx->stream_ = NULL;
    ctx->launch_in_flight_ = FALSE;
    return TRUE;
}

void gpu_synchronize(gpu_tread_ctx_t* _) {
}

void gpu_cleanup(gpu_tread_ctx_t* _) {
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl)) {
    (void)ctx;
    (void)dict_len;
    (void)pfn_kernel;
}

#define HC_GPU_STUB_HASH(name)                                                                         \
    void name##_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {                              \
        (void)ctx; (void)dict_len;                                                                     \
    }                                                                                                  \
    void name##_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,              \
                               const unsigned char* hash, gpu_tread_ctx_t* ctx) {                      \
        (void)device_ix; (void)dict; (void)dict_len; (void)hash; (void)ctx;                            \
    }

HC_GPU_HASHES(HC_GPU_STUB_HASH)
#undef HC_GPU_STUB_HASH
