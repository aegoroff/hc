#include "gpu_backends.h"

/* One worker thread runs one ctx per device (bf.zig), so the picked backend
 * lives on the ctx itself. A file-scope global would race: one worker's
 * gpu_cleanup resets it while siblings are still between gpu_run calls,
 * silently turning their launches into no-ops. */

static int pick_backend(void) {
    if (cuda_gpu_can_use_gpu()) return hc_gpu_backend_cuda;
    if (ocl_gpu_can_use_gpu()) return hc_gpu_backend_opencl;
    return hc_gpu_backend_none;
}

BOOL gpu_can_use_gpu(void) {
    return pick_backend() != hc_gpu_backend_none ? TRUE : FALSE;
}

void gpu_get_props(device_props_t* prop) {
    if (cuda_gpu_can_use_gpu()) cuda_gpu_get_props(prop);
    else ocl_gpu_get_props(prop);
}

BOOL gpu_get_device_props(int device_ix, device_props_t* prop) {
    if (cuda_gpu_can_use_gpu()) return cuda_gpu_get_device_props(device_ix, prop);
    return ocl_gpu_get_device_props(device_ix, prop);
}

int gpu_driver_version(void) {
    if (cuda_gpu_can_use_gpu()) return cuda_gpu_driver_version();
    return ocl_gpu_driver_version();
}

int gpu_runtime_version(void) {
    if (cuda_gpu_can_use_gpu()) return cuda_gpu_runtime_version();
    return ocl_gpu_runtime_version();
}

gpu_versions_t gpu_number_to_version(int version_number) {
    if (cuda_gpu_can_use_gpu()) return cuda_gpu_number_to_version(version_number);
    return ocl_gpu_number_to_version(version_number);
}

BOOL gpu_init_pipeline(gpu_tread_ctx_t* ctx) {
    ctx->backend_ = pick_backend();
    if (ctx->backend_ == hc_gpu_backend_cuda) return cuda_gpu_init_pipeline(ctx);
    if (ctx->backend_ == hc_gpu_backend_opencl) return ocl_gpu_init_pipeline(ctx);
    return FALSE;
}

void gpu_synchronize(gpu_tread_ctx_t* ctx) {
    if (ctx->backend_ == hc_gpu_backend_cuda) cuda_gpu_synchronize(ctx);
    else if (ctx->backend_ == hc_gpu_backend_opencl) ocl_gpu_synchronize(ctx);
}

void gpu_cleanup(gpu_tread_ctx_t* ctx) {
    if (ctx->backend_ == hc_gpu_backend_cuda) cuda_gpu_cleanup(ctx);
    else if (ctx->backend_ == hc_gpu_backend_opencl) ocl_gpu_cleanup(ctx);
    ctx->backend_ = hc_gpu_backend_none;
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl)) {
    if (ctx->backend_ == hc_gpu_backend_cuda) cuda_gpu_run(ctx, dict_len, pfn_kernel);
    else if (ctx->backend_ == hc_gpu_backend_opencl) ocl_gpu_run(ctx, dict_len, pfn_kernel);
}

/* Dual-backend wrappers for each hash: public ABI → cuda_* / ocl_*. */
#define HC_GPU_DISPATCH_HASH(name)                                                                        \
    void name##_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,                 \
                               const unsigned char* hash, gpu_tread_ctx_t* ctx) {                         \
        if (ctx->backend_ == hc_gpu_backend_cuda)                                                          \
            cuda_##name##_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);                           \
        else if (ctx->backend_ == hc_gpu_backend_opencl)                                                   \
            ocl_##name##_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);                            \
    }                                                                                                     \
    void name##_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {                                 \
        if (ctx->backend_ == hc_gpu_backend_cuda)                                                          \
            cuda_##name##_run_on_gpu(ctx, dict_len);                                                      \
        else if (ctx->backend_ == hc_gpu_backend_opencl)                                                   \
            ocl_##name##_run_on_gpu(ctx, dict_len);                                                       \
    }

HC_GPU_HASHES(HC_GPU_DISPATCH_HASH)
#undef HC_GPU_DISPATCH_HASH
