#include "gpu_backends.h"

void gpu_get_props(device_props_t* prop) { ocl_gpu_get_props(prop); }

BOOL gpu_get_device_props(int device_ix, device_props_t* prop) { return ocl_gpu_get_device_props(device_ix, prop); }

BOOL gpu_can_use_gpu(void) { return ocl_gpu_can_use_gpu(); }

BOOL gpu_is_opencl(void) { return ocl_gpu_can_use_gpu(); }

int gpu_driver_version(void) { return ocl_gpu_driver_version(); }

int gpu_runtime_version(void) { return ocl_gpu_runtime_version(); }

gpu_versions_t gpu_number_to_version(int version_number) { return ocl_gpu_number_to_version(version_number); }

BOOL gpu_init_pipeline(gpu_tread_ctx_t* ctx) { return ocl_gpu_init_pipeline(ctx); }

void gpu_synchronize(gpu_tread_ctx_t* ctx) { ocl_gpu_synchronize(ctx); }

void gpu_cleanup(gpu_tread_ctx_t* ctx) { ocl_gpu_cleanup(ctx); }

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl)) {
    ocl_gpu_run(ctx, dict_len, pfn_kernel);
}

#define HC_GPU_SHIM_HASH(name)                                                                            \
    void name##_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {                                 \
        ocl_##name##_run_on_gpu(ctx, dict_len);                                                           \
    }                                                                                                     \
    void name##_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,                 \
                               const unsigned char* hash, gpu_tread_ctx_t* ctx) {                         \
        ocl_##name##_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);                                \
    }

HC_GPU_HASHES(HC_GPU_SHIM_HASH)
#undef HC_GPU_SHIM_HASH
