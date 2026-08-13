#include "gpu_backends.h"

enum { HC_GPU_NONE = 0, HC_GPU_CUDA = 1, HC_GPU_OPENCL = 2 };
static int g_backend = HC_GPU_NONE;

static int pick_backend(void) {
    if (cuda_gpu_can_use_gpu()) return HC_GPU_CUDA;
    if (ocl_gpu_can_use_gpu()) return HC_GPU_OPENCL;
    return HC_GPU_NONE;
}

BOOL gpu_can_use_gpu(void) {
    return pick_backend() != HC_GPU_NONE ? TRUE : FALSE;
}

BOOL gpu_is_opencl(void) {
    return pick_backend() == HC_GPU_OPENCL;
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
    g_backend = pick_backend();
    if (g_backend == HC_GPU_CUDA) return cuda_gpu_init_pipeline(ctx);
    if (g_backend == HC_GPU_OPENCL) return ocl_gpu_init_pipeline(ctx);
    return FALSE;
}

void gpu_synchronize(gpu_tread_ctx_t* ctx) {
    if (g_backend == HC_GPU_CUDA) cuda_gpu_synchronize(ctx);
    else if (g_backend == HC_GPU_OPENCL) ocl_gpu_synchronize(ctx);
}

void gpu_cleanup(gpu_tread_ctx_t* ctx) {
    if (g_backend == HC_GPU_CUDA) cuda_gpu_cleanup(ctx);
    else if (g_backend == HC_GPU_OPENCL) ocl_gpu_cleanup(ctx);
    g_backend = HC_GPU_NONE;
}

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl)) {
    if (g_backend == HC_GPU_CUDA) cuda_gpu_run(ctx, dict_len, pfn_kernel);
    else if (g_backend == HC_GPU_OPENCL) ocl_gpu_run(ctx, dict_len, pfn_kernel);
}

/* Dual-backend wrappers for each hash: public ABI → cuda_* / ocl_*. */
#define HC_GPU_HASHES(X) \
    X(md5)               \
    X(md2)               \
    X(md4)               \
    X(sha1)              \
    X(sha224)            \
    X(sha256)            \
    X(sha384)            \
    X(sha512)            \
    X(sha3_224)          \
    X(sha3_256)          \
    X(sha3_384)          \
    X(sha3_512)          \
    X(keccak_224)        \
    X(keccak_256)        \
    X(keccak_384)        \
    X(keccak_512)        \
    X(rmd128)            \
    X(rmd160)            \
    X(rmd256)            \
    X(rmd320)            \
    X(blake2s)           \
    X(blake2b)           \
    X(blake3)            \
    X(tiger)             \
    X(tiger2)            \
    X(whirl)             \
    X(crc32)

#define HC_GPU_DISPATCH_HASH(name)                                                                        \
    void name##_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,                 \
                               const unsigned char* hash, gpu_tread_ctx_t* ctx) {                         \
        if (g_backend == HC_GPU_CUDA)                                                                     \
            cuda_##name##_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);                           \
        else if (g_backend == HC_GPU_OPENCL)                                                              \
            ocl_##name##_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);                            \
    }                                                                                                     \
    void name##_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {                                 \
        if (g_backend == HC_GPU_CUDA)                                                                     \
            cuda_##name##_run_on_gpu(ctx, dict_len);                                                      \
        else if (g_backend == HC_GPU_OPENCL)                                                              \
            ocl_##name##_run_on_gpu(ctx, dict_len);                                                       \
    }

HC_GPU_HASHES(HC_GPU_DISPATCH_HASH)

#undef HC_GPU_DISPATCH_HASH
#undef HC_GPU_HASHES
