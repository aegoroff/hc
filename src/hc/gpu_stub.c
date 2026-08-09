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

void gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,
             const size_t variants_size,
             void (*pfn_kernel)(gpu_tread_ctx_t* c, unsigned char* r, unsigned char* v, const size_t dl)) {
    (void)ctx;
    (void)dict_len;
    (void)variants;
    (void)variants_size;
    (void)pfn_kernel;
}

#define STUB_ALGO(run_name, prep_name)                                                                 \
    void run_name(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants,                 \
                  const size_t variants_size) {                                                        \
        (void)ctx; (void)dict_len; (void)variants; (void)variants_size;                                \
    }                                                                                                  \
    void prep_name(int device_ix, const unsigned char* dict, size_t dict_len,                          \
                   const unsigned char* hash, gpu_tread_ctx_t* ctx) {                                  \
        (void)device_ix; (void)dict; (void)dict_len; (void)hash; (void)ctx;                            \
    }

STUB_ALGO(md5_run_on_gpu, md5_on_gpu_prepare)
STUB_ALGO(md2_run_on_gpu, md2_on_gpu_prepare)
STUB_ALGO(md4_run_on_gpu, md4_on_gpu_prepare)
STUB_ALGO(sha1_run_on_gpu, sha1_on_gpu_prepare)
STUB_ALGO(sha3_224_run_on_gpu, sha3_224_on_gpu_prepare)
STUB_ALGO(sha3_256_run_on_gpu, sha3_256_on_gpu_prepare)
STUB_ALGO(sha3_384_run_on_gpu, sha3_384_on_gpu_prepare)
STUB_ALGO(sha3_512_run_on_gpu, sha3_512_on_gpu_prepare)
STUB_ALGO(keccak_224_run_on_gpu, keccak_224_on_gpu_prepare)
STUB_ALGO(keccak_256_run_on_gpu, keccak_256_on_gpu_prepare)
STUB_ALGO(keccak_384_run_on_gpu, keccak_384_on_gpu_prepare)
STUB_ALGO(keccak_512_run_on_gpu, keccak_512_on_gpu_prepare)
STUB_ALGO(sha224_run_on_gpu, sha224_on_gpu_prepare)
STUB_ALGO(sha256_run_on_gpu, sha256_on_gpu_prepare)
STUB_ALGO(sha384_run_on_gpu, sha384_on_gpu_prepare)
STUB_ALGO(sha512_run_on_gpu, sha512_on_gpu_prepare)
STUB_ALGO(rmd128_run_on_gpu, rmd128_on_gpu_prepare)
STUB_ALGO(rmd160_run_on_gpu, rmd160_on_gpu_prepare)
STUB_ALGO(rmd256_run_on_gpu, rmd256_on_gpu_prepare)
STUB_ALGO(rmd320_run_on_gpu, rmd320_on_gpu_prepare)
STUB_ALGO(blake2s_run_on_gpu, blake2s_on_gpu_prepare)
STUB_ALGO(blake2b_run_on_gpu, blake2b_on_gpu_prepare)
STUB_ALGO(tiger_run_on_gpu, tiger_on_gpu_prepare)
STUB_ALGO(tiger2_run_on_gpu, tiger2_on_gpu_prepare)
STUB_ALGO(whirl_run_on_gpu, whirl_on_gpu_prepare)
STUB_ALGO(crc32_run_on_gpu, crc32_on_gpu_prepare)
