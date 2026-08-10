#include "gpu_backends.h"

void gpu_get_props(device_props_t* prop) { ocl_gpu_get_props(prop); }

BOOL gpu_get_device_props(int device_ix, device_props_t* prop) { return ocl_gpu_get_device_props(device_ix, prop); }

BOOL gpu_can_use_gpu(void) { return ocl_gpu_can_use_gpu(); }

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

void md5_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_md5_run_on_gpu(ctx, dict_len);
}
void md5_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_md5_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void md2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_md2_run_on_gpu(ctx, dict_len);
}
void md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_md2_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_md4_run_on_gpu(ctx, dict_len);
}
void md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_md4_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha1_run_on_gpu(ctx, dict_len);
}
void sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha1_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha224_run_on_gpu(ctx, dict_len);
}
void sha224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha224_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha256_run_on_gpu(ctx, dict_len);
}
void sha256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha256_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha384_run_on_gpu(ctx, dict_len);
}
void sha384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha384_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha512_run_on_gpu(ctx, dict_len);
}
void sha512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha512_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha3_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha3_224_run_on_gpu(ctx, dict_len);
}
void sha3_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha3_224_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha3_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha3_256_run_on_gpu(ctx, dict_len);
}
void sha3_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha3_256_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha3_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha3_384_run_on_gpu(ctx, dict_len);
}
void sha3_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha3_384_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void sha3_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_sha3_512_run_on_gpu(ctx, dict_len);
}
void sha3_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_sha3_512_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void keccak_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_keccak_224_run_on_gpu(ctx, dict_len);
}
void keccak_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_keccak_224_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void keccak_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_keccak_256_run_on_gpu(ctx, dict_len);
}
void keccak_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_keccak_256_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void keccak_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_keccak_384_run_on_gpu(ctx, dict_len);
}
void keccak_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_keccak_384_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void keccak_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_keccak_512_run_on_gpu(ctx, dict_len);
}
void keccak_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_keccak_512_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void rmd128_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_rmd128_run_on_gpu(ctx, dict_len);
}
void rmd128_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_rmd128_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void rmd160_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_rmd160_run_on_gpu(ctx, dict_len);
}
void rmd160_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_rmd160_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void rmd256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_rmd256_run_on_gpu(ctx, dict_len);
}
void rmd256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_rmd256_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void rmd320_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_rmd320_run_on_gpu(ctx, dict_len);
}
void rmd320_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_rmd320_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void blake2s_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_blake2s_run_on_gpu(ctx, dict_len);
}
void blake2s_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_blake2s_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void blake2b_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_blake2b_run_on_gpu(ctx, dict_len);
}
void blake2b_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_blake2b_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void tiger_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_tiger_run_on_gpu(ctx, dict_len);
}
void tiger_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_tiger_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void tiger2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_tiger2_run_on_gpu(ctx, dict_len);
}
void tiger2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_tiger2_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void whirl_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_whirl_run_on_gpu(ctx, dict_len);
}
void whirl_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_whirl_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}

void crc32_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    ocl_crc32_run_on_gpu(ctx, dict_len);
}
void crc32_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                       const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    ocl_crc32_on_gpu_prepare(device_ix, dict, dict_len, hash, ctx);
}
