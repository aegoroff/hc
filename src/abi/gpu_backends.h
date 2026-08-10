#ifndef HC_GPU_BACKENDS_H_
#define HC_GPU_BACKENDS_H_
#include "gpu_abi.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Prefixed CUDA symbols (dual build). */
void cuda_gpu_get_props(device_props_t* prop);
BOOL cuda_gpu_get_device_props(int device_ix, device_props_t* prop);
BOOL cuda_gpu_can_use_gpu(void);
int cuda_gpu_driver_version(void);
int cuda_gpu_runtime_version(void);
gpu_versions_t cuda_gpu_number_to_version(int version_number);
BOOL cuda_gpu_init_pipeline(gpu_tread_ctx_t* ctx);
void cuda_gpu_synchronize(gpu_tread_ctx_t* ctx);
void cuda_gpu_cleanup(gpu_tread_ctx_t* ctx);
void cuda_gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl));
void cuda_md5_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_md5_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_md2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha1_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha3_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha3_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha3_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha3_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha3_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha3_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_sha3_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_sha3_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_keccak_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_keccak_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_keccak_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_keccak_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_keccak_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_keccak_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_keccak_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_keccak_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_rmd128_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_rmd128_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_rmd160_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_rmd160_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_rmd256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_rmd256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_rmd320_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_rmd320_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_blake2s_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_blake2s_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_blake2b_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_blake2b_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_tiger_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_tiger_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_tiger2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_tiger2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_whirl_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_whirl_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void cuda_crc32_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void cuda_crc32_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);

/* Prefixed OpenCL symbols. */
void ocl_gpu_get_props(device_props_t* prop);
BOOL ocl_gpu_get_device_props(int device_ix, device_props_t* prop);
BOOL ocl_gpu_can_use_gpu(void);
int ocl_gpu_driver_version(void);
int ocl_gpu_runtime_version(void);
gpu_versions_t ocl_gpu_number_to_version(int version_number);
BOOL ocl_gpu_init_pipeline(gpu_tread_ctx_t* ctx);
void ocl_gpu_synchronize(gpu_tread_ctx_t* ctx);
void ocl_gpu_cleanup(gpu_tread_ctx_t* ctx);
void ocl_gpu_run(gpu_tread_ctx_t* ctx, const size_t dict_len, void (*pfn_kernel)(gpu_tread_ctx_t* c, const size_t dl));
void ocl_md5_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_md5_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_md2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_md2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_md4_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_md4_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha1_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha3_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha3_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha3_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha3_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha3_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha3_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_sha3_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_sha3_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_keccak_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_keccak_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_keccak_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_keccak_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_keccak_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_keccak_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_keccak_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_keccak_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_rmd128_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_rmd128_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_rmd160_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_rmd160_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_rmd256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_rmd256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_rmd320_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_rmd320_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_blake2s_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_blake2s_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_blake2b_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_blake2b_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_tiger_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_tiger_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_tiger2_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_tiger2_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_whirl_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_whirl_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
void ocl_crc32_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
void ocl_crc32_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);

#ifdef __cplusplus
}
#endif
#endif /* HC_GPU_BACKENDS_H_ */
