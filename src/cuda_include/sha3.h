#ifndef LINQ2HASH_SHA3_GPU_H_
#define LINQ2HASH_SHA3_GPU_H_
#include <stddef.h>
#include "gpu_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
    void sha3_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void sha3_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
    void sha3_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void sha3_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
    void sha3_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void sha3_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
    void sha3_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void sha3_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);

    void keccak_224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void keccak_224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
    void keccak_256_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void keccak_256_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
    void keccak_384_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void keccak_384_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
    void keccak_512_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len);
    void keccak_512_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx);
#ifdef __cplusplus
}
#endif
#endif
