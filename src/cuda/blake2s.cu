/*!
 * \brief   BLAKE2s-256 CUDA brute-force (short passwords, one block).
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"
#include "blake2s.h"

#define BLOCK_LEN 64
#define HASH_LEN 32

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

__global__ static void prblake2s_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                                        const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len);
__device__ static BOOL prblake2s_compare(unsigned char* password, const int length, uint8_t* hash);
__device__ static void prblake2s_hash(const uint8_t* message, size_t len, uint8_t* hash);
__device__ static void prblake2s_compress(uint32_t* h, const uint8_t* block, uint64_t t, BOOL last);

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[HASH_LEN];

__constant__ static const uint32_t k_iv[8] = {
    UINT32_C(0x6A09E667), UINT32_C(0xBB67AE85), UINT32_C(0x3C6EF372), UINT32_C(0xA54FF53A),
    UINT32_C(0x510E527F), UINT32_C(0x9B05688C), UINT32_C(0x1F83D9AB), UINT32_C(0x5BE0CD19),
};

__constant__ static const uint8_t k_sigma[10][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

__host__ void blake2s_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len,
                                     const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, HASH_LEN, 0, cudaMemcpyHostToDevice));
    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ static void prblake2s_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    GPU_LAUNCH_INDEX_KERNEL(prblake2s_kernel, ctx, dict_len);
}

__host__ void blake2s_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prblake2s_run_kernel);
}

KERNEL_WITH_ALLOCATION(prblake2s_kernel, prblake2s_compare, uint8_t, HASH_LEN)

__device__ __forceinline__ BOOL prblake2s_compare(unsigned char* password, const int length, uint8_t* hash) {
    prblake2s_hash(password, length, hash);
    BOOL result = TRUE;
    for (int i = 0; i < HASH_LEN && result; ++i) {
        result &= hash[i] == k_hash[i];
    }
    return result;
}

__device__ __forceinline__ void prblake2s_hash(const uint8_t* message, size_t len, uint8_t* hash) {
    uint32_t h[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] = k_iv[i];
    }
    /* fanout=1, depth=1, keylen=0, digest_length=32 */
    h[0] ^= UINT32_C(0x01010000) ^ HASH_LEN;

    uint8_t block[BLOCK_LEN] = {};
    memcpy(block, message, len);
    prblake2s_compress(h, block, static_cast<uint64_t>(len), TRUE);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 0] = static_cast<uint8_t>(h[i]);
        hash[i * 4 + 1] = static_cast<uint8_t>(h[i] >> 8);
        hash[i * 4 + 2] = static_cast<uint8_t>(h[i] >> 16);
        hash[i * 4 + 3] = static_cast<uint8_t>(h[i] >> 24);
    }
}

__device__ __forceinline__ void prblake2s_G(uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = ROTR32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = ROTR32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = ROTR32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = ROTR32(v[b] ^ v[c], 7);
}

__device__ __forceinline__ void prblake2s_compress(uint32_t* h, const uint8_t* block, uint64_t t, BOOL last) {
    uint32_t m[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int o = i * 4;
        m[i] = static_cast<uint32_t>(block[o + 0])
            | (static_cast<uint32_t>(block[o + 1]) << 8)
            | (static_cast<uint32_t>(block[o + 2]) << 16)
            | (static_cast<uint32_t>(block[o + 3]) << 24);
    }

    uint32_t v[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = k_iv[i];
    }
    v[12] ^= static_cast<uint32_t>(t);
    v[13] ^= static_cast<uint32_t>(t >> 32);
    if (last) {
        v[14] = ~v[14];
    }

#pragma unroll
    for (int j = 0; j < 10; j++) {
        const uint8_t* s = k_sigma[j];
        prblake2s_G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        prblake2s_G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        prblake2s_G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        prblake2s_G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        prblake2s_G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        prblake2s_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        prblake2s_G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        prblake2s_G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}
