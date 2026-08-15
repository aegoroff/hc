/*!
 * \brief   BLAKE3 CUDA brute-force (short passwords ≤15 bytes, one block).
 * Copyright: (c) Alexander Egorov 2009-2026
 *
 * GPU_ATTEMPT_SIZE-1 ≪ BLAKE3_CHUNK_LEN, so one compress with
 * CHUNK_START|CHUNK_END|ROOT is enough (no tree / CV stack).
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"

#define BLOCK_LEN 64
#define HASH_LEN 32
#define CHUNK_START 1
#define CHUNK_END 2
#define ROOT 8
#define ROOT_FLAGS (CHUNK_START | CHUNK_END | ROOT)

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

__global__ static void prblake3_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                                       const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len);
__device__ static BOOL prblake3_compare(unsigned char* password, const int length, uint8_t* hash);
__device__ static void prblake3_hash(const uint8_t* message, size_t len, uint8_t* hash);
__device__ static void prblake3_compress(uint32_t* cv, const uint8_t* block, uint8_t block_len, uint8_t flags);

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[HASH_LEN];

__constant__ static const uint32_t k_iv[8] = {
    UINT32_C(0x6A09E667), UINT32_C(0xBB67AE85), UINT32_C(0x3C6EF372), UINT32_C(0xA54FF53A),
    UINT32_C(0x510E527F), UINT32_C(0x9B05688C), UINT32_C(0x1F83D9AB), UINT32_C(0x5BE0CD19),
};

__constant__ static const uint8_t k_schedule[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8 },
    { 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1 },
    { 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6 },
    { 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4 },
    { 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7 },
    { 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13 },
};

__host__ void HC_GPU_FN(blake3_on_gpu_prepare)(int device_ix, const unsigned char* dict, size_t dict_len,
                                    const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, HASH_LEN, 0, cudaMemcpyHostToDevice));
    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ static void prblake3_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    GPU_LAUNCH_INDEX_KERNEL(prblake3_kernel, ctx, dict_len);
}

__host__ void HC_GPU_FN(blake3_run_on_gpu)(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prblake3_run_kernel);
}

KERNEL_WITH_ALLOCATION(prblake3_kernel, prblake3_compare, uint8_t, HASH_LEN)

__device__ __forceinline__ BOOL prblake3_compare(unsigned char* password, const int length, uint8_t* hash) {
    prblake3_hash(password, length, hash);
    BOOL result = TRUE;
    for (int i = 0; i < HASH_LEN && result; ++i) {
        result &= hash[i] == k_hash[i];
    }
    return result;
}

__device__ __forceinline__ void prblake3_G(uint32_t* state, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    state[a] = state[a] + state[b] + x;
    state[d] = ROTR32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = ROTR32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + y;
    state[d] = ROTR32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = ROTR32(state[b] ^ state[c], 7);
}

__device__ __forceinline__ void prblake3_compress(uint32_t* cv, const uint8_t* block, uint8_t block_len,
                                                  uint8_t flags) {
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
        v[i] = cv[i];
        v[i + 8] = k_iv[i];
    }
    /* counter = 0 for the single short-password chunk */
    v[12] = 0;
    v[13] = 0;
    v[14] = static_cast<uint32_t>(block_len);
    v[15] = static_cast<uint32_t>(flags);

#pragma unroll
    for (int r = 0; r < 7; r++) {
        const uint8_t* s = k_schedule[r];
        prblake3_G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        prblake3_G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        prblake3_G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        prblake3_G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        prblake3_G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        prblake3_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        prblake3_G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        prblake3_G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        cv[i] = v[i] ^ v[i + 8];
    }
}

__device__ __forceinline__ void prblake3_hash(const uint8_t* message, size_t len, uint8_t* hash) {
    uint32_t cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        cv[i] = k_iv[i];
    }

    uint8_t block[BLOCK_LEN] = {};
    memcpy(block, message, len);
    prblake3_compress(cv, block, static_cast<uint8_t>(len), ROOT_FLAGS);

#pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i * 4 + 0] = static_cast<uint8_t>(cv[i]);
        hash[i * 4 + 1] = static_cast<uint8_t>(cv[i] >> 8);
        hash[i * 4 + 2] = static_cast<uint8_t>(cv[i] >> 16);
        hash[i * 4 + 3] = static_cast<uint8_t>(cv[i] >> 24);
    }
}
