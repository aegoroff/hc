/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains Ripemd 160 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-11-01
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu.h"
#include "rmd160.h"

#define BLOCK_LEN 64  // In bytes
#define HASH_LEN 20
#define NUM_ROUNDS 80


__global__ static void prrmd160_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length);
__device__ static BOOL prrmd160_compare(unsigned char* password, const int length, uint8_t* hash);
__device__ static void prrmd160_hash(const uint8_t* message, size_t len, uint8_t* hash);
__device__ static void prrmd160_compress(uint32_t* state, const uint8_t* blocks, size_t len);
__device__ uint32_t f(int i, uint32_t x, uint32_t y, uint32_t z);

 // Static initializers
__constant__ static const uint32_t KL[5] = {
    UINT32_C(0x00000000), UINT32_C(0x5A827999), UINT32_C(0x6ED9EBA1), UINT32_C(0x8F1BBCDC), UINT32_C(0xA953FD4E) };

__constant__ static const uint32_t KR[5] = {
    UINT32_C(0x50A28BE6), UINT32_C(0x5C4DD124), UINT32_C(0x6D703EF3), UINT32_C(0x7A6D76E9), UINT32_C(0x00000000) };

__constant__ static const int RL[NUM_ROUNDS] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    7,  4, 13,  1, 10,  6, 15,  3, 12,  0,  9,  5,  2, 14, 11,  8,
    3, 10, 14,  4,  9, 15,  8,  1,  2,  7,  0,  6, 13, 11,  5, 12,
    1,  9, 11, 10,  0,  8, 12,  4, 13,  3,  7, 15, 14,  5,  6,  2,
    4,  0,  5,  9,  7, 12,  2, 10, 14,  1,  3,  8, 11,  6, 15, 13 };

__constant__ static const int RR[NUM_ROUNDS] = {
    5, 14,  7,  0,  9,  2, 11,  4, 13,  6, 15,  8,  1, 10,  3, 12,
    6, 11,  3,  7,  0, 13,  5, 10, 14, 15,  8, 12,  4,  9,  1,  2,
    15,  5,  1,  3,  7, 14,  6,  9, 11,  8, 12,  2, 10,  0,  4, 13,
    8,  6,  4,  1,  3, 11, 15,  0,  5, 12,  2, 13,  9,  7, 10, 14,
    12, 15, 10,  4,  1,  5,  8,  7,  6,  2, 13, 14,  0,  3,  9, 11 };

__constant__ static const int SL[NUM_ROUNDS] = {
    11, 14, 15, 12,  5,  8,  7,  9, 11, 13, 14, 15,  6,  7,  9,  8,
    7,  6,  8, 13, 11,  9,  7, 15,  7, 12, 15,  9, 11,  7, 13, 12,
    11, 13,  6,  7, 14,  9, 13, 15, 14,  8, 13,  6,  5, 12,  7,  5,
    11, 12, 14, 15, 14, 15,  9,  8,  9, 14,  5,  6,  8,  6,  5, 12,
    9, 15,  5, 11,  6,  8, 13, 12,  5, 12, 13, 14, 11,  8,  5,  6 };

__constant__ static const int SR[NUM_ROUNDS] = {
    8,  9,  9, 11, 13, 15, 15,  5,  7,  7,  8, 11, 14, 14, 12,  6,
    9, 13, 15,  7, 12,  8,  9, 11,  7,  7, 12,  7,  6, 15, 13, 11,
    9,  7, 15, 11,  8,  6,  6, 14, 12, 13,  5, 14, 13, 13,  7,  5,
    15,  5,  8, 11, 14, 14,  6, 14,  6,  9, 12,  9, 12,  5, 15,  8,
    8,  5, 12,  9, 12,  5, 14,  6,  8, 13,  6,  5, 15, 13, 11, 11 };


__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[HASH_LEN];


__host__ void rmd160_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, HASH_LEN, 0, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_variants_), ctx->variants_size_ * sizeof(unsigned char)));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ void prwhirl_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len) {
    prrmd160_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>>(dev_result, dev_variants, static_cast<uint32_t>(dict_len));
}

__host__ void rmd160_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prwhirl_run_kernel);
}

KERNEL_WITH_ALLOCATION(prrmd160_kernel, prrmd160_compare, uint8_t, HASH_LEN)

__device__ __forceinline__ BOOL prrmd160_compare(unsigned char* password, const int length, uint8_t* hash) {
    prrmd160_hash(password, length, hash);

    BOOL result = TRUE;

    for (int i = 0; i < HASH_LEN && result; ++i) {
        result &= hash[i] == k_hash[i];
    }

    return result;
}

__device__ __forceinline__ void prrmd160_hash(const uint8_t* message, size_t len, uint8_t* hash) {
    uint32_t state[5] = { UINT32_C(0x67452301), UINT32_C(0xEFCDAB89), UINT32_C(0x98BADCFE), UINT32_C(0x10325476), UINT32_C(0xC3D2E1F0) };
    size_t off = len & ~static_cast<size_t>(BLOCK_LEN - 1);
    prrmd160_compress(state, message, off);

    // Final blocks, padding, and length
    uint8_t block[BLOCK_LEN] = {};
    memcpy(block, &message[off], len - off);
    off = len & (BLOCK_LEN - 1);
    block[off] = 0x80;
    ++off;
    if (off + 8 > BLOCK_LEN) {
        prrmd160_compress(state, block, BLOCK_LEN);
        memset(block, 0, BLOCK_LEN);
    }
    block[BLOCK_LEN - 8] = static_cast<uint8_t>((len & 0x1FU) << 3);
    len >>= 5;
    for (int i = 1; i < 8; i++, len >>= 8)
        block[BLOCK_LEN - 8 + i] = static_cast<uint8_t>(len);
    prrmd160_compress(state, block, BLOCK_LEN);

    // Uint32 array to bytes in little endian
    for (int i = 0; i < HASH_LEN; i++)
        hash[i] = static_cast<uint8_t>(state[i >> 2] >> ((i & 3) << 3));
}

__device__ __forceinline__ void prrmd160_compress(uint32_t* state, const uint8_t* blocks, size_t len) {
#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))  // Assumes that x is uint32_t and 0 < n < 32
    uint32_t schedule[16];
    for (size_t i = 0; i < len; ) {

        // Message schedule
#pragma unroll (4)
        for (int j = 0; j < 16; j++, i += 4) {
            schedule[j] = static_cast<uint32_t>(blocks[i + 0]) << 0
                | static_cast<uint32_t>(blocks[i + 1]) << 8
                | static_cast<uint32_t>(blocks[i + 2]) << 16
                | static_cast<uint32_t>(blocks[i + 3]) << 24;
        }

        // The 80 rounds
        uint32_t al = state[0], ar = state[0];
        uint32_t bl = state[1], br = state[1];
        uint32_t cl = state[2], cr = state[2];
        uint32_t dl = state[3], dr = state[3];
        uint32_t el = state[4], er = state[4];
#pragma unroll (4)
        for (int j = 0; j < NUM_ROUNDS; j++) {
            uint32_t temp = 0U + ROTL32(0U + al + f(j, bl, cl, dl) + schedule[RL[j]] + KL[j >> 4], SL[j]) + el;
            al = el;
            el = dl;
            dl = ROTL32(cl, 10);
            cl = bl;
            bl = temp;
            temp = 0U + ROTL32(0U + ar + f(NUM_ROUNDS - 1 - j, br, cr, dr) + schedule[RR[j]] + KR[j >> 4], SR[j]) + er;
            ar = er;
            er = dr;
            dr = ROTL32(cr, 10);
            cr = br;
            br = temp;
        }
        uint32_t temp = 0U + state[1] + cl + dr;
        state[1] = 0U + state[2] + dl + er;
        state[2] = 0U + state[3] + el + ar;
        state[3] = 0U + state[4] + al + br;
        state[4] = 0U + state[0] + bl + cr;
        state[0] = temp;
    }
}

__device__ __forceinline__ uint32_t f(int i, uint32_t x, uint32_t y, uint32_t z) {
    switch (i >> 4) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        case 4: return x ^ (y | ~z);
        default: return 0; // Dummy value to please the compiler
    }
}
