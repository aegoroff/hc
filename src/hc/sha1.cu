/*
* This is an open source non-commercial project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
*/
/*!
 * \brief   The file contains SHA-1 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2021
 */

#include <stdint.h>
#include "sha1.h"
#include "cuda_runtime.h"
#include "gpu.h"

#define DIGESTSIZE 20
#define BLOCK_LEN 64  // In bytes
#define STATE_LEN 5  // In words

__device__ static BOOL prsha1_compare(unsigned char* password, const int length);
__global__ static void prsha1_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length);
__device__ static void prsha1_compress(uint32_t state[], const uint8_t block[]);
__device__ static void prsha1_hash(const uint8_t* message, size_t len, uint32_t hash[]);
__host__ static void prsha1_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len);

__constant__ static uint8_t k_dict[CHAR_MAX];
__constant__ static uint8_t k_hash[DIGESTSIZE];
__device__ static BOOL g_found;

__host__ void sha1_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_variants_), ctx->variants_size_ * sizeof(unsigned char)));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));

    const BOOL f = FALSE;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_found, &f, sizeof(BOOL)));
}

__host__ void prsha1_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len) {
    prsha1_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>>(dev_result, dev_variants, static_cast<uint32_t>(dict_len));
}

__host__ void sha1_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prsha1_run_kernel);
}

__global__ void prsha1_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length) {
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned char* attempt = variants + ix * GPU_ATTEMPT_SIZE;
    size_t len = 0;

    if (g_found) {
        return;
    }

    // strlen
    while (attempt[len]) {
        ++len;
    }

    for (int i = 0; i < dict_length; ++i) {
        attempt[len] = k_dict[i];

        // Optimization: it was calculated before
        // Calculate only on first iteration
        if (len + 1 == 4) {
            if (g_found) {
                return;
            }

            if (prsha1_compare(attempt, len + 1)) {
                memcpy(result, attempt, len + 1);
                g_found = TRUE;
                return;
            }
        }

        for (int j = 0; j < dict_length; ++j) {
            attempt[len + 1] = k_dict[j];

            if (g_found) {
                return;
            }

            if (prsha1_compare(attempt, len + 2)) {
                memcpy(result, attempt, len + 2);
                g_found = TRUE;
                return;
            }
        }
    }
}

__device__ __forceinline__ BOOL prsha1_compare(unsigned char* password, const int length) {
    // load into register
    const uint32_t h0 = (unsigned)k_hash[3] | (unsigned)k_hash[2] << 8 | (unsigned)k_hash[1] << 16 | (unsigned)k_hash[0] << 24;
    const uint32_t h1 = (unsigned)k_hash[7] | (unsigned)k_hash[6] << 8 | (unsigned)k_hash[5] << 16 | (unsigned)k_hash[4] << 24;
    const uint32_t h2 = (unsigned)k_hash[11] | (unsigned)k_hash[10] << 8 | (unsigned)k_hash[9] << 16 | (unsigned)k_hash[8] << 24;
    const uint32_t h3 = (unsigned)k_hash[15] | (unsigned)k_hash[14] << 8 | (unsigned)k_hash[13] << 16 | (unsigned)k_hash[12] << 24;
    const uint32_t h4 = (unsigned)k_hash[19] | (unsigned)k_hash[18] << 8 | (unsigned)k_hash[17] << 16 | (unsigned)k_hash[16] << 24;

    uint32_t hash[STATE_LEN];
    prsha1_hash(password, length, hash);

    return hash[0] == h0 &&
        hash[1] == h1 &&
        hash[2] == h2 &&
        hash[3] == h3 &&
        hash[4] == h4;
}

__device__ __forceinline__ void prsha1_hash(const uint8_t* message, size_t len, uint32_t hash[]) {
    hash[0] = UINT32_C(0x67452301);
    hash[1] = UINT32_C(0xEFCDAB89);
    hash[2] = UINT32_C(0x98BADCFE);
    hash[3] = UINT32_C(0x10325476);
    hash[4] = UINT32_C(0xC3D2E1F0);

#define LENGTH_SIZE 8  // In bytes

    size_t off;
    for (off = 0; len - off >= BLOCK_LEN; off += BLOCK_LEN)
        prsha1_compress(hash, &message[off]);

    uint8_t block[BLOCK_LEN] = { 0 };
    size_t rem = len - off;
    memcpy(block, &message[off], rem);

    block[rem] = 0x80;
    rem++;
    if (BLOCK_LEN - rem < LENGTH_SIZE) {
        prsha1_compress(hash, block);
        memset(block, 0, sizeof(block));
    }

    block[BLOCK_LEN - 1] = (uint8_t)((len & 0x1FU) << 3);
    len >>= 5;

    for (int i = 1; i < LENGTH_SIZE; i++, len >>= 8)
        block[BLOCK_LEN - 1 - i] = (uint8_t)(len & 0xFFU);
    prsha1_compress(hash, block);
}

__device__ __forceinline__ void prsha1_compress(uint32_t state[], const uint8_t block[]) {
#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))  // Assumes that x is uint32_t and 0 < n < 32

#define LOADSCHEDULE(i)  \
		schedule[i] = (uint32_t)block[i * 4 + 0] << 24  \
		            | (uint32_t)block[i * 4 + 1] << 16  \
		            | (uint32_t)block[i * 4 + 2] <<  8  \
		            | (uint32_t)block[i * 4 + 3] <<  0;

#define SCHEDULE(i)  \
		temp = schedule[(i - 3) & 0xF] ^ schedule[(i - 8) & 0xF] ^ schedule[(i - 14) & 0xF] ^ schedule[(i - 16) & 0xF];  \
		schedule[i & 0xF] = ROTL32(temp, 1);

#define ROUND0a(a, b, c, d, e, i)  LOADSCHEDULE(i)  ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999)
#define ROUND0b(a, b, c, d, e, i)  SCHEDULE(i)      ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999)
#define ROUND1(a, b, c, d, e, i)   SCHEDULE(i)      ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0x6ED9EBA1)
#define ROUND2(a, b, c, d, e, i)   SCHEDULE(i)      ROUNDTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8F1BBCDC)
#define ROUND3(a, b, c, d, e, i)   SCHEDULE(i)      ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0xCA62C1D6)

#define ROUNDTAIL(a, b, e, f, i, k)  \
		e = 0U + e + ROTL32(a, 5) + f + UINT32_C(k) + schedule[i & 0xF];  \
		b = ROTL32(b, 30);

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];

    uint32_t schedule[16];
    uint32_t temp;
    ROUND0a(a, b, c, d, e, 0)
    ROUND0a(e, a, b, c, d, 1)
    ROUND0a(d, e, a, b, c, 2)
    ROUND0a(c, d, e, a, b, 3)
    ROUND0a(b, c, d, e, a, 4)
    ROUND0a(a, b, c, d, e, 5)
    ROUND0a(e, a, b, c, d, 6)
    ROUND0a(d, e, a, b, c, 7)
    ROUND0a(c, d, e, a, b, 8)
    ROUND0a(b, c, d, e, a, 9)
    ROUND0a(a, b, c, d, e, 10)
    ROUND0a(e, a, b, c, d, 11)
    ROUND0a(d, e, a, b, c, 12)
    ROUND0a(c, d, e, a, b, 13)
    ROUND0a(b, c, d, e, a, 14)
    ROUND0a(a, b, c, d, e, 15)
    ROUND0b(e, a, b, c, d, 16)
    ROUND0b(d, e, a, b, c, 17)
    ROUND0b(c, d, e, a, b, 18)
    ROUND0b(b, c, d, e, a, 19)
    ROUND1(a, b, c, d, e, 20)
    ROUND1(e, a, b, c, d, 21)
    ROUND1(d, e, a, b, c, 22)
    ROUND1(c, d, e, a, b, 23)
    ROUND1(b, c, d, e, a, 24)
    ROUND1(a, b, c, d, e, 25)
    ROUND1(e, a, b, c, d, 26)
    ROUND1(d, e, a, b, c, 27)
    ROUND1(c, d, e, a, b, 28)
    ROUND1(b, c, d, e, a, 29)
    ROUND1(a, b, c, d, e, 30)
    ROUND1(e, a, b, c, d, 31)
    ROUND1(d, e, a, b, c, 32)
    ROUND1(c, d, e, a, b, 33)
    ROUND1(b, c, d, e, a, 34)
    ROUND1(a, b, c, d, e, 35)
    ROUND1(e, a, b, c, d, 36)
    ROUND1(d, e, a, b, c, 37)
    ROUND1(c, d, e, a, b, 38)
    ROUND1(b, c, d, e, a, 39)
    ROUND2(a, b, c, d, e, 40)
    ROUND2(e, a, b, c, d, 41)
    ROUND2(d, e, a, b, c, 42)
    ROUND2(c, d, e, a, b, 43)
    ROUND2(b, c, d, e, a, 44)
    ROUND2(a, b, c, d, e, 45)
    ROUND2(e, a, b, c, d, 46)
    ROUND2(d, e, a, b, c, 47)
    ROUND2(c, d, e, a, b, 48)
    ROUND2(b, c, d, e, a, 49)
    ROUND2(a, b, c, d, e, 50)
    ROUND2(e, a, b, c, d, 51)
    ROUND2(d, e, a, b, c, 52)
    ROUND2(c, d, e, a, b, 53)
    ROUND2(b, c, d, e, a, 54)
    ROUND2(a, b, c, d, e, 55)
    ROUND2(e, a, b, c, d, 56)
    ROUND2(d, e, a, b, c, 57)
    ROUND2(c, d, e, a, b, 58)
    ROUND2(b, c, d, e, a, 59)
    ROUND3(a, b, c, d, e, 60)
    ROUND3(e, a, b, c, d, 61)
    ROUND3(d, e, a, b, c, 62)
    ROUND3(c, d, e, a, b, 63)
    ROUND3(b, c, d, e, a, 64)
    ROUND3(a, b, c, d, e, 65)
    ROUND3(e, a, b, c, d, 66)
    ROUND3(d, e, a, b, c, 67)
    ROUND3(c, d, e, a, b, 68)
    ROUND3(b, c, d, e, a, 69)
    ROUND3(a, b, c, d, e, 70)
    ROUND3(e, a, b, c, d, 71)
    ROUND3(d, e, a, b, c, 72)
    ROUND3(c, d, e, a, b, 73)
    ROUND3(b, c, d, e, a, 74)
    ROUND3(a, b, c, d, e, 75)
    ROUND3(e, a, b, c, d, 76)
    ROUND3(d, e, a, b, c, 77)
    ROUND3(c, d, e, a, b, 78)
    ROUND3(b, c, d, e, a, 79)

    state[0] = 0U + state[0] + a;
    state[1] = 0U + state[1] + b;
    state[2] = 0U + state[2] + c;
    state[3] = 0U + state[3] + d;
    state[4] = 0U + state[4] + e;
}