/*!
 * \brief   The file contains SHA-224 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-10-31
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2022
 */

#include <stdint.h>
#include "sha224.h"
#include "cuda_runtime.h"
#include "gpu.h"

#define DIGESTSIZE 28
#define BLOCK_LEN 64  // In bytes
#define STATE_LEN 8  // In words
#define HASH_LEN (STATE_LEN-1)  // In words

__device__ static BOOL prsha224_compare(unsigned char* password, const int length);
__global__ static void prsha224_kernel(unsigned char* result, unsigned char* variants, const uint32_t dict_length);
__device__ static void prsha256_compress(uint32_t state[], const uint8_t block[]);
__device__ static void prsha224_hash(const uint8_t* message, size_t len, uint32_t* hash);
__host__ static void prsha224_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len);

__constant__ static unsigned char k_dict[CHAR_MAX];
__constant__ static unsigned char k_hash[DIGESTSIZE];

__host__ void sha224_on_gpu_prepare(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE, 0, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_variants_), ctx->variants_size_ * sizeof(unsigned char)));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));
}

__host__ void prsha224_run_kernel(gpu_tread_ctx_t* ctx, unsigned char* dev_result, unsigned char* dev_variants, const size_t dict_len) {
    prsha224_kernel<<<ctx->max_gpu_blocks_number_, ctx->max_threads_per_block_>>>(dev_result, dev_variants, static_cast<uint32_t>(dict_len));
}

__host__ void sha224_run_on_gpu(gpu_tread_ctx_t* ctx, const size_t dict_len, unsigned char* variants, const size_t variants_size) {
    gpu_run(ctx, dict_len, variants, variants_size, &prsha224_run_kernel);
}

KERNEL_WITHOUT_ALLOCATION(prsha224_kernel, prsha224_compare)

__device__ BOOL prsha224_compare(unsigned char* password, const int length) {
    uint32_t hash[HASH_LEN];
    prsha224_hash(password, length, hash);

    BOOL result = TRUE;

    for(int i = 0; i < HASH_LEN && result; ++i) {
        result &= hash[i] == ((unsigned)k_hash[3 + i * 4] | (unsigned)k_hash[2 + i * 4] << 8 | (unsigned)k_hash[1 + i * 4] << 16 | (unsigned)k_hash[0 + i * 4] << 24);
    }

    return result;
}

__device__ void prsha224_hash(const uint8_t* message, size_t len, uint32_t* hash) {
    uint32_t state[STATE_LEN] = {
        UINT32_C(0xC1059ED8),
        UINT32_C(0x367CD507),
        UINT32_C(0x3070DD17),
        UINT32_C(0xF70E5939),
        UINT32_C(0xFFC00B31),
        UINT32_C(0x68581511),
        UINT32_C(0x64F98FA7),
        UINT32_C(0xBEFA4FA4),
    };

#define LENGTH_SIZE 8  // In bytes

    size_t off;
    for (off = 0; len - off >= BLOCK_LEN; off += BLOCK_LEN)
        prsha256_compress(state, &message[off]);

    uint8_t block[BLOCK_LEN] = { 0 };
    size_t rem = len - off;
    memcpy(block, &message[off], rem);

    block[rem] = 0x80;
    rem++;
    if (BLOCK_LEN - rem < LENGTH_SIZE) {
        prsha256_compress(state, block);
        memset(block, 0, sizeof(block));
    }

    block[BLOCK_LEN - 1] = (uint8_t)((len & 0x1FU) << 3);
    len >>= 5;

    for (int i = 1; i < LENGTH_SIZE; i++, len >>= 8)
        block[BLOCK_LEN - 1 - i] = (uint8_t)(len & 0xFFU);
    prsha256_compress(state, block);
    memcpy(hash, state, HASH_LEN * sizeof(uint32_t));
}

__device__ void prsha256_compress(uint32_t state[], const uint8_t block[]) {
#define ROTR32(x, n)  (((0U + (x)) << (32 - (n))) | ((x) >> (n)))  // Assumes that x is uint32_t and 0 < n < 32

#define LOADSCHEDULE(i)  \
		schedule[i] = (uint32_t)block[i * 4 + 0] << 24  \
		            | (uint32_t)block[i * 4 + 1] << 16  \
		            | (uint32_t)block[i * 4 + 2] <<  8  \
		            | (uint32_t)block[i * 4 + 3] <<  0;

#define SCHEDULE(i)  \
		schedule[i] = 0U + schedule[i - 16] + schedule[i - 7]  \
			+ (ROTR32(schedule[i - 15], 7) ^ ROTR32(schedule[i - 15], 18) ^ (schedule[i - 15] >> 3))  \
			+ (ROTR32(schedule[i - 2], 17) ^ ROTR32(schedule[i - 2], 19) ^ (schedule[i - 2] >> 10));

#define ROUND(a, b, c, d, e, f, g, h, i, k) \
		h = 0U + h + (ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25)) + (g ^ (e & (f ^ g))) + UINT32_C(k) + schedule[i];  \
		d = 0U + d + h;  \
		h = 0U + h + (ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22)) + ((a & (b | c)) | (b & c));

    uint32_t schedule[64];
    LOADSCHEDULE(0)
    LOADSCHEDULE(1)
    LOADSCHEDULE(2)
    LOADSCHEDULE(3)
    LOADSCHEDULE(4)
    LOADSCHEDULE(5)
    LOADSCHEDULE(6)
    LOADSCHEDULE(7)
    LOADSCHEDULE(8)
    LOADSCHEDULE(9)
    LOADSCHEDULE(10)
    LOADSCHEDULE(11)
    LOADSCHEDULE(12)
    LOADSCHEDULE(13)
    LOADSCHEDULE(14)
    LOADSCHEDULE(15)
    SCHEDULE(16)
    SCHEDULE(17)
    SCHEDULE(18)
    SCHEDULE(19)
    SCHEDULE(20)
    SCHEDULE(21)
    SCHEDULE(22)
    SCHEDULE(23)
    SCHEDULE(24)
    SCHEDULE(25)
    SCHEDULE(26)
    SCHEDULE(27)
    SCHEDULE(28)
    SCHEDULE(29)
    SCHEDULE(30)
    SCHEDULE(31)
    SCHEDULE(32)
    SCHEDULE(33)
    SCHEDULE(34)
    SCHEDULE(35)
    SCHEDULE(36)
    SCHEDULE(37)
    SCHEDULE(38)
    SCHEDULE(39)
    SCHEDULE(40)
    SCHEDULE(41)
    SCHEDULE(42)
    SCHEDULE(43)
    SCHEDULE(44)
    SCHEDULE(45)
    SCHEDULE(46)
    SCHEDULE(47)
    SCHEDULE(48)
    SCHEDULE(49)
    SCHEDULE(50)
    SCHEDULE(51)
    SCHEDULE(52)
    SCHEDULE(53)
    SCHEDULE(54)
    SCHEDULE(55)
    SCHEDULE(56)
    SCHEDULE(57)
    SCHEDULE(58)
    SCHEDULE(59)
    SCHEDULE(60)
    SCHEDULE(61)
    SCHEDULE(62)
    SCHEDULE(63)

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    ROUND(a, b, c, d, e, f, g, h, 0, 0x428A2F98)
    ROUND(h, a, b, c, d, e, f, g, 1, 0x71374491)
    ROUND(g, h, a, b, c, d, e, f, 2, 0xB5C0FBCF)
    ROUND(f, g, h, a, b, c, d, e, 3, 0xE9B5DBA5)
    ROUND(e, f, g, h, a, b, c, d, 4, 0x3956C25B)
    ROUND(d, e, f, g, h, a, b, c, 5, 0x59F111F1)
    ROUND(c, d, e, f, g, h, a, b, 6, 0x923F82A4)
    ROUND(b, c, d, e, f, g, h, a, 7, 0xAB1C5ED5)
    ROUND(a, b, c, d, e, f, g, h, 8, 0xD807AA98)
    ROUND(h, a, b, c, d, e, f, g, 9, 0x12835B01)
    ROUND(g, h, a, b, c, d, e, f, 10, 0x243185BE)
    ROUND(f, g, h, a, b, c, d, e, 11, 0x550C7DC3)
    ROUND(e, f, g, h, a, b, c, d, 12, 0x72BE5D74)
    ROUND(d, e, f, g, h, a, b, c, 13, 0x80DEB1FE)
    ROUND(c, d, e, f, g, h, a, b, 14, 0x9BDC06A7)
    ROUND(b, c, d, e, f, g, h, a, 15, 0xC19BF174)
    ROUND(a, b, c, d, e, f, g, h, 16, 0xE49B69C1)
    ROUND(h, a, b, c, d, e, f, g, 17, 0xEFBE4786)
    ROUND(g, h, a, b, c, d, e, f, 18, 0x0FC19DC6)
    ROUND(f, g, h, a, b, c, d, e, 19, 0x240CA1CC)
    ROUND(e, f, g, h, a, b, c, d, 20, 0x2DE92C6F)
    ROUND(d, e, f, g, h, a, b, c, 21, 0x4A7484AA)
    ROUND(c, d, e, f, g, h, a, b, 22, 0x5CB0A9DC)
    ROUND(b, c, d, e, f, g, h, a, 23, 0x76F988DA)
    ROUND(a, b, c, d, e, f, g, h, 24, 0x983E5152)
    ROUND(h, a, b, c, d, e, f, g, 25, 0xA831C66D)
    ROUND(g, h, a, b, c, d, e, f, 26, 0xB00327C8)
    ROUND(f, g, h, a, b, c, d, e, 27, 0xBF597FC7)
    ROUND(e, f, g, h, a, b, c, d, 28, 0xC6E00BF3)
    ROUND(d, e, f, g, h, a, b, c, 29, 0xD5A79147)
    ROUND(c, d, e, f, g, h, a, b, 30, 0x06CA6351)
    ROUND(b, c, d, e, f, g, h, a, 31, 0x14292967)
    ROUND(a, b, c, d, e, f, g, h, 32, 0x27B70A85)
    ROUND(h, a, b, c, d, e, f, g, 33, 0x2E1B2138)
    ROUND(g, h, a, b, c, d, e, f, 34, 0x4D2C6DFC)
    ROUND(f, g, h, a, b, c, d, e, 35, 0x53380D13)
    ROUND(e, f, g, h, a, b, c, d, 36, 0x650A7354)
    ROUND(d, e, f, g, h, a, b, c, 37, 0x766A0ABB)
    ROUND(c, d, e, f, g, h, a, b, 38, 0x81C2C92E)
    ROUND(b, c, d, e, f, g, h, a, 39, 0x92722C85)
    ROUND(a, b, c, d, e, f, g, h, 40, 0xA2BFE8A1)
    ROUND(h, a, b, c, d, e, f, g, 41, 0xA81A664B)
    ROUND(g, h, a, b, c, d, e, f, 42, 0xC24B8B70)
    ROUND(f, g, h, a, b, c, d, e, 43, 0xC76C51A3)
    ROUND(e, f, g, h, a, b, c, d, 44, 0xD192E819)
    ROUND(d, e, f, g, h, a, b, c, 45, 0xD6990624)
    ROUND(c, d, e, f, g, h, a, b, 46, 0xF40E3585)
    ROUND(b, c, d, e, f, g, h, a, 47, 0x106AA070)
    ROUND(a, b, c, d, e, f, g, h, 48, 0x19A4C116)
    ROUND(h, a, b, c, d, e, f, g, 49, 0x1E376C08)
    ROUND(g, h, a, b, c, d, e, f, 50, 0x2748774C)
    ROUND(f, g, h, a, b, c, d, e, 51, 0x34B0BCB5)
    ROUND(e, f, g, h, a, b, c, d, 52, 0x391C0CB3)
    ROUND(d, e, f, g, h, a, b, c, 53, 0x4ED8AA4A)
    ROUND(c, d, e, f, g, h, a, b, 54, 0x5B9CCA4F)
    ROUND(b, c, d, e, f, g, h, a, 55, 0x682E6FF3)
    ROUND(a, b, c, d, e, f, g, h, 56, 0x748F82EE)
    ROUND(h, a, b, c, d, e, f, g, 57, 0x78A5636F)
    ROUND(g, h, a, b, c, d, e, f, 58, 0x84C87814)
    ROUND(f, g, h, a, b, c, d, e, 59, 0x8CC70208)
    ROUND(e, f, g, h, a, b, c, d, 60, 0x90BEFFFA)
    ROUND(d, e, f, g, h, a, b, c, 61, 0xA4506CEB)
    ROUND(c, d, e, f, g, h, a, b, 62, 0xBEF9A3F7)
    ROUND(b, c, d, e, f, g, h, a, 63, 0xC67178F2)
    state[0] = 0U + state[0] + a;
    state[1] = 0U + state[1] + b;
    state[2] = 0U + state[2] + c;
    state[3] = 0U + state[3] + d;
    state[4] = 0U + state[4] + e;
    state[5] = 0U + state[5] + f;
    state[6] = 0U + state[6] + g;
    state[7] = 0U + state[7] + h;
}