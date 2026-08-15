/*!
 * \brief   The file contains SHA-1 CUDA code implementation
 * \author  \verbatim
            Created by: Alexander Egorov
            \endverbatim
 * \date    \verbatim
            Creation date: 2017-09-27
            \endverbatim
 * Copyright: (c) Alexander Egorov 2009-2026
 */

#include <stdint.h>
#include "cuda_runtime.h"
#include "gpu_abi.h"

#define DIGESTSIZE 20

__device__ static BOOL prsha1_compare(unsigned char* password, const int length);
__global__ static void prsha1_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                                     const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len);
__host__ static void prsha1_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len);

__constant__ static uint8_t k_dict[CHAR_MAX];
__constant__ static uint8_t k_hash[DIGESTSIZE];
__device__ static int g_found;

__host__ void HC_GPU_FN(sha1_on_gpu_prepare)(int device_ix, const unsigned char* dict, size_t dict_len, const unsigned char* hash, gpu_tread_ctx_t* ctx) {
    CUDA_SAFE_CALL(cudaSetDevice(device_ix));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_dict, dict, dict_len * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_hash, hash, DIGESTSIZE, 0, cudaMemcpyHostToDevice));

    size_t result_size_in_bytes = GPU_ATTEMPT_SIZE * sizeof(unsigned char); // include trailing zero
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ctx->dev_result_), result_size_in_bytes));

    const int f = 0;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_found, &f, sizeof(int)));
}

__host__ void prsha1_run_kernel(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    GPU_LAUNCH_INDEX_KERNEL(prsha1_kernel, ctx, dict_len);
}

__host__ void HC_GPU_FN(sha1_run_on_gpu)(gpu_tread_ctx_t* ctx, const size_t dict_len) {
    gpu_run(ctx, dict_len, &prsha1_run_kernel);
}

__global__ void prsha1_kernel(unsigned char* result, const uint64_t start, const uint32_t count,
                              const uint32_t pass_len, const uint32_t dict_length, const uint32_t min_len) {
    const uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= count || g_found) {
        return;
    }

    uint64_t idx = start + ix;
    unsigned char attempt[GPU_ATTEMPT_SIZE];
    for (int pos = (int)pass_len - 1; pos >= 0; --pos) {
        attempt[pos] = k_dict[idx % dict_length];
        idx /= dict_length;
    }

    for (uint32_t i = 0; i < dict_length; ++i) {
        attempt[pass_len] = k_dict[i];

        if (pass_len + 1u == 4u && pass_len + 1u >= min_len) {
            if (g_found) {
                return;
            }
            if (prsha1_compare(attempt, (int)(pass_len + 1u))) {
                memcpy(result, attempt, pass_len + 1u);
                atomicExch(&g_found, 1);
                return;
            }
        }

        if (pass_len + 2u < min_len) {
            continue;
        }
        for (uint32_t j = 0; j < dict_length; ++j) {
            attempt[pass_len + 1u] = k_dict[j];
            if (g_found) {
                return;
            }
            if (prsha1_compare(attempt, (int)(pass_len + 2u))) {
                memcpy(result, attempt, pass_len + 2u);
                atomicExch(&g_found, 1);
                return;
            }
        }
    }
}

/* Short-password SHA-1: single block, big-endian words (len <= GPU_ATTEMPT_SIZE). */
__device__ __forceinline__ BOOL prsha1_compare(unsigned char* password, const int length) {
    const uint32_t h0 = (unsigned)k_hash[3] | (unsigned)k_hash[2] << 8 | (unsigned)k_hash[1] << 16 | (unsigned)k_hash[0] << 24;
    const uint32_t h1 = (unsigned)k_hash[7] | (unsigned)k_hash[6] << 8 | (unsigned)k_hash[5] << 16 | (unsigned)k_hash[4] << 24;
    const uint32_t h2 = (unsigned)k_hash[11] | (unsigned)k_hash[10] << 8 | (unsigned)k_hash[9] << 16 | (unsigned)k_hash[8] << 24;
    const uint32_t h3 = (unsigned)k_hash[15] | (unsigned)k_hash[14] << 8 | (unsigned)k_hash[13] << 16 | (unsigned)k_hash[12] << 24;
    const uint32_t h4 = (unsigned)k_hash[19] | (unsigned)k_hash[18] << 8 | (unsigned)k_hash[17] << 16 | (unsigned)k_hash[16] << 24;

    const uint32_t a0 = UINT32_C(0x67452301);
    const uint32_t b0 = UINT32_C(0xEFCDAB89);
    const uint32_t c0 = UINT32_C(0x98BADCFE);
    const uint32_t d0 = UINT32_C(0x10325476);
    const uint32_t e0 = UINT32_C(0xC3D2E1F0);

    uint32_t schedule[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    int i;
#pragma unroll (4)
    for (i = 0; i < length; ++i) {
        schedule[i / 4] |= (uint32_t)password[i] << (24 - (i % 4) * 8);
    }
    schedule[i / 4] |= 0x80u << (24 - (i % 4) * 8);
    schedule[15] = (uint32_t)length * 8u;

#define ROTL32(x, n)  (((0U + (x)) << (n)) | ((x) >> (32 - (n))))
#define SCHEDULE(i)  \
                temp = schedule[(i - 3) & 0xF] ^ schedule[(i - 8) & 0xF] ^ schedule[(i - 14) & 0xF] ^ schedule[(i - 16) & 0xF];  \
                schedule[i & 0xF] = ROTL32(temp, 1);
#define ROUND0(a, b, c, d, e, i)  ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999)
#define ROUND0b(a, b, c, d, e, i) SCHEDULE(i) ROUNDTAIL(a, b, e, ((b & c) | (~b & d))         , i, 0x5A827999)
#define ROUND1(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0x6ED9EBA1)
#define ROUND2(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, ((b & c) ^ (b & d) ^ (c & d)), i, 0x8F1BBCDC)
#define ROUND3(a, b, c, d, e, i)  SCHEDULE(i) ROUNDTAIL(a, b, e, (b ^ c ^ d)                  , i, 0xCA62C1D6)
#define ROUNDTAIL(a, b, e, f, i, k)  \
                e = 0U + e + ROTL32(a, 5) + f + UINT32_C(k) + schedule[i & 0xF];  \
                b = ROTL32(b, 30);

    uint32_t a = a0;
    uint32_t b = b0;
    uint32_t c = c0;
    uint32_t d = d0;
    uint32_t e = e0;
    uint32_t temp;

    ROUND0(a, b, c, d, e, 0)
    ROUND0(e, a, b, c, d, 1)
    ROUND0(d, e, a, b, c, 2)
    ROUND0(c, d, e, a, b, 3)
    ROUND0(b, c, d, e, a, 4)
    ROUND0(a, b, c, d, e, 5)
    ROUND0(e, a, b, c, d, 6)
    ROUND0(d, e, a, b, c, 7)
    ROUND0(c, d, e, a, b, 8)
    ROUND0(b, c, d, e, a, 9)
    ROUND0(a, b, c, d, e, 10)
    ROUND0(e, a, b, c, d, 11)
    ROUND0(d, e, a, b, c, 12)
    ROUND0(c, d, e, a, b, 13)
    ROUND0(b, c, d, e, a, 14)
    ROUND0(a, b, c, d, e, 15)
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

    a = 0U + a0 + a;
    b = 0U + b0 + b;
    c = 0U + c0 + c;
    d = 0U + d0 + d;
    e = 0U + e0 + e;

#undef ROUNDTAIL
#undef ROUND3
#undef ROUND2
#undef ROUND1
#undef ROUND0b
#undef ROUND0
#undef SCHEDULE
#undef ROTL32

    return a == h0 && b == h1 && c == h2 && d == h3 && e == h4;
}
